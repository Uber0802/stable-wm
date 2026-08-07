"""Goal-conditioned BC on a frozen encoder."""

import torch
from einops import rearrange
from torch import nn


class FrozenEncoder(nn.Module):
    """Frozen per-frame features from a world model (``wm_ckpt``) or a raw
    HuggingFace backbone (``backbone``), pooled to ``cls``, ``mean``, or
    ``patch`` (every token, keeping the spatial layout)."""

    POOLS = ('cls', 'mean', 'patch')

    def __init__(
        self,
        wm_ckpt: str | None = None,
        backbone: str | None = None,
        pool: str = 'cls',
    ) -> None:
        super().__init__()
        if (wm_ckpt is None) == (backbone is None):
            raise ValueError('pass exactly one of wm_ckpt / backbone')
        if pool not in self.POOLS:
            raise ValueError(f'pool must be one of {self.POOLS}, got {pool!r}')

        self.pool = pool
        self.projector = None

        if wm_ckpt is not None:
            from stable_worldmodel.wm.utils import load_pretrained

            wm = load_pretrained(wm_ckpt)
            self.encoder = wm.encoder
            self.projector = getattr(wm, 'projector', None)
        else:
            from stable_worldmodel.wm.prejepa.module import create_backbone

            self.encoder = create_backbone(backbone)

        self.requires_grad_(False)
        self.out_dim, self.num_tokens = self._probe()

    @torch.no_grad()
    def _probe(self) -> tuple[int, int]:
        self.eval()
        x = torch.zeros(
            1, 1, 3, 224, 224, device=next(self.encoder.parameters()).device
        )
        out = self.forward(x)
        n = int(out.shape[2]) if self.pool == 'patch' else 1
        return int(out.shape[-1]), n

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, D), or (B, T, P, D) when pool='patch'."""
        b = pixels.shape[0]
        x = rearrange(pixels, 'b t ... -> (b t) ...')
        x = x.to(next(self.encoder.parameters()).dtype)
        h = self.encoder(x, interpolate_pos_encoding=True).last_hidden_state

        if self.pool == 'cls':
            z = h[:, 0]
        elif self.pool == 'mean':
            z = h[:, 1:].mean(1)
        else:
            z = h[:, 1:]

        if self.projector is not None:
            z = self.projector(z)
        return rearrange(z, '(b t) ... -> b t ...', b=b)


def mlp_head(feat_dim, num_frames, action_dim, hidden_dim, depth, dropout=0.0):
    """Concatenate the frame descriptors and regress the action; each frame
    owns a fixed slice, so position needs no encoding."""
    dims = [feat_dim * num_frames] + [hidden_dim] * depth
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], action_dim))
    return nn.Sequential(*layers)


class TransformerHead(nn.Module):
    """Attend over every patch of every frame, then read out the action.

    Attention is permutation invariant, so the position embedding is what makes
    a patch's location legible; without it this degrades to mean pooling.
    """

    def __init__(self, dim, num_tokens, action_dim, depth=4, heads=6,
                 mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.readout = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.action = nn.Linear(dim, action_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, D) or (B, T, P, D) -> (B, action_dim)"""
        x = feats.flatten(1, 2) if feats.ndim == 4 else feats
        x = x.float() + self.pos
        x = torch.cat([self.readout.expand(len(x), -1, -1), x], dim=1)
        return self.action(self.norm(self.transformer(x)[:, 0]))


class LatentBC(nn.Module):
    """Frozen encoder + action head over ``history_size`` recent frames plus
    the goal.

    ``head_type='auto'`` picks a transformer for patch tokens and an MLP for
    pooled ones; naming it explicitly lets the two be compared on identical
    inputs, since the encoder already reports how many tokens it emits.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        history_size: int = 3,
        action_dim: int = 5,
        hidden_dim: int = 512,
        depth: int = 2,
        heads: int = 6,
        dropout: float = 0.0,
        head_type: str = 'auto',
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.history_size = history_size
        num_frames = history_size + 1

        if head_type == 'auto':
            head_type = 'transformer' if encoder.pool == 'patch' else 'mlp'
        self.head_type = head_type

        if head_type == 'transformer':
            self.head = TransformerHead(
                encoder.out_dim, encoder.num_tokens * num_frames, action_dim,
                depth=depth, heads=heads, dropout=dropout,
            )
        elif head_type == 'mlp':
            self.head = mlp_head(
                encoder.out_dim * encoder.num_tokens, num_frames, action_dim,
                hidden_dim, depth, dropout,
            )
        else:
            raise ValueError(f'unknown head_type {head_type!r}')
        self._history: torch.Tensor | None = None

    def apply_head(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, D) or (B, T, P, D) -> (B, action_dim)"""
        if self.head_type == 'transformer':
            return self.head(feats)
        return self.head(feats.flatten(1).float())

    def forward(
        self, pixels: torch.Tensor, goal_pixels: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(pixels)[:, -self.history_size:]
        zg = self.encoder(goal_pixels)[:, -1:]
        return self.apply_head(torch.cat([z, zg], dim=1))

    def reset(self) -> None:
        """Drop the rolling history (call between episodes)."""
        self._history = None

    def get_action(self, info: dict) -> torch.Tensor:
        """Act from a single observation, stacking history across calls.

        The evaluation harness only ever hands over the current frame, so the
        history the head was trained on is accumulated here; the first step
        repeats the current frame to fill the window.
        """
        z = self.encoder(info['pixels'])[:, -1:]

        if self._history is None or self._history.shape[0] != z.shape[0]:
            self._history = z.expand(-1, self.history_size, *z.shape[2:])
        self._history = torch.cat([self._history[:, 1:], z], dim=1)

        zg = self.encoder(info['goal_pixels'])[:, -1:]
        return self.apply_head(torch.cat([self._history, zg], dim=1))


__all__ = ['FrozenEncoder', 'LatentBC', 'TransformerHead', 'mlp_head']
