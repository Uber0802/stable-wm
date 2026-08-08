"""Train the BC head on precomputed latents.

The encoder is frozen, so its output is cached once by precompute_latents.py and
the whole dataset lives on the GPU as a single tensor -- no image decoding, no
DataLoader. Saves a LatentBC checkpoint that scripts/plan/eval_ff.py can load.

    python latent_bc_cached.py --latents ../../dataset/latents_lewm.npz \
        --wm-ckpt cube_single_lewm/weights_epoch_60.pt --out cube_lewm_bc
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.wm.bc import FrozenEncoder, LatentBC
from stable_worldmodel.wm.utils import save_pretrained

HISTORY = 3


def build_index(episode, step):
    """Anchor rows with a full history behind them and a future ahead.

    Returns the sort order, the anchor rows, the last row of each anchor's
    episode, and the episode id each anchor belongs to.
    """
    order = np.lexsort((step, episode))
    ep = episode[order]
    starts = np.searchsorted(ep, np.unique(ep), 'left')
    ends = np.append(starts[1:], len(ep))
    anchors, ep_end, ep_id = [], [], []
    for eid, s, e in zip(np.unique(ep), starts, ends):
        if e - s <= HISTORY:
            continue
        idx = np.arange(s + HISTORY - 1, e - 1)
        anchors.append(idx)
        ep_end.append(np.full(len(idx), e - 1))
        ep_id.append(np.full(len(idx), eid))
    return (order, np.concatenate(anchors), np.concatenate(ep_end),
            np.concatenate(ep_id))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents', required=True)
    ap.add_argument('--wm-ckpt')
    ap.add_argument('--backbone')
    ap.add_argument('--pool', default='cls')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--head-type', default='auto',
                    choices=['auto', 'mlp', 'transformer'])
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init-from',
                    help='checkpoint whose head weights start this run')
    ap.add_argument('--holdout-episodes', type=int, default=400)
    ap.add_argument('--holdout-seed', type=int, default=42)
    ap.add_argument('--action-norm', default='none',
                    choices=['none', 'standard'])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.latents)
    z_raw, episode, step, action = d['z'], d['episode'], d['step'], d['action']
    print(f'{len(z_raw)} frames, dim {z_raw.shape[1]}', flush=True)

    action = np.nan_to_num(action, nan=0.0)
    if args.action_norm == 'standard':
        action = preprocessing.StandardScaler().fit_transform(action)

    order, anchors, ep_end, anchor_ep = build_index(episode, step)
    z = torch.from_numpy(z_raw[order]).float().cuda()
    a = torch.from_numpy(action[order]).float().cuda()
    anchors_t = torch.from_numpy(anchors).cuda()
    ep_end_t = torch.from_numpy(ep_end).cuda()

    # split by episode, not by anchor: neighbouring anchors share two of their
    # three history frames, so a random anchor split leaks straight into val
    held = swm.data.holdout_episodes(
        len(np.unique(episode)), args.holdout_episodes, args.holdout_seed
    )
    is_val = torch.from_numpy(np.isin(anchor_ep, held)).cuda()
    g = torch.Generator(device='cuda').manual_seed(args.seed)
    val_idx = torch.nonzero(is_val, as_tuple=True)[0]
    train_idx = torch.nonzero(~is_val, as_tuple=True)[0]
    print(f'{len(held)} episodes held out | train {len(train_idx)}, '
          f'val {len(val_idx)}', flush=True)

    encoder_cfg = {
        'wm_ckpt': args.wm_ckpt, 'backbone': args.backbone, 'pool': args.pool,
    }
    model = LatentBC(
        encoder=FrozenEncoder(**encoder_cfg),
        history_size=HISTORY, action_dim=a.shape[1],
        hidden_dim=args.hidden_dim, depth=args.depth, heads=args.heads,
        head_type=args.head_type,
    ).cuda()
    if args.init_from:
        src = torch.load(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints')
            / args.init_from, map_location='cuda',
        )
        model.head.load_state_dict(
            {k[len('head.'):]: v for k, v in src.items() if k.startswith('head.')}
        )
        print(f'head initialised from {args.init_from}', flush=True)

    head = model.head
    print(f'{model.head_type} head, '
          f'{sum(p.numel() for p in head.parameters()):,} params', flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    steps = args.epochs * (len(train_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def batch(sel, train):
        i = anchors_t[sel]
        hist = torch.stack([z[i - k] for k in reversed(range(HISTORY))], 1)
        if train:
            r = torch.rand(len(i), device='cuda', generator=g)
            j = (i + 1 + (r * (ep_end_t[sel] - i).float()).long()).clamp(
                max=len(z) - 1
            )
        else:
            j = ep_end_t[sel]
        return torch.cat([hist, z[j].unsqueeze(1)], dim=1), a[i]

    for ep in range(args.epochs):
        head.train()
        sel = train_idx[torch.randperm(len(train_idx), generator=g,
                                       device='cuda')]
        tot = nb = 0
        for k in range(0, len(sel) - args.batch_size + 1, args.batch_size):
            x, y = batch(sel[k:k + args.batch_size], True)
            loss = F.mse_loss(model.apply_head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            vx, vy = batch(val_idx, False)
            vloss = F.mse_loss(model.apply_head(vx), vy).item()
        print(f'epoch {ep + 1}/{args.epochs}  train {tot / nb:.5f}  '
              f'val {vloss:.5f}', flush=True)

    cfg = OmegaConf.create({
        '_target_': 'stable_worldmodel.wm.bc.LatentBC',
        'encoder': {
            '_target_': 'stable_worldmodel.wm.bc.FrozenEncoder',
            **encoder_cfg,
        },
        'history_size': HISTORY, 'action_dim': int(a.shape[1]),
        'hidden_dim': args.hidden_dim, 'depth': args.depth,
        'heads': args.heads, 'head_type': model.head_type,
        'action_norm': args.action_norm,
    })
    save_pretrained(model, run_name=args.out, config=cfg,
                    filename=f'weights_epoch_{args.epochs}.pt')
    print(f'saved -> {args.out}/weights_epoch_{args.epochs}.pt')


if __name__ == '__main__':
    main()
