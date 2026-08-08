"""BC over every patch token of a frozen encoder.

Patch tokens are 512x larger than a pooled descriptor -- 395 GB for the full
dataset -- so they are never cached. Each rank re-encodes a freshly sampled
batch of episodes every epoch instead, which covers far more of the dataset
than any single shard would fit.

    torchrun --nproc_per_node=4 patch_bc.py --out cube_dino_patch_bc
"""

import argparse
import os

import numpy as np
import stable_pretraining as spt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.wm.bc import FrozenEncoder, LatentBC
from stable_worldmodel.wm.utils import save_pretrained

DATASET = 'galilai-group/ogb_cube_single'
HISTORY = 3
EP_LEN = 201


def log(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def get_dataset():
    return swm.data.load_dataset(
        DATASET,
        transform=spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source='pixels', target='pixels',
            ),
            spt.data.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        cache_dir=os.environ.get('LOCAL_DATASET_DIR', None),
        num_steps=1, frameskip=1,
        keys_to_load=['pixels', 'action'], keys_to_cache=['action'],
    )


@torch.no_grad()
def encode_episodes(encoder, dataset, episodes, batch_size, workers):
    """Encode whole episodes so each frame passes the backbone exactly once."""
    rows = np.concatenate([
        np.arange(e * EP_LEN, (e + 1) * EP_LEN) for e in episodes
    ])
    loader = DataLoader(
        Subset(dataset, rows.tolist()), batch_size=batch_size,
        num_workers=workers, shuffle=False, pin_memory=True, prefetch_factor=4,
    )
    out, seen = None, 0
    with torch.autocast('cuda', torch.bfloat16):
        for batch in loader:
            z = encoder(batch['pixels'].cuda(non_blocking=True))[:, 0]
            if out is None:
                out = torch.empty(
                    (len(rows), *z.shape[1:]), dtype=torch.float16,
                    device='cuda',
                )
            out[seen:seen + len(z)] = z.half()
            seen += len(z)
            if seen % (batch_size * 50) == 0:
                log(f'  encoded {seen}/{len(rows)}')
    return out.view(len(episodes), EP_LEN, *out.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', default='facebook/dinov2-small')
    ap.add_argument('--wm-ckpt')
    ap.add_argument('--pool', default='patch')
    ap.add_argument('--episodes-per-rank', type=int, default=700)
    ap.add_argument('--val-episodes', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-resume', action='store_true')
    args = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(args.seed)

    dataset = get_dataset()
    n_ep = len(dataset) // EP_LEN
    actions = torch.from_numpy(
        np.nan_to_num(dataset.get_col_data('action'), nan=0.0)
    ).float().view(n_ep, EP_LEN, -1)

    # validation episodes are held out once and kept, so val loss is comparable
    # across epochs even though the training episodes are resampled
    g = np.random.default_rng(args.seed)
    order = g.permutation(n_ep)
    val_pool, train_pool = order[: args.val_episodes], order[args.val_episodes:]
    val_eps = val_pool[rank::world]
    log(f'{n_ep} episodes: {len(train_pool)} train pool, '
        f'{len(val_pool)} held out | {args.episodes_per_rank} resampled per '
        f'rank per epoch, {world} ranks')

    model = LatentBC(
        encoder=FrozenEncoder(wm_ckpt=args.wm_ckpt,
                              backbone=args.backbone, pool=args.pool),
        history_size=HISTORY, action_dim=actions.shape[-1],
        depth=args.depth, heads=args.heads,
    ).cuda()
    log(f'head params {sum(p.numel() for p in model.head.parameters()):,} | '
        f'tokens per sample {model.encoder.num_tokens * (HISTORY + 1)}')

    ckpt_dir = get_cache_dir(sub_folder='checkpoints') / args.out
    weights = ckpt_dir / f'weights_epoch_{args.epochs}.pt'
    state_path = ckpt_dir / 'train_state.pt'
    state = None
    if weights.exists() and state_path.exists() and not args.no_resume:
        model.load_state_dict(torch.load(weights, map_location='cuda'))
        state = torch.load(state_path, map_location='cuda')
        log(f'resuming from epoch {state["epoch"]}')

    def encode(eps):
        return (
            encode_episodes(
                model.encoder, dataset, eps, args.encode_batch,
                args.num_workers,
            ),
            actions[eps].cuda(),
        )

    z_val, a_val = encode(val_eps)
    log(f'val cache {tuple(z_val.shape)} = {z_val.numel() * 2 / 1e9:.1f} GB')

    head = DistributedDataParallel(
        model.head, device_ids=[int(os.environ['LOCAL_RANK'])]
    )
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    gen = torch.Generator(device='cuda').manual_seed(args.seed + rank)
    start_epoch = state['epoch'] if state else 0

    def anchors(n_eps):
        return torch.stack(torch.meshgrid(
            torch.arange(n_eps, device='cuda'),
            torch.arange(HISTORY - 1, EP_LEN - 1, device='cuda'),
            indexing='ij',
        ), -1).view(-1, 2)

    val_pairs = anchors(len(val_eps))
    steps = args.epochs * (
        len(anchors(args.episodes_per_rank)) // args.batch_size
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    if state:
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        # keep the episode stream going instead of replaying the same draws
        for _ in range(start_epoch):
            g.choice(train_pool, args.episodes_per_rank * world, False)

    def batch(z, a, sel, train):
        e, t = sel.unbind(-1)
        hist = torch.stack([z[e, t - k] for k in reversed(range(HISTORY))], 1)
        if train:
            r = torch.rand(len(e), device='cuda', generator=gen)
            j = t + 1 + (r * (EP_LEN - 1 - t).float()).long()
        else:
            j = torch.full_like(t, EP_LEN - 1)
        return torch.cat([hist, z[e, j].unsqueeze(1)], dim=1), a[e, t]

    def save(epoch):
        cfg = OmegaConf.create({
            '_target_': 'stable_worldmodel.wm.bc.LatentBC',
            'encoder': {
                '_target_': 'stable_worldmodel.wm.bc.FrozenEncoder',
                'wm_ckpt': args.wm_ckpt, 'backbone': args.backbone,
                'pool': args.pool,
            },
            'history_size': HISTORY,
            'action_dim': int(actions.shape[-1]),
            'depth': args.depth, 'heads': args.heads,
            'action_norm': 'none',
        })
        # one rolling file: a timeout then leaves a usable checkpoint where
        # the eval scripts already look for it
        save_pretrained(model, run_name=args.out, config=cfg,
                        filename=f'weights_epoch_{args.epochs}.pt')
        torch.save({'epoch': epoch, 'opt': opt.state_dict(),
                    'sched': sched.state_dict()}, state_path)

    for ep in range(start_epoch, args.epochs):
        eps = g.choice(train_pool, args.episodes_per_rank * world, False)
        eps = eps[rank::world]
        z_tr, a_tr = encode(eps)

        head.train()
        pairs = anchors(len(eps))
        sel = pairs[torch.randperm(len(pairs), generator=gen, device='cuda')]
        tot = nb = 0
        for k in range(0, len(sel) - args.batch_size + 1, args.batch_size):
            x, y = batch(z_tr, a_tr, sel[k:k + args.batch_size], True)
            loss = F.mse_loss(model.apply_head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1

        del z_tr, a_tr
        torch.cuda.empty_cache()

        head.eval()
        with torch.no_grad():
            vt = vn = 0
            for k in range(0, len(val_pairs), args.batch_size):
                x, y = batch(z_val, a_val, val_pairs[k:k + args.batch_size],
                             False)
                vt += F.mse_loss(model.apply_head(x), y).item()
                vn += 1
        stats = torch.tensor([tot / nb, vt / vn], device='cuda')
        dist.all_reduce(stats, dist.ReduceOp.AVG)
        log(f'epoch {ep + 1}/{args.epochs}  train {stats[0]:.5f}  '
            f'val {stats[1]:.5f}')
        if rank == 0:
            save(ep + 1)

    log(f'saved -> {args.out}/weights_epoch_{args.epochs}.pt')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
