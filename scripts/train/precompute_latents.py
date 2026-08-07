"""Encode every frame once with a frozen encoder and cache the result.

The encoder never updates during BC training, so re-encoding the images each
epoch is wasted work. One pass here turns the rest of training into an MLP fit.

    python precompute_latents.py --wm-ckpt cube_single_lewm/weights_epoch_60.pt
    python precompute_latents.py --backbone facebook/dinov2-small --pool mean
"""

import argparse
import os

import numpy as np
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.wm.bc import FrozenEncoder

DATASET = 'galilai-group/ogb_cube_single'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wm-ckpt')
    ap.add_argument('--backbone')
    ap.add_argument('--pool', default='cls', choices=['cls', 'mean'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--num-workers', type=int, default=8)
    args = ap.parse_args()

    dataset = swm.data.load_dataset(
        DATASET,
        transform=spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source='pixels', target='pixels',
            ),
            spt.data.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        cache_dir=os.environ.get('LOCAL_DATASET_DIR', None),
        num_steps=1,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    print(f'{len(dataset)} frames', flush=True)

    enc = FrozenEncoder(
        wm_ckpt=args.wm_ckpt, backbone=args.backbone, pool=args.pool
    ).cuda().eval()
    print(f'encoder out_dim {enc.out_dim}', flush=True)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, prefetch_factor=4,
    )

    Z = np.empty((len(dataset), enc.out_dim), dtype=np.float16)
    i = 0
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for n, batch in enumerate(loader):
            px = batch['pixels'].cuda(non_blocking=True)
            z = enc(px)[:, 0].float().cpu().numpy()
            Z[i:i + len(z)] = z
            i += len(z)
            if n % 200 == 0:
                print(f'  {i}/{len(dataset)}', flush=True)

    names = set(dataset.column_names) | set(getattr(dataset, '_schema_names', ()))
    col = 'episode_idx' if 'episode_idx' in names else 'ep_idx'
    np.savez(
        args.out,
        z=Z[:i],
        episode=dataset.get_col_data(col),
        step=dataset.get_col_data('step_idx'),
        action=dataset.get_col_data('action'),
    )
    print(f'saved {i} x {enc.out_dim} -> {args.out}')


if __name__ == '__main__':
    main()
