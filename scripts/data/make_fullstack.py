"""Filter raw p_stack=1.0 collections down to full-tower episodes.

Reads a raw dataset, keeps only episodes that build a complete tower (all cubes
stacked in one column), truncates each to end at the moment the tower is first
completed, and writes a clean ``*_fullstack.lance`` whose final frame is the
full stack — ready to serve as the goal source for a full-stack eval.

Usage:
  STABLEWM_HOME=/mnt/tank/uber/stable-wm/dataset \
  python make_fullstack.py triple 3
  python make_fullstack.py quadruple 4
"""

import sys

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.data.format import get_format
from stable_worldmodel.data.utils import _episode_to_step_lists, get_cache_dir


def tower_height_at(p):
    """Max stacked-column height for cube positions ``p`` (ncube, 3)."""
    ncube = len(p)
    best = 0
    for base in range(ncube):
        h, cur = 1, base
        for _ in range(ncube):
            nxt = [
                j
                for j in range(ncube)
                if j != cur
                and abs(p[j][2] - p[cur][2] - 0.04) < 0.02
                and np.linalg.norm(p[j][:2] - p[cur][:2]) < 0.03
            ]
            if not nxt:
                break
            cur = nxt[0]
            h += 1
        best = max(best, h)
    return best


def first_full_tower_step(chunk, ncube):
    """Return the first timestep index with a complete tower, or -1."""
    pos = np.stack(
        [
            np.asarray(chunk[f'privileged/block_{i}_pos']).reshape(-1, 3)
            for i in range(ncube)
        ],
        axis=1,
    )  # (T, ncube, 3)
    for t in range(pos.shape[0]):
        if tower_height_at(pos[t]) >= ncube:
            return t
    return -1


def main(env_type, ncube):
    src_name = f'ogbench/cube_{env_type}_fullstack_raw.lance'
    dst_path = (
        get_cache_dir()
        / 'datasets'
        / f'ogbench/cube_{env_type}_fullstack.lance'
    )

    ds = swm.data.load_dataset(src_name, num_steps=1, frameskip=1)
    n = len(ds.lengths)
    kept, tower_steps = 0, []

    writer = get_format('lance').open_writer(dst_path, mode='overwrite')

    def episode_iter():
        nonlocal kept
        for ep in range(n):
            chunk = ds.load_episode(ep)
            t = first_full_tower_step(chunk, ncube)
            if t < 0:
                continue
            end = t + 1
            trunc = {
                k: (v[:end] if hasattr(v, '__len__') and len(v) >= end else v)
                for k, v in chunk.items()
            }
            kept += 1
            tower_steps.append(end)
            yield _episode_to_step_lists(trunc, end)

    with writer as w:
        w.write_episodes(episode_iter())

    ts = np.array(tower_steps) if tower_steps else np.array([0])
    print(
        f'{env_type}: kept {kept}/{n} full-tower episodes '
        f'(tower built by step min/mean/max = '
        f'{ts.min()}/{ts.mean():.0f}/{ts.max()}) -> {dst_path.name}'
    )


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
