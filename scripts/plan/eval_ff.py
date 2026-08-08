"""Script to evaluate a feedforward policy on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'


import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


def img_transform():
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
        ]
    )
    return transform


def episode_col(dataset):
    """Lance hides its index columns from ``column_names`` and exposes them
    only via ``_schema_names``; consult both."""
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world, image_shape=(224, 224), render_mode='rgb_array'
    )

    # create the transform
    transform = {
        'pixels': img_transform(),
        'goal': img_transform(),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    def fit_scaler(col):
        scaler = preprocessing.StandardScaler()
        scaler.fit(dataset.get_col_data(col))
        return scaler

    policy = cfg.get('policy', 'random')
    if policy == 'random':
        policy = swm.policy.RandomPolicy()
    else:
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to('cuda').eval()
        model.requires_grad_(False)

        process = {}
        if getattr(model, 'action_norm', 'standard') != 'none':
            process['action'] = fit_scaler('action')
        if 'proprio' in dataset.column_names:
            process['proprio'] = process['goal_proprio'] = fit_scaler('proprio')

        policy = swm.policy.FeedForwardPolicy(
            model=model, process=process, transform=transform
        )

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # concurrent seeds share results_path and would overwrite each other's
    # env_*.mp4, so give each run its own directory when asked
    video_dir = cfg.output.get('video_dir', None)
    video_path = Path(video_dir) if video_dir else results_path
    video_path.mkdir(parents=True, exist_ok=True)

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = episode_col(dataset)
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row

    n_hold = cfg.eval.get('holdout_episodes', 0)
    if n_hold:
        # policies are trained on this same dataset, so restrict the start
        # states to the episodes their training reserved
        held = swm.data.holdout_episodes(
            len(ep_indices), n_hold, cfg.eval.get('holdout_seed', 42)
        )
        valid_mask &= np.isin(dataset.get_col_data(col_name), held)
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
    eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video=video_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
