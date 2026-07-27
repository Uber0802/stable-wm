import os
from pathlib import Path


os.environ.setdefault('MUJOCO_GL', 'glfw')
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='ogb')
def run(cfg: DictConfig):
    """Run parallel data collection script"""
    env_type = cfg.get('env_type', 'single')
    chain_tasks = cfg.get('chain_tasks', True)
    p_stack = cfg.get('p_stack', None)
    terminate_at_goal = cfg.get('terminate_at_goal', False)
    multiview = cfg.get('multiview', True)

    world = swm.World(
        'swm/OGBCube-v0',
        **cfg.world,
        env_type=env_type,
        multiview=multiview,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=terminate_at_goal,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy(chain_tasks=chain_tasks, p_stack=p_stack))

    view_tag = 'multiview' if multiview else 'singleview'
    dataset_file = (
        cfg.get('dataset_file')
        or f'ogbench/cube_{env_type}_{view_tag}_expert.lance'
    )
    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / dataset_file
    )

    mode = cfg.get('mode', 'append')
    writer = swm.data.format.get_format('lance').open_writer(
        out_path, mode=mode
    )

    world.collect(
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
        writer=writer,
    )

    logging.success('🎉🎉🎉 Completed data collection for ogbench cube 🎉🎉🎉')


if __name__ == '__main__':
    run()
