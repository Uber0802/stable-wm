## Installation

```bash
conda create -n stable-wm python=3.10 -y
conda activate stable-wm

conda install -c conda-forge swig
pip install 'stable-worldmodel[all]'
pip install -e . --no-deps
```

## Environment Setup
```bash
conda activate stable-wm
export STABLEWM_HOME=/mnt/tank/uber/stable-wm/dataset
export MUJOCO_GL=egl
```

## Dataset Collection

```bash
cd scripts/data

python collect_cube.py env_type=double num_traj=200 world.num_envs=5 multiview=True \
  chain_tasks=false terminate_at_goal=true world.max_episode_steps=200
```

Output → `$STABLEWM_HOME/datasets/ogbench/cube_double_singleview_expert.lance`

