## Installation

```bash
conda create -n stable-wm python=3.10 -y
conda activate stable-wm

conda install -c conda-forge swig
pip install 'stable-worldmodel[all]'
pip install -e . --no-deps         
```

## Environment

```bash
conda activate stable-wm
export STABLEWM_HOME=./stable-wm/dataset
export MUJOCO_GL=egl
```

## 1. Collect data

```bash
cd scripts/data

python collect_cube.py \
  env_type=double p_stack=1.0 chain_tasks=true terminate_at_goal=false \
  multiview=false num_traj=1000 world.num_envs=5 \
  world.max_episode_steps=301 \
  dataset_file=ogbench/cube_double_fullstack_raw.lance mode=overwrite

python make_fullstack.py double 2
```

Outputs (under `$STABLEWM_HOME/datasets/ogbench/`):
- `cube_double_fullstack_raw.lance` — all raw play episodes
- `cube_double_fullstack.lance` — full-tower episodes, final frame = the tower


## 2. Train DINO-WM (PreJEPA)

Frozen DINOv2-small encoder + causal transformer predictor. `frameskip=1`
(one model step = one sim step); drop proprio for the single-view setup.

```bash
cd scripts/train

python prejepa.py \
  dataset_name=ogbench/cube_double_fullstack_raw.lance \
  output_model_name=cube_double_dinov2_small \
  frameskip=1 \
  '~wm.encoding.proprio'
```

Checkpoint → `$STABLEWM_HOME/checkpoints/cube_double_dinov2_small/`
(train on `_raw` for more trajectories; eval always uses the filtered
`_fullstack` set as the goal source).

## 3. Evaluate with MPC

CEM planner rolls action sequences through the world model and scores the final
predicted latent against the goal latent (`goal_mse`). `action_block` must equal
the training `frameskip`.

```bash
cd scripts/plan

python eval_wm.py --config-name cube_double \
  policy=cube_double_dinov2_small
```

### Eval settings (`config/cube_double.yaml`)

- `goal_at_end: true` — goal is each episode's final frame (the completed tower).
- `goal_offset_steps: N` — start `N` steps before the goal. Small `N` = easy
  (finish an almost-done tower); large `N` = hard.
- `init_first_goal_last: true` — ignore `goal_offset`, start from each episode's
  first frame and target the last: build the tower from scratch.
- `eval_budget` — max env steps per episode; must be `>= horizon * action_block`.
