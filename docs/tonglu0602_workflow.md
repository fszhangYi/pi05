# Tonglu 2026-06-02 Three-View Workflow

This workflow uses:

- raw episodes: `/home/znyyb/hww/vla/tonglu0602/raw_data/<episode_id>/`
- annotations: `/home/znyyb/hww/vla/tonglu0602/annotation/<episode_id>.txt`
- images: `rgb_chest_<frame>.jpg`, `rgb_top_<frame>.jpg`, `rgb_wrist_2_<frame>.jpg`
- config: `configs/pi05_tonglu0602_example.yaml`

## Annotation Rules

Each annotation file has 12 numeric rows. The converter reads:

- row 1, column 2: start frame
- row 3, column 2: end frame
- row 11, column 2: grid row
- row 12, column 2: grid column

The frame range is inclusive: `start_frame <= frame <= end_frame`.

The task prompt for each sliced episode is:

```text
pick the workpiece from the cardboard box in order and place it in the {row} row, {column} column of the grid tray
```

## State And Action

State is Cartesian pose plus gripper observation:

```text
[x, y, z, rx, ry, rz, gripper_obs]
```

Action is recomputed from observations:

```text
T_action = inv(T_current) @ T_target
action = [dx, dy, dz, drx, dry, drz, target_gripper_obs]
```

The converter does not use `steps.json.actions.cartesian_position`. For this
dataset, `gripper_action_source: next_observation`, so the gripper command is
the next observed gripper value, with the final frame using its current value.

Because `rx` is often near `pi`, `normalize_rx_to_2pi: true` maps absolute state
`rx < 0` to `rx + 2*pi`. The same normalization is applied by the serving code
before policy inference and on the returned absolute next state.

## Validate On This Machine

This machine does not need training dependencies for the raw-data validation
path. Run:

```bash
cd /home/znyyb/hww/vla/pi05_jax_sft/pi05_jax_sft
USE_VENV=0 PYTHON_BIN=python3 DRY_RUN=1 \
  bash scripts/prepare_dataset.sh configs/pi05_tonglu0602_example.yaml
```

The dry run checks annotation parsing, frame slicing, three image streams,
state normalization, action recomputation, and prompt generation. It writes:

```text
artifacts/prepare/pi05_tonglu0602_jax_summary.json
```

## Convert On The Training Machine

Install the JAX/OpenPI environment first:

```bash
cd /path/to/pi05_jax_sft
USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
source .venv/bin/activate
```

Convert the dataset:

```bash
bash scripts/prepare_dataset.sh configs/pi05_tonglu0602_example.yaml
```

Compute normalization stats:

```bash
bash scripts/compute_norm_stats.sh configs/pi05_tonglu0602_example.yaml
```

Validate that the OpenPI training config can be built:

```bash
python -m pi05_jax_sft.train \
  --config configs/pi05_tonglu0602_example.yaml \
  --print-only
```

Train:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
bash scripts/train_8gpu.sh configs/pi05_tonglu0602_example.yaml
```

For 2 GPUs, edit the config:

```yaml
training:
  fsdp_devices: 2
  batch_size: 32
  num_train_steps: 40000
```

Then run:

```bash
bash scripts/train_2gpu.sh configs/pi05_tonglu0602_example.yaml
```

## Offline Evaluation

After a checkpoint exists:

```bash
bash scripts/evaluate_checkpoint.sh configs/pi05_tonglu0602_example.yaml 20000
```

Omit the step to use the latest checkpoint.

## Serving

For a legacy client that sends only one wrist image:

```bash
bash scripts/serve.sh configs/pi05_tonglu0602_example.yaml 20000
```

For a three-view client, send three framed JPEG payloads per step in this order:

```text
4B chest_len | chest JPEG |
4B top_len   | top JPEG   |
4B wrist_len | wrist JPEG |
28B state    | 4B refresh
```

Launch the server with:

```bash
TASK_PROMPT="pick the workpiece from the cardboard box in order and place it in the 2 row, 3 column of the grid tray" \
IMAGE_PROTOCOL=three-view \
bash scripts/serve.sh configs/pi05_tonglu0602_example.yaml 20000
```
