# 训练说明

pi0.5 JAX 微调入口：`python -m pi05_jax_sft.train` → 官方 `external/openpi/scripts/train.py`。

## 前置条件

1. **JAX 环境**（Python 3.11/3.12 + CUDA 12）：

```bash
cd ~/autodl-tmp/pi05
USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
source .venv/bin/activate
```

2. **LeRobot 数据集**（见 [dataset_conversion.md](dataset_conversion.md)）

3. **归一化统计**：

```bash
bash scripts/compute_norm_stats.sh configs/pi05_act_robot_smoke.yaml
```

4. **预训练权重**：配置中 `paths.base_checkpoint_path` 指向 pi0.5 base，例如：

```text
./checkpoints/pi0.5_base/params
```

## 验证配置（不启动训练）

```bash
python -m pi05_jax_sft.train \
  --config configs/pi05_act_robot_smoke.yaml \
  --print-only
```

`train.py` 会在正式训练前检查：base checkpoint、norm_stats、LeRobot parquet 是否存在。

## 启动训练

单卡冒烟（`configs/pi05_act_robot_smoke.yaml` 已设 `fsdp_devices: 1`）：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

bash scripts/train_8gpu.sh configs/pi05_act_robot_smoke.yaml
```

多卡时修改 YAML 中 `training.fsdp_devices` / `batch_size`，或使用：

```bash
bash scripts/train_2gpu.sh configs/pi05_act_robot_smoke.yaml   # 会校验 GPU 数
```

所有训练脚本通过 `scripts/common_env.sh` 设置 `PYTHONPATH`（`src` + `lerobot` + `openpi`）。

## Checkpoint 输出

```text
artifacts/checkpoints/<project.name>/<exp_name>/<step>/
```

例如冒烟配置：`artifacts/checkpoints/pi05_act_robot_smoke/convert_smoke/100/`

## 常见问题

| 报错 | 处理 |
|---|---|
| `ModuleNotFoundError: flax/jax` | 先跑 `scripts/setup_env.sh` 安装 JAX 栈 |
| `Norm stats not found` | 先跑 `scripts/compute_norm_stats.sh` |
| `fsdp_devices > available GPUs` | 调小 YAML 中 `training.fsdp_devices` |
| `Base checkpoint not found` | 下载/软链 pi0.5 base 到 `checkpoints/pi0.5_base/params` |
