# Tonglu `mlu_full_ft*` 全参复现（PyTorch）

本文档对齐 **hww 服务端实际挂载的 checkpoint**（`artifacts/checkpoints/**/mlu_full_ft*`），不是仓库 README 主推的 JAX LoRA 冒烟路径。

## 证据（以 artifacts / serve 为准）

| 服务脚本 | 默认 YAML / exp | 保留 step |
|---|---|---|
| `serve.sh` | `mlu_full_ft_two_view` | 16000 |
| `serve_abs.sh` | `mlu_full_ft_abs_pose` | 12000 |
| `serve_rtc.sh` | `mlu_full_ft_filter` | 10000 |

共同特征：

- `paligemma_variant: gemma_2b`，`action_expert_variant: gemma_300m`（**双全参**，非 LoRA）
- `batch_size: 256`，`fsdp_devices: 8`
- 落盘：`model.safetensors` + `metadata.pt`（openpi **PyTorch** trainer）
- base：`pi05_base_pytorch`

硬件量级：**约 8 × 80GB 级 GPU**。单卡 32GB（如 AutoDL 5090）**不能**按原配置复现；本配方 **不会降配**。

## 本仓库配方文件

| 文件 | 对应 |
|---|---|
| `configs/pi05_tonglu0630_full_ft_two_view.yaml` | serve.sh |
| `configs/pi05_tonglu0630_full_ft_abs_pose.yaml` | serve_abs.sh |
| `configs/pi05_tonglu0602_full_ft_filter.yaml` | serve_rtc.sh |
| `scripts/train_tonglu_full_ft.sh` | `torchrun` → `pi05_jax_sft.train_pytorch` |

路径已改到 autodl 风格（`./artifacts`、`./checkpoints/pi05_base_pytorch`、`./data/lerobot`）。Tonglu 原始 `raw_root` / `annotation_root` 需在训练机上存在或改 YAML。

## 步骤

```bash
cd /root/autodl-tmp/pi05

# 1) 环境（与现有 setup 相同）
USE_VENV=1 bash scripts/setup_env.sh
source .venv/bin/activate

# 2) 放置 PyTorch base（目录内必须有 model.safetensors）
mkdir -p checkpoints/pi05_base_pytorch
# 将官方/已转换的 pi0.5 PyTorch 权重放到该目录

# 3) 数据 → LeRobot + norm
bash scripts/prepare_dataset.sh configs/pi05_tonglu0630_full_ft_two_view.yaml
bash scripts/compute_norm_stats.sh configs/pi05_tonglu0630_full_ft_two_view.yaml

# 4) 仅打印解析结果（不训练）
python -m pi05_jax_sft.train_pytorch \
  --config configs/pi05_tonglu0630_full_ft_two_view.yaml --print-only

# 5) 8 卡全参训练
bash scripts/train_tonglu_full_ft.sh configs/pi05_tonglu0630_full_ft_two_view.yaml
```

Checkpoint 输出：`artifacts/checkpoints/<project.name>/<exp_name>/<step>/`。

## Preflight 会硬失败的情况

- 缺少 `checkpoints/pi05_base_pytorch/model.safetensors`
- 无 LeRobot parquet / 无 `norm_stats.json`
- 可见 GPU 数 `< training.fsdp_devices`（默认 8）
- YAML 仍带 `_lora` variant（本入口拒绝）

## 与 LoRA 冒烟路径的关系

| | 全参复现（本页） | LoRA 冒烟 |
|---|---|---|
| 脚本 | `train_tonglu_full_ft.sh` | `train_8gpu.sh` |
| 后端 | PyTorch `train_pytorch` | JAX `train` |
| 配置 | `*_full_ft_*.yaml` | `pi05_act_robot_smoke.yaml` |
| 硬件 | ~8×80GB | 可单卡 32GB 双 LoRA |

单卡调试请走冒烟路径，不要改本配方的 `batch_size` / `fsdp_devices` 冒充复现。
