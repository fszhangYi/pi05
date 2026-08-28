# 冒烟训练前改动记录

在启动 **10000 steps** 单卡冒烟测速之前，对本仓库与环境做过的修改汇总如下（截至 2026-08-28）。

---

## 1. 配置

| 文件 | 改动 |
|---|---|
| `configs/pi05_act_robot_smoke.yaml` | `num_train_steps: 10000`；`log_interval: 50`；`save_interval: 2000`；`keep_period: 5000`；`warmup/decay_steps` 对齐 10k；`fsdp_devices: 1`，`batch_size: 8`（单卡 5090） |
| `configs/pi05_act_robot_local.yaml` | `annotation_root` 指向 `.../restored_txt`；全量用，冒烟用 smoke 配置 |

---

## 2. 数据与标注

| 项 | 内容 |
|---|---|
| 标注解析 | `convert_company_dataset.py` 支持 `1: 5` 与 `1 5` 两种 txt 行格式 |
| 过滤 | Tonglu 模式只转换「有 annotation + steps.json」的 episode |
| 图像落盘 | `--image-link-mode hardlink`（默认），避免 JPEG 再占一份磁盘 |
| 冒烟数据 | 已转换 `company/act_robot_three_view_smoke`：100 ep / 14849 frames |
| norm_stats | 已写入 `artifacts/assets/pi05_act_robot_smoke/company/act_robot_three_view_smoke/norm_stats.json` |

---

## 3. 训练脚本 / 环境路径

| 文件 | 改动 |
|---|---|
| `scripts/common_env.sh` | 统一 `PYTHONPATH`；**强制** `HF_HOME` / `HF_DATASETS_CACHE` / `TMPDIR` / `PIP_CACHE_DIR` / `OPENPI_DATA_HOME` 落到数据盘（默认 `/root/autodl-tmp/pi05-cache`），避免写满 AutoDL 系统盘 overlay |
| `scripts/train_8gpu.sh` 等 | 去掉错误的 `/data1/...` PYTHONPATH；source `common_env.sh`；校验 `fsdp_devices ≤ GPU 数` |
| `scripts/prepare_dataset.sh` / `setup_env.sh` / `compute_norm_stats.sh` / `evaluate_checkpoint.sh` | 均 source `common_env.sh` |
| `src/pi05_jax_sft/train.py` | 恢复 CLI；`_preflight` 检查 base ckpt / norm_stats / parquet |
| `src/pi05_jax_sft/runtime.py` | import 路径加入 `lerobot` |

---

## 4. 依赖与兼容性

| 项 | 内容 |
|---|---|
| 环境 | `USE_VENV=1` 安装 `.venv`（JAX 0.5.3 + CUDA12） |
| `requirements/openpi_jax.txt` | 增加 `chex`、`pytest`、`torchvision==0.22.0`、`datasets` 等 |
| `setup_env.sh` | `--no-deps` 之后补装 `chex`、`pytest` |
| lerobot | `_stack_hf_column`：兼容 `datasets` 返回的 `Column`（避免 `torch.stack(Column)` 报错） |

---

## 5. 权重与路径

| 项 | 内容 |
|---|---|
| base checkpoint | 软链 `checkpoints/pi0.5_base` → `hww/pi05_jax_sft/checkpoints/pi0.5_base` |
| 测速命令 | `USE_VENV=1 bash scripts/train_8gpu.sh configs/pi05_act_robot_smoke.yaml` |

---

## 6. 已知问题（详见 `docs/issues.md`）

- **ISSUE-001**：HF/tmp 默认写系统盘 → ENOSPC（已用 `common_env.sh` 修复）
- **ISSUE-002**：datasets Column 与旧 lerobot 不兼容（已修）
- **ISSUE-003**：`--no-deps` 缺包（已补）
- **ISSUE-004**：全量 `LeRobotDataset()` 调试会把图解进 RAM（~90GB）；正式训练用 DataLoader 按 batch 取，勿一次性物化

---

## 7. 测速预期（测前估计，以日志为准）

- GPU：1× RTX 5090 32GB，`CUDA_VISIBLE_DEVICES=0`
- 模式：LoRA + action expert；batch=8；三视角
- 首次含 XLA 编译可能较慢；稳态后看 `Step N` 间隔估算 sec/step，再外推 10000 steps 总时长
