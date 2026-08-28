# Issues / 踩坑记录

记录本仓库在 AutoDL / 单机训练中遇到的问题、根因与修复。

---

## ISSUE-001：HF / 临时文件默认写到系统盘导致 `No space left on device`

**状态：** 已修复（`scripts/common_env.sh`）  
**日期：** 2026-08-28

### 现象

训练或首次 `LeRobotDataset` 加载时报：

```text
OSError: [Errno 28] No space left on device
```

出现在 HuggingFace `datasets` 把 parquet 重建为 Arrow 的阶段（日志里常见 `Generating train split`）。

容易误判成「内存爆了」或「数据盘不够」。实际：

- 报错是 **磁盘 ENOSPC**，不是 OOM（OOM 一般是 `CUDA OutOfMemory` / `Killed` / JAX `RESOURCE_EXHAUSTED`）
- **数据盘** `/root/autodl-tmp` 往往还有上百 G 空闲
- 真正写满的是 **系统盘 overlay（`/`，约 30G）或 `/tmp`**

### 根因

未设置环境变量时，库默认路径在系统盘上：

| 变量 | 默认位置 | 挂载 |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | overlay |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | overlay |
| `TMPDIR` | `/tmp` | overlay |
| `PIP_CACHE_DIR` | 常在 `~/.cache/pip` | overlay |

冒烟集 LeRobot 数据约 **12G**（图嵌在 parquet 里）。`datasets` 加载时会再生成一份 **~12G Arrow 缓存**。系统盘只剩几 G～十几 G 时就会炸。

这不是「故意放系统盘」，而是 **默认行为**；AutoDL 的大盘是 `/root/autodl-tmp`，库不会自动用它。

### 修复

`scripts/common_env.sh` 在所有训练 / 转换 / 安装脚本里统一设置：

```bash
# 默认（存在且可写时）
PI05_DATA_DISK=/root/autodl-tmp
PI05_CACHE_ROOT=/root/autodl-tmp/pi05-cache

HF_HOME=$PI05_CACHE_ROOT/hf
HF_DATASETS_CACHE=$HF_HOME/datasets
TMPDIR=$PI05_CACHE_ROOT/tmp
PIP_CACHE_DIR=$PI05_CACHE_ROOT/pip
OPENPI_DATA_HOME=$PI05_CACHE_ROOT/openpi
```

可用环境变量覆盖：`PI05_DATA_DISK`、`PI05_CACHE_ROOT`、或直接覆盖 `HF_HOME` / `TMPDIR` 等。

已接入：`train_*.sh`、`prepare_dataset.sh`、`compute_norm_stats.sh`、`setup_env.sh`、`evaluate_checkpoint.sh` 等（凡 source `common_env.sh` 的脚本）。

### 空间参考（冒烟 100 ep）

| 项 | 约占用 |
|---|---|
| LeRobot 数据 | 12G |
| HF Arrow 缓存（副本） | +12G |
| `.venv` | ~8G |
| 训练 checkpoint | 5–15G |
| **数据盘建议再留** | **≥20G**（冒烟）；全量 951 ep 建议 **≥100G** |

### 相关

- 数据转换见 `docs/dataset_conversion.md`
- 安装见 `docs/installation.md`
- 训练见 `docs/training.md`

---

## ISSUE-002：`datasets` 新版与 vendored lerobot 的 `torch.stack(Column)` 不兼容

**状态：** 已修复（`lerobot/.../lerobot_dataset.py`）  
**日期：** 2026-08-28

### 现象

```text
TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column
```

出在 `LeRobotDataset._query_hf_dataset`。

### 根因

`datasets>=3/5` 对 `dataset.select(...)[key]` 返回 `Column`，旧 lerobot 直接 `torch.stack(...)` 假设是 list of Tensor。

### 修复

增加 `_stack_hf_column()`：先 `to_pylist` / `to_list`，再 `torch.as_tensor` + `stack`。

---

## ISSUE-003：训练依赖未随 `--no-deps` 安装完整

**状态：** 已修复（`requirements/openpi_jax.txt` + `setup_env.sh` 补装）  
**日期：** 2026-08-28

### 现象

`ModuleNotFoundError: datasets` / `torchvision` / `chex` / `pytest` / `numpydantic` 等。

### 根因

`setup_env.sh` 对 `lerobot` / `openpi` 使用 `--no-deps`，传递依赖不会自动装上。

### 修复

- `openpi_jax.txt` 增加 `chex`、`pytest`、`numpydantic` 等
- `setup_env.sh` 末尾补装 `chex`、`pytest`
- 训练环境另需：`datasets`、`torchvision`（与 `torch==2.7.0` 匹配的 `torchvision==0.22.0`）等；后续可继续收进 requirements
