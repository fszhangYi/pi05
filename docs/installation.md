# 环境安装说明

本文档记录在 AutoDL 实例上安装 pi05 JAX 训练环境的完整流程，基于 2026-08-28 的实际安装经验。

---

## 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.11 或 3.12（推荐 3.12） |
| CUDA | 12.x |
| GPU | 单卡即可冒烟；多卡按 YAML 中 `fsdp_devices` 配置 |
| 磁盘 | 虚拟环境约 **7–8 GiB**；数据盘建议预留 **20 GiB+** |
| 系统盘 | 仅 `/` overlay ~30 GiB，安装时把 pip 缓存/临时目录放到数据盘 |

实测环境：

- GPU：NVIDIA GeForce RTX 5090 32G
- Python：3.12.3
- JAX：0.5.3（CUDA 12）

---

## 一键安装

```bash
cd ~/autodl-tmp/pi05

# 建议：缓存与临时文件放数据盘，避免系统盘满
export PIP_CACHE_DIR=~/autodl-tmp/pip-cache
export TMPDIR=~/autodl-tmp/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# 创建项目虚拟环境（推荐）
USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh 2>&1 | tee logs/setup_env.log
```

安装完成后激活：

```bash
source ~/autodl-tmp/pi05/.venv/bin/activate
```

---

## 安装脚本做了什么

`scripts/setup_env.sh` 按顺序执行：

1. **创建 `.venv`**（`USE_VENV=1` 时）
2. **升级 pip / setuptools / wheel / hatchling**
3. **安装 JAX 栈**：`pip install -r requirements/openpi_jax.txt`
   - 核心：`jax[cuda12]==0.5.3`、`flax==0.10.2`、`torch==2.7.0`
   - `numpy>=1.22.4,<2.0.0`（必须 `<2`，openpi 不兼容 numpy 2.x）
   - `orbax-checkpoint==0.11.13`、`transformers==4.53.2` 等
4. **安装 openpi-client**（带依赖）
5. **安装 vendored lerobot**：`lerobot/`（`--no-deps`）
6. **安装 openpi 本体**：`external/openpi/`（`--no-deps`）
7. **安装 pi05 包**：`src/`（`--no-deps`）
8. **补装传递依赖**：`chex`、`pytest`（openpi import 链需要，但 `--no-deps` 不会自动装上）

---

## 依赖清单

完整列表见 `requirements/openpi_jax.txt`。关键 pin：

```text
jax[cuda12]==0.5.3
flax==0.10.2
numpy>=1.22.4,<2.0.0
torch==2.7.0
orbax-checkpoint==0.11.13
chex>=0.1.86
pytest>=8.0.0
```

---

## 安装后验证

### 1. JAX / GPU

```bash
source ~/autodl-tmp/pi05/.venv/bin/activate
python -c "import jax; print(jax.__version__, jax.devices())"
```

期望输出类似：

```text
0.5.3 [CudaDevice(id=0)]
```

### 2. 训练配置（不启动训练）

```bash
cd ~/autodl-tmp/pi05
source .venv/bin/activate
bash scripts/train_8gpu.sh configs/pi05_act_robot_smoke.yaml --print-only
# 或直接：
python -m pi05_jax_sft.train --config configs/pi05_act_robot_smoke.yaml --print-only
```

期望输出 JSON，包含 `assets_dir`、`checkpoint_dir`、`repo_id` 等字段。

> 训练脚本通过 `scripts/common_env.sh` 自动设置 `PYTHONPATH`（`src` + `lerobot` + `openpi`）。

---

## 安装前还需准备的内容

环境装好后，训练还需要：

| 步骤 | 命令 / 路径 |
|---|---|
| 数据转换 | 见 [dataset_conversion.md](dataset_conversion.md) |
| 归一化统计 | `bash scripts/compute_norm_stats.sh configs/pi05_act_robot_smoke.yaml` |
| 预训练权重 | `./checkpoints/pi0.5_base/params`（可软链已有权重目录） |

预训练权重软链示例：

```bash
mkdir -p ~/autodl-tmp/pi05/checkpoints
ln -sfn /path/to/pi0.5_base ~/autodl-tmp/pi05/checkpoints/pi0.5_base
```

---

## 实际安装中遇到的问题

### 1. 数据盘满导致转换/训练失败

- **现象**：`No space left on device`
- **处理**：清理重复 tar/旧数据集；冒烟集只转 100 episodes；图片用 `IMAGE_LINK_MODE=hardlink`

### 2. 单独装 `chex` 会把 JAX 升级到 0.11.x

- **现象**：`orbax-checkpoint` 报 `DeviceLocalLayout` 不存在
- **原因**：新版 chex 拉高了 jax/jaxlib 版本，与 openpi 要求的 `0.5.3` 冲突
- **处理**：按 pin 重装：

```bash
pip install "jax[cuda12]==0.5.3" "numpy>=1.22.4,<2.0.0" "orbax-checkpoint==0.11.13" "chex>=0.1.86"
```

**不要**在 JAX 0.5.3 环境上无约束地 `pip install chex`。

### 3. `ModuleNotFoundError: chex` / `pytest`

- **原因**：openpi 用 `--no-deps` 安装，部分 import 链上的包未自动安装
- **处理**：`setup_env.sh` 末尾已补装；手动安装：

```bash
pip install "chex>=0.1.86" "pytest>=8.0.0"
```

### 4. 计算 norm_stats 缺 `numpydantic`

- **现象**：`compute_norm_stats.sh` 报 `No module named 'numpydantic'`
- **处理**：已在 `openpi_jax.txt` 中；若单独用其他 Python 环境跑转换脚本，需自行 `pip install numpydantic pydantic`

### 5. torch 版本警告

- openpi 元数据写 `torch==2.7.1`，requirements 固定 `2.7.0`
- 实际安装中 **2.7.0 可正常 `--print-only` 验证**；若遇兼容问题可尝试升到 2.7.1

---

## 不使用虚拟环境（USE_VENV=0）

```bash
USE_VENV=0 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
```

会安装到当前 Python 环境，适合已有干净 conda 环境的训练机。

---

## 内网 / 镜像源

安装前可指定 pip 源：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
```

---

## 磁盘占用参考

| 路径 | 大小（本机实测） |
|---|---|
| `.venv/` | ~7.4 GiB |
| 冒烟 LeRobot 数据集 | ~12 GiB（100 episodes，图像嵌 parquet） |
| `pip-cache/` | 视缓存而定 |

---

## 相关文档

- [dataset_conversion.md](dataset_conversion.md) — 数据转换
- [training.md](training.md) — 训练启动
- [tonglu0602_workflow.md](tonglu0602_workflow.md) — Tonglu 原始数据格式说明
