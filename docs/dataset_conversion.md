# 数据集转换说明

将 `act_robot` 原始 episode（`steps.json` + 多路 JPEG + 标注 txt）转换为 pi0.5 训练所需的 **LeRobot** 格式。

---

## 目录结构

### 输入（raw）

```text
~/autodl-tmp/act_robot/data/raw/<episode_id>/
├─ steps.json          # observations / actions
├─ metadata.json
├─ rgb_chest_<frame>.jpg
├─ rgb_top_<frame>.jpg
└─ rgb_wrist_2_<frame>.jpg
```

### 标注（annotation）

```text
~/autodl-tmp/act_robot/data/annotation/annotation/restored_txt/<episode_id>.txt
```

每个 txt 共 12 行，支持两种写法（空格或冒号）：

```text
1: 5        # 或  1 5
2: 120
3: 185
...
11: 2       # grid row
12: 1       # grid column
```

转换器读取：

| 行 | 含义 |
|---|---|
| 第 1 行第 2 列 | 起始帧（含） |
| 第 3 行第 2 列 | 结束帧（含） |
| 第 11 行第 2 列 | 托盘行号 |
| 第 12 行第 2 列 | 托盘列号 |

### 输出（LeRobot）

```text
~/autodl-tmp/pi05/data/lerobot/<repo_id>/
├─ data/chunk-*/episode_*.parquet
├─ meta/info.json
├─ meta/episodes.jsonl
└─ meta/tasks.jsonl
```

本机默认 `repo_id`：`company/act_robot_three_view`（全量）  
冒烟集（100 episodes）：`company/act_robot_three_view_smoke`，配置见 `configs/pi05_act_robot_smoke.yaml`

---

## 环境准备

需要 Python 3.10+，并安装：

- `numpy`, `scipy`, `Pillow`, `PyYAML`
- 项目内 vendored 的 `lerobot`（已在 `pi05/lerobot/`）
- `pandas`, `pyarrow`（LeRobot 写 parquet 用）

本机可用 conda 环境：

```bash
export PYTHON_BIN=/root/autodl-tmp/miniconda3/envs/lerobot/bin/python
```

若缺少依赖：

```bash
$PYTHON_BIN -m pip install scipy PyYAML pandas pyarrow
```

---

## 配置文件

本机示例：`configs/pi05_act_robot_local.yaml`

关键字段：

```yaml
data:
  repo_id: company/act_robot_three_view
  dataset_format: tonglu_annotation
  raw_root: /root/autodl-tmp/act_robot/data/raw
  annotation_root: /root/autodl-tmp/act_robot/data/annotation/annotation/restored_txt
  hf_lerobot_home: /root/autodl-tmp/pi05/data/lerobot
  chest_image_prefix: rgb_chest
  top_image_prefix: rgb_top
  wrist_image_prefix: rgb_wrist_2
  normalize_rx_to_2pi: true
  gripper_action_source: next_observation
  prompt_from_task: true
```

---

## 使用方法

### 1. 进入项目目录

```bash
cd ~/autodl-tmp/pi05
export PYTHON_BIN=/root/autodl-tmp/miniconda3/envs/lerobot/bin/python
```

### 2. 干跑验证（不写盘）

只检查标注解析、帧切片、图像路径、state/action 计算：

```bash
DRY_RUN=1 USE_VENV=0 bash scripts/prepare_dataset.sh configs/pi05_act_robot_local.yaml
```

输出摘要：`artifacts/prepare/pi05_act_robot_summary.json`

### 3. 冒烟转换（100 episodes，推荐先跑）

磁盘有限时先用冒烟集验证整条链路：

```bash
RESUME=0 COPY_ORIGINAL_IMAGES=1 IMAGE_LINK_MODE=hardlink \
USE_VENV=0 bash scripts/prepare_dataset.sh configs/pi05_act_robot_smoke.yaml
```

输出：`data/lerobot/company/act_robot_three_view_smoke/`

### 4. 全量转换

```bash
RESUME=0 COPY_ORIGINAL_IMAGES=1 IMAGE_LINK_MODE=hardlink \
USE_VENV=0 bash scripts/prepare_dataset.sh configs/pi05_act_robot_local.yaml
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `COPY_ORIGINAL_IMAGES` | `1` | 复用原始 JPEG，不解码重编码 |
| `IMAGE_LINK_MODE` | `hardlink` | `hardlink` / `symlink` / `copy` |
| `RESUME` | `1` | `1` 断点续转；全新转换设 `0` |
| `OVERWRITE_FLAG` | `--overwrite` | 配合 `RESUME=0` 覆盖已有数据集 |
| `EPISODE_START_INDEX` | `0` | 从第 N 个 episode 开始（排序后） |
| `EPISODE_END_INDEX` | 空 | 转换到第 N 个 episode 前停止 |
| `PROGRESS_INTERVAL` | `10` | 每 N 个 episode 打印进度 |

### 5. 直接调用 Python 模块

```bash
export PYTHONPATH="$PWD/src:$PWD/lerobot:$PYTHONPATH"
$PYTHON_BIN -m pi05_jax_sft.convert_company_dataset \
  --config configs/pi05_act_robot_local.yaml \
  --overwrite \
  --copy-original-images \
  --image-link-mode hardlink
```

常用参数：

```text
--dry-run                 只验证，不写 LeRobot 文件
--resume                  跳过已完成的 parquet episode
--repair-resume           修复不连续 tail 后再 resume
--validate-images-only    只解码图像做完整性检查
--episode-start-index N
--episode-end-index N
```

---

## 转换逻辑摘要

1. 扫描 `raw_root` 下与 `annotation_root/<id>.txt` 匹配的 episode
2. 按标注切帧 `[start, end]`
3. **State (7D)**：`[x, y, z, rx, ry, rz, gripper_obs]`（可选 `rx → [0, 2π)`）
4. **Action (7D)**：6D 相对位姿 `inv(T_t) @ T_{t+1}` + gripper（默认取下一帧观测）
5. 三路图像映射为 LeRobot keys：`chest_image` / `top_image` / `wrist_image`
6. Task prompt 写入 LeRobot task 字段

---

## 转换后检查

```bash
$PYTHON_BIN - <<'PY'
import json
from pathlib import Path
info = json.loads(Path("data/lerobot/company/act_robot_three_view/meta/info.json").read_text())
print("episodes:", info["total_episodes"])
print("frames:", info["total_frames"])
print("features:", list(info["features"].keys()))
PY
```

---

## 常见问题

**Q: 磁盘空间不足？**  
优先使用 `IMAGE_LINK_MODE=hardlink`（默认）。若仍不足，可删旧 checkpoint 或归档后再转。

**Q: 只有部分 episode 有标注？**  
转换器自动跳过无标注目录；只有 `raw/<id>/` 与 `restored_txt/<id>.txt` 同时存在才会转换。

**Q: 与 act_robot 自带 `convert_episodes.py` 的区别？**  
act_robot 输出 ACT 格式（`observation.images.*`）；本脚本输出 openpi/pi0.5 所需的 LeRobot keys 与 task prompt，二者不可混用。

**Q: 下一步？**  
转换完成后计算归一化统计：

```bash
bash scripts/compute_norm_stats.sh configs/pi05_act_robot_local.yaml
```
