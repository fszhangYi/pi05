# pi05_jax_sft

基于官方 [openpi](https://github.com/Physical-Intelligence/openpi) JAX 训练栈的 **pi0.5 微调框架**，面向公司内网单机多卡场景。

**原则：** 训练主循环、模型、checkpoint 全部复用 `external/openpi`；本项目只补数据处理、配置解析、推理服务三层工程代码。

---

## 项目结构

```text
pi05_jax_sft/
├─ configs/
│  ├─ pi05_company_example.yaml   # 8 卡全参微调
│  ├─ pi05_lora_example.yaml      # 8 卡 LoRA 微调
│  └─ pi05_tonglu0602_example.yaml # Tonglu 三视角数据
├─ external/openpi/               # 官方 openpi（不修改）
├─ requirements/openpi_jax.txt
├─ scripts/
│  ├─ check_project.sh
│  ├─ setup_env.sh
│  ├─ prepare_dataset.sh
│  ├─ compute_norm_stats.sh
│  ├─ train_8gpu.sh
│  ├─ train_2gpu.sh
│  ├─ evaluate_checkpoint.sh
│  ├─ serve.sh
│  └─ pi05_platform_train.sh
└─ src/pi05_jax_sft/
   ├─ project_config.py           # YAML → PipelineConfig
   ├─ runtime.py                  # build_train_config()
   ├─ company_policy.py           # CompanyWristInputs/Outputs
   ├─ convert_company_dataset.py  # 原始数据 → LeRobot
   ├─ compute_norm_stats.py
   ├─ train.py
   ├─ evaluate_checkpoint.py
   └─ serve.py                    # TCP 推理服务端
```

---

## 数据集格式

数据根目录：`/home/znyyb/hww/vla/cam_100_15`，共 552 个 episode，每个约 132 帧，控制频率 15 Hz。

```text
episode_XXXXXX/
├─ steps.json           observations.cartesian_position (T,6)  xyz + RPY Euler (xyz-extrinsic, rad)
│                       observations.gripper_position   [[g0..gT-1]]
│                       actions.gripper_position        [[a0..aT-1]]
├─ metadata.json        (仅相机内外参，task 统一由 YAML 的 task_name 指定)
├─ rgb_wrist_1_XXXXXX.jpg    wrist 相机（训练使用）
└─ rgb_rear_left_XXXXXX.jpg  后左相机（暂不使用）
```

**State（7D）**：`[x, y, z, rx, ry, rz, gripper_obs]`  
**Action（7D）**：`[dx, dy, dz, drx, dry, drz, gripper_cmd]`，其中 6D 姿态 = `se3_to_pose(inv(T_t) @ T_{t+1})`

Tonglu 2026-06-02 三视角数据见：

```bash
docs/tonglu0602_workflow.md
```

---

## 快速开始

### 1. 安装环境

```bash
cd /path/to/pi05_jax_sft

bash scripts/check_project.sh

# 安装到当前 Python 环境（内网 mirror）
USE_VENV=0 PYTHON_BIN=python3.12 bash scripts/setup_env.sh

# 如需虚拟环境
USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
source .venv/bin/activate
```

要求：Python 3.12 · CUDA 12 · JAX 0.5.3 · torch 2.7.0

### 2. 数据转换

```bash
bash scripts/prepare_dataset.sh configs/pi05_company_example.yaml
```

输出：`./data/lerobot/company/pi05_company_task/`

### 3. 计算归一化统计

```bash
bash scripts/compute_norm_stats.sh configs/pi05_company_example.yaml
```

输出：`./artifacts/assets/pi05_company_jax/company/pi05_company_task/norm_stats.json`

### 4. 训练

```bash
# 8 卡全参
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
bash scripts/train_8gpu.sh configs/pi05_company_example.yaml

# 8 卡 LoRA
bash scripts/train_8gpu.sh configs/pi05_lora_example.yaml

# 2 卡（需先将 config 中 fsdp_devices=2, batch_size=32, num_train_steps=40000）
bash scripts/train_2gpu.sh configs/pi05_company_example.yaml
```

Checkpoint 保存位置：`./artifacts/checkpoints/<project.name>/<exp_name>/<step>/`

### 5. 离线推理验证

```bash
bash scripts/evaluate_checkpoint.sh configs/pi05_company_example.yaml [step]
```

输出 `.npz` 到 `./artifacts/eval/`

### 6. 启动推理服务端

```bash
# 与 act_robot/serve.py 相同的 TCP 协议，客户端无需改动
bash scripts/serve.sh configs/pi05_company_example.yaml [step]

# temporal aggregation 模式
TEMPORAL_AGG=1 bash scripts/serve.sh configs/pi05_company_example.yaml
```

---

## 配置说明

### 全参微调（`pi05_company_example.yaml`）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data.task_name` | `"抓取镜头模组"` | 训练 & 推理 prompt |
| `model.discrete_state_input` | `true` | pi0.5 必须为 true，state 离散化进 prompt |
| `model.paligemma_variant` | `gemma_2b` | 全参 |
| `model.action_expert_variant` | `gemma_300m` | 全参 |
| `training.batch_size` | `128` | 8 卡全局 batch，per-GPU=16 |
| `training.num_train_steps` | `20000` | ≈35 epoch，552 ep × 132 帧 |
| `paths.base_checkpoint_path` | `/data/pretrained/pi05_base/params` | 本地预训练权重 |

### LoRA 微调（`pi05_lora_example.yaml`）

| 字段 | 值 | 说明 |
|---|---|---|
| `model.paligemma_variant` | `gemma_2b_lora` | PaliGemma 主干冻结 + LoRA (rank=16) |
| `model.action_expert_variant` | `gemma_300m` | Action Expert 全参 |
| `training.num_train_steps` | `30000` | ≈52 epoch |
| `training.peak_lr` | `1e-4` | LoRA 参数量小，lr 可高一些 |

### 2 卡调整项

```yaml
training:
  fsdp_devices: 2
  batch_size: 32        # per_gpu=16 不变
  num_train_steps: 40000  # 全参；LoRA 用 60000
```

---

## 核心实现说明

### Transform 链（训练）

```
LeRobot sample
  → PromptFromLeRobotTask
  → CompanyWristInputs        wrist_image → left_wrist_0_rgb；其余 mask=False
  → Normalize (quantile)      state/action 映射到 [-1, 1]
  → ResizeImages(224, 224)
  → TokenizePrompt(discrete_state_input=True)
      "Task: 抓取镜头模组, State: 128 64 ...;\nAction: "
  → PadStatesAndActions(32)   7D → 32D padding
```

### 推理服务协议（与 ACT 服务端兼容）

```
client → 4B img_len | N B wrist JPEG | 4B extra_len | M B extra | 28B state | 4B refresh
server → 28B next_state  (T_current @ T_action)
```

---

## 硬件要求

- 推荐：8 × A100 80G
- 最低：2 × A100 40G（LoRA 模式）
- 仅支持单机多卡；官方 openpi JAX trainer 不支持多机


PYTHONPATH=src python3 -m pi05_jax_sft.export_tensorrt \
  --config configs/pi05_tonglu0602_mlu_v2.yaml \
  --checkpoint-dir artifacts/checkpoints/pi05_tonglu0602_mlu/mlu_full_ft_v2 \
  --checkpoint-step 12000 \
  --output artifacts/trt/pi05_sample_fp16.ep \
  --device cuda:0 \
  --num-steps 10

PYTHONPATH=src python3 -m pi05_jax_sft.serve_cuda \
  --config pi05_jax_sft/configs/pi05_tonglu0602_example.yaml \
  --checkpoint-dir /path/to/checkpoint_root \
  --checkpoint-step 24000 \
  --port 5000 \
  --device cuda:0 \
  --trt-engine artifacts/trt/pi05_sample_fp16.ep
