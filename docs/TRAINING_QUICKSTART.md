# Training Quickstart

这份文档说明如何从零开始启动当前 tool policy 训练流水线。

当前训练对象是：

```text
Trajectory Transformer tool policy
```

它训练 agent 什么时候调用：

```text
list_files / read_file / search_code / apply_patch / run_tests / inspect_failure / finish
```

当前不训练 patch generator。`apply_patch` 的具体 patch 内容仍来自 expert patch provider。

## 1. 环境要求

推荐训练机：

```text
Python >= 3.11
PyTorch >= 2.2
GPU: RTX 4090 24GB
```

安装依赖：

```bash
python -m pip install -e '.[train,dev]'
```

如果要使用 ReAct/API agent，再装：

```bash
python -m pip install -e '.[train,llm,dev]'
```

如果想在仓库内创建一个本地 venv，可以这样做：

```bash
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -U pip
.venv-train/bin/python -m pip install -e '.[train,dev]'
```

确认 torch 和 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

也可以直接使用仓库准备好的 GPU 预检脚本：

```bash
bash scripts/gpu_preflight.sh
```

如果你使用了上面的本地 venv，则显式指定：

```bash
PYTHON=.venv-train/bin/python bash scripts/gpu_preflight.sh
```

这个脚本会检查：

```text
1. 当前 Python 版本和解释器路径
2. /dev/nvidia* 是否可见
3. /proc/driver/nvidia 是否能看到内核驱动和 GPU
4. nvidia-smi 是否可用
5. torch.cuda.is_available() 是否为 True
```

只有第 5 步通过，默认训练脚本才会继续跑。

## 2. 一键训练

推荐先预检，再训练：

```bash
bash scripts/gpu_preflight.sh
bash scripts/train_policy.sh
```

如果你使用了本机 `.venv-train`：

```bash
PYTHON=.venv-train/bin/python bash scripts/gpu_preflight.sh
PYTHON=.venv-train/bin/python bash scripts/train_policy.sh
```

脚本会自动查找解释器，顺序是：

```text
PYTHON 环境变量
python3.12
python3.11
python3
python
```

如果你的环境没有 `python` 命令，建议显式指定：

```bash
PYTHON=python3 bash scripts/train_policy.sh
```

或者指定 conda/venv 里的完整路径：

```bash
PYTHON=/home/robot/miniconda3/envs/<env-name>/bin/python bash scripts/train_policy.sh
```

如果使用本机创建的训练 venv：

```bash
PYTHON=.venv-train/bin/python bash scripts/train_policy.sh
```

脚本会按顺序执行：

```text
1. 检查 Python 和 PyTorch
2. 生成 synthetic benchmark 到 data/
3. 跑 pytest
4. train-sft
5. train-ppo
6. train-grpo
7. eval --agent grpo
8. report --run runs/latest
```

最终报告：

```text
runs/latest/report.md
```

主要训练产物：

```text
runs/checkpoints/sft.json
runs/checkpoints/sft.pt
runs/checkpoints/ppo.json
runs/checkpoints/ppo.pt
runs/checkpoints/grpo.json
runs/checkpoints/grpo.pt
runs/training/sft/
runs/training/ppo/
runs/training/grpo/
```

`data/` 和 `runs/` 都是生成物，已被 `.gitignore` 忽略。

## 3. 脚本参数

脚本通过环境变量配置。

指定 Python：

```bash
PYTHON=/path/to/python bash scripts/train_policy.sh
```

生成更多任务：

```bash
TASK_COUNT=100 bash scripts/train_policy.sh
```

跳过 benchmark 生成，复用已有 `data/`：

```bash
RUN_BENCHMARK=0 bash scripts/train_policy.sh
```

跳过测试：

```bash
RUN_TESTS=0 bash scripts/train_policy.sh
```

只跑 SFT：

```bash
RUN_PPO=0 RUN_GRPO=0 EVAL_AGENT=sft bash scripts/train_policy.sh
```

只跑 PPO，复用已有 SFT checkpoint：

```bash
RUN_BENCHMARK=0 RUN_TESTS=0 RUN_SFT=0 RUN_GRPO=0 EVAL_AGENT=ppo bash scripts/train_policy.sh
```

只跑 GRPO，复用已有 PPO checkpoint：

```bash
RUN_BENCHMARK=0 RUN_TESTS=0 RUN_SFT=0 RUN_PPO=0 EVAL_AGENT=grpo bash scripts/train_policy.sh
```

## 4. 分步命令

如果不使用脚本，可以手动执行：

```bash
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
python -m pytest -q
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
python -m agentic_code_rl eval --config configs/eval.yaml --agent grpo
python -m agentic_code_rl report --run runs/latest
```

## 5. 配置文件

训练配置在：

```text
configs/sft.yaml
configs/ppo.yaml
configs/grpo.yaml
configs/eval.yaml
```

默认 Transformer 配置：

```text
vocab_size: 8192
task_text_len: 128
observation_text_len: 256
max_steps: 16
d_model: 512
num_layers: 6
num_heads: 8
ffn_dim: 2048
dropout: 0.1
device: cuda
```

默认训练要求 CUDA 可用；如果 `torch.cuda.is_available()` 为 `False`，脚本会退出，不会自动改用 CPU。

脚本也会先运行 `nvidia-smi`。如果 `nvidia-smi` 失败，优先修 NVIDIA driver/runtime；这通常不是 Python 包能解决的问题。

如果 `nvidia-smi` 在你的普通终端正常，但在 Codex/sandbox 里失败，说明 GPU 设备没有暴露给 sandbox。此时应在普通终端运行：

```bash
bash scripts/gpu_preflight.sh
bash scripts/train_policy.sh
```

如果使用本机 venv，则加上 `PYTHON`：

```bash
PYTHON=.venv-train/bin/python bash scripts/gpu_preflight.sh
PYTHON=.venv-train/bin/python bash scripts/train_policy.sh
```

如果 `scripts/gpu_preflight.sh` 在普通终端也失败，先不要跑训练；优先修 NVIDIA driver、device node 或 CUDA runtime。常见判断：

```text
/proc/driver/nvidia 能看到 4090，但没有 /dev/nvidia*
  可能是设备节点没有创建，或当前执行环境没有暴露 GPU。

nvidia-smi 失败，torch.cuda.is_available() 也为 False
  训练不能启动；需要先修驱动/runtime，必要时重启。

nvidia-smi 正常，但 torch.cuda.is_available() 为 False
  通常是当前 Python 环境里的 torch 不是 CUDA build，或 CUDA runtime 不匹配。
```

仅当你明确要做 CPU debug 时才使用：

```bash
ALLOW_CPU_TRAINING=1 bash scripts/train_policy.sh
```

正式训练应确认 `torch.cuda.is_available()` 为 `True`。

## 6. 如何判断训练是否真的走了 torch

检查 checkpoint：

```bash
ls runs/checkpoints/*.pt
```

检查 JSON metadata：

```bash
python - <<'PY'
import json
from pathlib import Path
for path in ["sft", "ppo", "grpo"]:
    data = json.loads(Path(f"runs/checkpoints/{path}.json").read_text())
    print(path, data.get("torch_checkpoint"), data.get("metadata", {}).get("torch_status"))
PY
```

如果看到：

```text
torch unavailable: ...
```

说明没有走真实 torch 训练，只写了 JSON fallback checkpoint。

## 7. 重要边界

当前训练不是完整自主代码修复模型。

当前训练的是：

```text
Tool Policy:
  什么时候读文件、搜索、patch、跑测试、结束
```

当前没有训练：

```text
Patch Generator:
  具体生成什么代码 diff
```

所以 `apply_patch` 目前仍然使用 expert patch provider。这个路径适合做 SFT smoke、tool-policy PPO/GRPO 闭环和 harness 验证；如果要做真正自主修 bug，下一阶段需要加入 patch generator。
