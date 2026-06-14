# Cold Start Handoff: Agentic Code RL

这份文档写给后续接手本项目的 agent。目标是在完全 cold start 的情况下，快速理解项目目标、当前实现状态、可运行命令、训练边界和下一步开发方向。

当前仓库根目录：

```text
/home/robot/Projects/agentic-code-rl
```

## 1. 项目目标

`agentic-code-rl` 是一个面向 Python 代码修复任务的 Agentic RL 项目。它包含：

- SWE-bench-style code repair evaluation harness
- guarded tool layer
- public/hidden test 隔离
- scripted / ReAct / learned-policy agent
- SFT / PPO / GRPO tool-policy 训练入口
- evaluation report 和 trajectory artifacts

这里的 harness 指隔离 workspace、受控工具执行、public/hidden 评测边界和可复现实验 artifacts；普通 MDP loop、PPO/GRPO 算法本身不叫 harness。

当前训练对象是高层 tool policy：

```text
pi(action | task, memory, tool history)
```

动作空间固定为：

```text
list_files / read_file / search_code / apply_patch / run_tests / inspect_failure / finish
```

当前不训练 patch generator。`apply_patch` 的具体 patch 内容仍来自 synthetic expert patch provider。

## 2. 当前实现状态

核心模块：

- `src/agentic_code_rl/benchmark.py`
  - 生成 synthetic Python bug tasks。
  - 生成 visible repo、public tests、hidden tests、task JSON 和独立 expert patch artifact。

- `src/agentic_code_rl/environment.py`
  - `EpisodeWorkspace` 复制 repo 到隔离 workspace。
  - `resolve_path()` 拒绝绝对路径和路径逃逸。
  - public tests 在 episode 内可见；hidden tests 只在最终评测运行。

- `src/agentic_code_rl/tools.py`
  - 实现 `list_files`、`read_file`、`search_code`、`apply_patch`、`run_tests`、`inspect_failure`、`finish`。
  - episode 内默认不允许 `run_tests(scope="hidden")` 或 `run_tests(scope="all")`。

- `src/agentic_code_rl/agents.py`
  - `ScriptedAgent`：使用 expert patch artifact 跑通专家轨迹。
  - `ReactAgent`：OpenAI-compatible chat completion 入口；失败或无 key 时降级 scripted。
  - `LearnedPolicyAgent`：优先加载 `.pt` Transformer checkpoint；JSON checkpoint 作为 metadata/fallback。

- `src/agentic_code_rl/policy.py`
  - `SimpleTextTokenizer`
  - `PolicyFeatureEncoder`
  - `TrajectoryTransformerPolicy`
  - checkpoint save/load 和 action sampling helpers。

- `src/agentic_code_rl/training.py`
  - `train-sft`：从 scripted expert trajectories 做 supervised warm start。
  - `train-ppo`：真实 rollout + clipped PPO objective。
  - `train-grpo`：same-task group rollout + group-relative advantage。
  - 默认配置使用 `device: cuda`。

- `src/agentic_code_rl/evaluation.py`
  - 批量跑 tasks，生成 `eval_summary.json` 和 `runs/latest`。

- `src/agentic_code_rl/reporting.py`
  - 从 eval summary 或 single trajectory 生成 Markdown report。

- `src/agentic_code_rl/cli.py`
  - 提供 benchmark/run/train/eval/report 命令。

## 3. 训练网络

当前 learned policy 是 Trajectory Transformer，不是 LLM，也不生成代码。

默认配置：

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

输入结构：

```text
task_tokens              LongTensor [B, 128]
observation_tokens       LongTensor [B, 256]
global_features          FloatTensor [B, 25]
history_actions          LongTensor [B, 16]
history_statuses         LongTensor [B, 16]
history_positions        LongTensor [B, 16]
history_numeric_features FloatTensor [B, 16, 8]
history_padding_mask     BoolTensor [B, 16]
action_mask              BoolTensor [B, 7]
```

Transformer sequence：

```text
[ task, observation, global, step_1, step_2, ..., step_16, decision ]
```

输出：

```text
logits: [B, 7]
value:  [B]
```

详细说明：

- [POLICY_NETWORK_ARCHITECTURE.md](POLICY_NETWORK_ARCHITECTURE.md)
- [POLICY_NETWORK_IO_WALKTHROUGH.md](POLICY_NETWORK_IO_WALKTHROUGH.md)
- [TRAINING_POLICY_HANDOFF.md](TRAINING_POLICY_HANDOFF.md)

## 4. 快速验证

安装依赖：

```bash
python -m pip install -e '.[train,dev]'
```

生成 benchmark：

```bash
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
```

跑测试：

```bash
python -m pytest -q
```

跑 scripted smoke：

```bash
python -m agentic_code_rl run --task data/tasks/task_001.json --agent scripted
```

批量评测：

```bash
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
python -m agentic_code_rl report --run runs/latest
```

## 5. GPU 训练

训练前先预检 GPU：

```bash
bash scripts/gpu_preflight.sh
```

如果使用本地 venv：

```bash
PYTHON=.venv-train/bin/python bash scripts/gpu_preflight.sh
```

一键训练：

```bash
bash scripts/train_policy.sh
```

或者显式指定 Python：

```bash
PYTHON=.venv-train/bin/python bash scripts/train_policy.sh
```

脚本顺序：

```text
1. 检查 Python/PyTorch/CUDA
2. benchmark create
3. pytest
4. train-sft
5. train-ppo
6. train-grpo
7. eval --agent grpo
8. report --run runs/latest
```

训练产物：

```text
runs/checkpoints/sft.json
runs/checkpoints/sft.pt
runs/checkpoints/ppo.json
runs/checkpoints/ppo.pt
runs/checkpoints/grpo.json
runs/checkpoints/grpo.pt
runs/latest/report.md
```

`data/` 和 `runs/` 是生成物，已被 `.gitignore` 忽略，不要提交。

## 6. 当前验证结果

在本机普通终端已验证：

```text
Python: 3.12.13
GPU: NVIDIA GeForce RTX 4090
torch: 2.12.0+cu130
torch.cuda.is_available(): True
pytest: 9 passed
```

一键训练已跑通，生成：

```text
runs/checkpoints/sft.pt
runs/checkpoints/ppo.pt
runs/checkpoints/grpo.pt
runs/latest/report.md
```

最近一次 synthetic benchmark 评测：

```text
agent: grpo
task_count: 30
pass@1: 1.000
hidden_pass_rate: 1.000
public_pass_rate: 1.000
```

注意：这个 100% 结果来自 synthetic tasks + expert patch provider。它证明 tool-policy 训练和评测闭环能跑通，不代表模型已经学会自主生成代码 patch。

## 7. 重要边界

必须保持：

- hidden tests 不能进入 episode 内 observation。
- task JSON 不能包含 expert patch 内容。
- `data/`、`runs/`、`.env`、本地 venv 不能提交。
- 默认训练使用 GPU；CPU 只允许 debug。
- 报告必须标明 `training_target: tool_policy` 和 `patch_generation: expert_patch_provider`。

当前尚未实现：

- learned patch generator
- 强 ReAct/API patch generation 闭环
- failure classifier
- experiment registry
- multi-run comparison report
- SWE-bench Lite adapter

## 8. 下一步建议

优先级建议：

1. 实现 patch generator 训练或 API patch generation，让 `apply_patch` 不再依赖 expert patch provider。
2. 给 `ReactAgent` 增加严格 JSON patch prompt，并记录 `fallback_used`。
3. 增加 failure classification：`path_error`、`patch_no_match`、`syntax_error`、`public_test_fail`、`hidden_test_fail`、`timeout`、`premature_finish`、`tool_loop`。
4. 增加 experiment registry 和多 run 对比报告。
5. 扩展 synthetic benchmark 到更多 task 类型和难度层级。
6. 接入 SWE-bench Lite 子集，但不要破坏现有 synthetic benchmark。

## 9. Cold-Start Prompt

可以直接把下面这段作为后续 agent 的启动 prompt：

```text
你正在接手 /home/robot/Projects/agentic-code-rl。请先阅读 README.md、docs/PROJECT_DESIGN.md、docs/TRAINING_QUICKSTART.md、docs/POLICY_NETWORK_ARCHITECTURE.md、docs/POLICY_NETWORK_IO_WALKTHROUGH.md 和 docs/COLD_START_HANDOFF.md。当前已经实现 Transformer tool policy 的 SFT/PPO/GRPO 训练，并在 RTX 4090 上跑通。请先运行 pytest 和必要 CLI smoke，确认当前状态。必须保持 hidden tests 只用于 final evaluation，不要提交 data/、runs/、.env 或本地 venv。下一阶段优先减少对 expert patch provider 的依赖，推进真实 patch generation。
```
