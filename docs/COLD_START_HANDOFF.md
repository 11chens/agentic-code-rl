# Cold Start Handoff: Agentic Code RL

这份文档写给后续接手本项目的 agent。目标是在完全 cold start 的情况下，快速理解用户真实需求、原始项目计划、当前实现边界、可运行命令，以及下一步如何把项目升级成更可靠的代码修复评测框架。

## 1. 任务背景与用户目标

用户已有 VLA 相关经历，但当前没有机器人、仿真环境、VLM model。为了避免强行做一个空泛的具身 demo，本项目选择转向 **代码修复任务上的 Agentic RL**：让 agent 在多轮工具调用中读代码、搜索、打 patch、跑测试、反思失败，并通过 reward 优化决策。

求职方向关注的能力包括：

- Agent 规划、记忆、工具调用、反思与重试。
- PPO、GRPO、Agentic RL、多轮强化学习。
- Python/PyTorch 工程能力、Git、可复现实验和评测报告。
- 能讲清楚从已有 VLA 背景到上层长时序智能体决策的迁移关系。

后续开发的总目标不是继续堆 demo，而是把本项目做成一个 **SWE-bench-style code repair evaluation harness + tool-using agent runtime + training loop scaffold**。这里的 harness 只指隔离 workspace、受控工具执行、public/hidden 评测边界和可复现实验 artifacts，不把普通 MDP loop 或 PPO/GRPO 算法本身叫 harness。

## 2. 原始项目计划

项目名称：`agentic-code-rl`。

核心任务：面向 Python 代码修复任务，构建一个 tool-using code repair agent 的评测框架，并预留 SFT/PPO/GRPO 训练入口。

五层设计：

1. **Task Environment**
   - 每个 episode 复制一个小型 buggy Python repo 到隔离 workspace。
   - 输入包括 bug 描述、文件树、公开测试。
   - hidden tests 只用于最终 reward 和 evaluation，不能在 episode 内暴露给 agent。

2. **Tool Layer**
   - 固定工具：`list_files`、`read_file`、`search_code`、`apply_patch`、`run_tests`、`inspect_failure`、`finish`。
   - 所有文件操作必须限制在 episode workspace 内。
   - `run_tests` 有 timeout，记录 stdout、stderr、失败摘要。

3. **Agent Loop**
   - API LLM 负责具体分析、patch 生成和反思。
   - 小型 trainable policy 负责高层动作选择。
   - Memory 保存最近观察、工具调用、失败原因、patch diff、测试变化。

4. **RL Training**
   - SFT：用 scripted expert 轨迹训练初始 policy。
   - PPO：用离散 tool action policy 优化多轮决策。
   - GRPO-style：同一任务采样多条轨迹，用组内相对 reward 优化策略或 ranker。

5. **Evaluation**
   - 指标包括 `pass@1`、hidden pass rate、public pass rate、平均工具调用数、无效 patch 率、语法/导入错误率、API 成本、episode 时长。
   - 对比 agent：ReAct/API-only、SFT、PPO、GRPO-style。
   - 消融：无 memory、无 reflection、无 RL、无 hidden reward shaping。

默认 CLI 设计：

```powershell
python -m agentic_code_rl benchmark create --out data/tasks
python -m agentic_code_rl run --task data/tasks/task_001.json --agent react
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
python -m agentic_code_rl eval --config configs/eval.yaml --agent grpo
python -m agentic_code_rl report --run runs/<run_id>
```

Reward baseline：

```text
+1.00 hidden tests all pass
+0.40 public tests all pass
+0.20 public test failure count improves
-0.01 each tool call
-0.05 invalid tool call
-0.10 patch creates syntax/import failure
-0.20 finish while hidden tests fail
```

重要边界：

- v1 不做具身、不接机器人、不依赖 VLM。
- v1 不先接 SWE-bench，先用可控 synthetic benchmark 做出完整闭环。
- 当前 scripted/expert path 是 baseline 和 smoke path，不是最终算法成果。

## 3. 当前实现状态

当前仓库位置：

```text
D:\Vibethon\agentic-code-rl
```

Git 初始提交：

```text
0e877fb Initial agentic code RL project
```

已实现模块：

- `src/agentic_code_rl/benchmark.py`
  - 生成 synthetic Python bug tasks。
  - 默认 case library 覆盖 prime、factorial、median、binary search、unique、chunk、safe divide、anagram、CSV parse、rotation 等 bug 类型。
  - 每个 task 生成 visible repo、public tests、私有 hidden tests、metadata 和单独的 expert patch artifact。

- `src/agentic_code_rl/environment.py`
  - `EpisodeWorkspace` 负责复制 repo 到隔离 workspace。
  - `resolve_path()` 拒绝绝对路径和路径逃逸。
  - `run_tests()` 通过 pytest 子进程运行 public tests；final evaluation 通过私有 hidden source 运行 hidden tests。

- `src/agentic_code_rl/tools.py`
  - 实现 guarded tool layer。
  - episode 内默认不允许 `run_tests(scope="hidden")` 或 `run_tests(scope="all")`。
  - `apply_patch` 支持 find/replace 和 full content 写入。

- `src/agentic_code_rl/agents.py`
  - `ScriptedAgent`：使用 synthetic case library 中的 expert patch 跑通稳定专家轨迹；task JSON 不暴露答案。
  - `ReactAgent`：有 OpenAI-compatible chat completion 入口；没有 key 或失败时降级 scripted。
  - `LearnedPolicyAgent`：读取 JSON checkpoint 的 lightweight action score policy。

- `src/agentic_code_rl/runner.py`
  - 实现 episode loop、memory、reward delta、final public/hidden evaluation、trajectory artifacts。

- `src/agentic_code_rl/training.py`
  - `train-sft`、`train-ppo`、`train-grpo` 已有 CLI 入口。
  - 当前 PPO/GRPO 是 lightweight checkpoint/ranker scaffold，不是真实 rollout RL。
  - 无 torch 时写 JSON checkpoint；有 torch 时 SFT 会额外训练一个小型 `.pt`。
  - 每次训练写 `replay_buffer.json`，支持 `resume_from` 读取旧 checkpoint 合并 scores。

- `src/agentic_code_rl/evaluation.py`
  - 批量跑 tasks，生成 `eval_summary.json` 和 `runs/latest`。

- `src/agentic_code_rl/reporting.py`
  - 从 single trajectory 或 eval summary 生成 Markdown report。

- `src/agentic_code_rl/cli.py`
  - 提供 benchmark/run/train/eval/report 命令。

配置文件：

```text
configs/eval.yaml
configs/sft.yaml
configs/ppo.yaml
configs/grpo.yaml
```

验证状态：

```powershell
..\conda-envs\web-rpa\python.exe -m pytest -q
# 6 passed
```

已跑通过的 smoke 命令：

```powershell
python -m agentic_code_rl benchmark create --out data/tasks --count 3 --overwrite
python -m agentic_code_rl run --task data/tasks/task_001.json --agent scripted --run-id smoke-scripted
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
python -m agentic_code_rl report --run runs/latest
```

注意：当前笔记本默认 `python` 指向 Python 3.4，不能用于本项目。此前验证使用的是：

```text
D:\Vibethon\conda-envs\web-rpa\python.exe
Python 3.12.13
```

## 4. Cold Start 操作手册

### 4.1 本机快速验证

使用 Python 3.11+。如果在当前笔记本上，建议显式调用：

```powershell
cd D:\Vibethon\agentic-code-rl
..\conda-envs\web-rpa\python.exe -m pip install -e .[dev]
..\conda-envs\web-rpa\python.exe -m pytest -q
```

如果 shell 中 `python` 已经是 3.11+：

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

### 4.2 生成 benchmark

```powershell
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
```

生成内容：

```text
data/tasks/task_001.json
data/tasks/manifest.json
data/repos/task_001/src/buggy_lib.py
data/repos/task_001/tests/test_public.py
data/hidden_tests/task_001/tests/test_hidden.py
data/expert_patches/task_001/patch.json
```

`data/` 被 `.gitignore` 忽略，不要提交。

### 4.3 跑单个 episode

```powershell
python -m agentic_code_rl run --task data/tasks/task_001.json --agent scripted
python -m agentic_code_rl run --task data/tasks/task_001.json --agent react
```

输出 artifacts：

```text
runs/<run_id>/workspace/
runs/<run_id>/trajectory.json
runs/<run_id>/summary.json
```

`runs/` 被 `.gitignore` 忽略，不要提交。

### 4.4 批量评测和报告

```powershell
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
python -m agentic_code_rl report --run runs/latest
```

关键输出：

```text
runs/latest/eval_summary.json
runs/latest/report.md
```

### 4.5 训练入口

本机无 torch 也能写 JSON fallback checkpoint：

```powershell
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
```

开发机建议安装完整 extras：

```powershell
python -m pip install -e .[train,llm,dev]
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
python -m agentic_code_rl eval --config configs/eval.yaml --agent grpo
python -m agentic_code_rl report --run runs/latest
```

### 4.6 Push 到 GitHub 后在开发机继续

```powershell
cd D:\Vibethon\agentic-code-rl
git remote add origin https://github.com/<you>/agentic-code-rl.git
git push -u origin main
```

开发机：

```powershell
git clone https://github.com/<you>/agentic-code-rl.git
cd agentic-code-rl
python -m pip install -e .[train,llm,dev]
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
python -m pytest -q
```

如果要启用 OpenAI-compatible ReAct：

```powershell
Copy-Item .env.example .env
# 填写 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
```

当前代码读取环境变量，不自动加载 `.env`。如果需要 `.env` 自动加载，后续可加 `python-dotenv` 或在 shell 中显式设置环境变量。

## 5. 升级路线

### P0：开发机真实训练路径

目标：确认开发机能跑 PyTorch，并让 SFT 至少产生 `.pt` checkpoint。

推荐步骤：

1. 安装 `python -m pip install -e .[train,llm,dev]`。
2. 跑 `train-sft`，确认同时生成 `runs/checkpoints/sft.json` 和 `runs/checkpoints/sft.pt`。
3. 增加一个测试或 smoke script，检查 torch checkpoint 可加载。

验收标准：

- `train-sft` 在开发机稳定完成。
- `sft.pt` 可被加载并用于 action logits 推理。
- fallback JSON checkpoint 仍保留，方便无 GPU/无 torch 环境查看。

### P1：真实 PPO rollout training

目标：把当前 PPO scaffold 升级成真实 trajectory rollout + clipped objective。

推荐设计：

- 保留 `ACTIONS` 离散动作空间。
- 将 `Memory` 编码成固定维度 features：step index、action counts、last test status、patch count、invalid count、public failure count。
- PPO rollout 调用真实 `run_episode` 或可控 lightweight environment。
- 保存：policy checkpoint、optimizer state、replay/rollout buffer、training metrics。

验收标准：

- `train-ppo` 不只是改 action scores，而是采样 episode、计算 advantage、更新 policy。
- 支持 `resume_from` 恢复 checkpoint 和 optimizer state。
- `eval --agent ppo` 能读取新 checkpoint 并通过统一 evaluation harness 评测。

### P1：真实 GRPO-style group sampling

目标：同一 task 采样 K 条轨迹，用组内相对 reward 优化 planner/policy。

推荐设计：

- `group_size` 默认 4。
- 同一 task 用不同 temperature/epsilon/prompt variant 采样多条 trajectory。
- 计算 group mean reward，使用 `reward_i - mean(group)` 作为 relative advantage。
- 记录每组的 winner、失败类型、tool cost。

验收标准：

- `train-grpo` 产出 group-level artifact。
- report 能展示 PPO vs GRPO-style 的 hidden pass rate、tool cost、失败类型差异。

### P1：增强 ReAct/API agent

目标：让 LLM 真实负责代码分析和 patch 生成，而不是主要依赖 expert patch。

推荐步骤：

1. 给 `ReactAgent` 增加严格 JSON repair prompt。
2. 允许 LLM 在 `apply_patch` 中生成 find/replace。
3. 如果 find 不匹配，引导 agent 先 `read_file` 或 `search_code`，不要直接用 expert patch。
4. 保留 scripted fallback，但在 metrics 中标记 fallback 是否发生。

验收标准：

- `react` 在没有 expert patch 的 task 上也能尝试修复。
- trajectory 能看到 LLM rationale、patch diff、失败反思。
- API cost 被记录到 metrics。

### P2：Benchmark 扩展到 100+ tasks

目标：从 30 个可控 tasks 扩到 100+，并做难度分层。

推荐维度：

- bug 类型：边界条件、索引、异常、浮点、字符串解析、数据结构、递归、状态污染、排序稳定性、日期时间。
- 难度：single-line、multi-line、multi-file、requires test inference、requires API behavior inference。
- hidden tests：覆盖 public tests 看不到的边界，不要只复制 public 的同类断言。

验收标准：

- `benchmark create --count 100` 可稳定生成。
- scripted expert pass rate 为 100%。
- react/sft/ppo/grpo 的指标有可区分度。

### P2：Experiment registry 和对比报告

目标：把实验流程从“能跑命令”升级成“能管理实验”。

推荐设计：

- 新增 `configs/experiments/*.yaml`。
- 每个 experiment 声明：task split、agent、checkpoint、seed、limit、ablation flags。
- `eval` 输出 run registry：run id、git commit、config hash、agent、metrics。
- `report` 支持多 run 对比表格。

验收标准：

- 一个命令能跑 scripted/react/sft/ppo/grpo 对比。
- report 自动输出 baseline table、learning curve 路径、失败案例。

### P2：Trajectory store 和失败分类

目标：让训练和评测能复用 episode artifacts。

推荐分类：

- `path_error`
- `patch_no_match`
- `syntax_error`
- `public_test_fail`
- `hidden_test_fail`
- `timeout`
- `premature_finish`
- `tool_loop`

验收标准：

- 每条 trajectory 有 stable schema version。
- replay buffer 能按 task/tag/failure type 过滤。
- report 展示 top failure modes。

### P2：Safety 和 sandbox 加强

目标：让 code repair harness 更像真实评测系统。

推荐加强：

- 限制 subprocess timeout、输出长度、workspace 大小。
- patch 前后做 diff 审查。
- 拒绝写入 secrets-like 文件名。
- 将 `apply_patch(content=...)` 的 full overwrite 能力改为可配置。

验收标准：

- 恶意路径、超时测试、巨大输出、无效 patch 都有可解释 failure artifact。
- 安全策略不破坏 scripted baseline。

### P3：接 SWE-bench Lite 子集

目标：在 synthetic benchmark 稳定后，接入更真实的外部 benchmark。

推荐约束：

- 先只接 5-10 个任务。
- 不破坏现有 synthetic benchmark。
- 外部 task adapter 输出同一个 `TaskSpec`/trajectory schema。

验收标准：

- `eval` 可混合或单独跑 synthetic/SWE-bench-lite。
- report 中区分 benchmark source。

## 6. 交接任务清单

### P0

- 在开发机安装 `.[train,llm,dev]` 并跑通 torch SFT。
- 让 `LearnedPolicyAgent` 读取 `.pt`，而不只是 JSON action scores。
- 加 smoke script：一键 benchmark -> train-sft -> eval -> report。

### P1

- 将 PPO 从 lightweight checkpoint 升级成真实 rollout training。
- 将 GRPO-style 从静态 group advantage 升级成真实 group sampling。
- 增强 ReAct/API agent 的 patch 生成闭环。
- 给 fallback path 增加 metrics 标记，避免报告中混淆 LLM 和 scripted。

### P2

- 扩展 benchmark 到 100+ tasks。
- 加 experiment registry。
- 加多 run 对比报告。
- 加 failure classification。

### P3

- 接 SWE-bench Lite 子集。
- 增加更强 sandbox/resource limit。
- 生成面向简历和面试的实验分析报告。

## 7. 后续 agent 必须遵守的约束

- 不要破坏 hidden-test 边界：episode 内工具只能跑 public tests，也不能读/搜 hidden tests；hidden tests 只在 final evaluation 中运行。
- 不要提交 `data/`、`runs/`、`.env` 或任何真实 API key。
- 不要把 current scripted/expert path 当作最终算法成果；它只是 baseline、expert trace source 和 smoke path。
- 每次新增 agent 或算法，都必须能通过同一套 `eval` 和 `report` 输出可比较指标。
- 优先提升 evaluation harness 的可复现性、可比较性和安全边界，再扩展更复杂模型。
- 任何训练升级都要保留无 torch 环境下可 inspect 的 JSON artifact。

## 8. 推荐给下一位 agent 的第一条指令

可以直接把下面这段作为 cold-start prompt：

```text
你正在接手 D:\Vibethon\agentic-code-rl。请先阅读 README.md、docs/PROJECT_DESIGN.md、docs/COLD_START_HANDOFF.md。目标是把当前 Agentic Code RL 从 lightweight scaffold 升级为可靠的 SWE-bench-style code repair evaluation harness + training loop。先不要大改架构；请运行 pytest 和 CLI smoke，确认当前状态，然后优先实现 P0：开发机真实 torch SFT checkpoint 加载与 LearnedPolicyAgent .pt 推理。必须保持 hidden tests 只用于 final evaluation，不要提交 data/、runs/、.env。
```
