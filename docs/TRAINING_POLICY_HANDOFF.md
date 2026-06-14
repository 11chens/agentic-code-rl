# Training Policy Handoff

这份文档说明 `agentic-code-rl` 当前训练系统的设计和实现路线。当前已经把早期 lightweight checkpoint scaffold 升级成真实可训练的 **高层 tool-use policy**。本文先不考虑 patch 生成训练。

关键结论：

```text
训练对象：Tool Policy / Planner
不训练对象：Patch Generator / Code Editor
推荐网络：Trajectory Transformer + policy head + value head
训练顺序：SFT warm start -> PPO rollout fine-tune -> GRPO group rollout fine-tune
```

网络输入输出的逐字段走读见 [POLICY_NETWORK_IO_WALKTHROUGH.md](POLICY_NETWORK_IO_WALKTHROUGH.md)。
从零启动训练的操作文档见 [TRAINING_QUICKSTART.md](TRAINING_QUICKSTART.md)。

## 1. 训练边界

当前阶段只训练：

```text
pi(action | task, memory, tool history)
```

也就是训练 agent 在每一步选择哪个工具动作：

```text
list_files
read_file
search_code
apply_patch
run_tests
inspect_failure
finish
```

当前阶段不训练：

```text
pi(patch | source code, bug prompt, test output, memory)
```

所以 policy 的输出不是代码 diff，也不是 `find/replace` 字符串，而是 7 个 tool action 的概率分布。

当 policy 输出：

```text
apply_patch
```

第一版仍然由 expert patch provider 生成具体 patch payload：

```json
{
  "path": "src/buggy_lib.py",
  "find": "...old source...",
  "replace": "...fixed source..."
}
```

这必须在 artifact 中明确标记：

```text
training_target: tool_policy
patch_generation: expert_patch_provider
scripted_patch: true
```

这样做的原因是：高层 tool policy 是一个可控的离散 RL 问题，可以先把 SFT/PPO/GRPO 闭环做扎实；patch 生成是代码编辑生成问题，后续需要单独做 ReAct/API patch generator、patch SFT 或 DPO。

## 2. 是不是 Transformer

是。推荐 P0/P1 默认实现为：

```text
Trajectory Transformer Policy
```

但这里的 Transformer 不是大语言模型，也不是用来生成代码。它是一个中小型 transformer encoder，用来建模一次 episode 里的工具调用历史和当前状态。

推荐默认规模，适合 24G 4090：

```text
d_model: 512
num_layers: 6
num_heads: 8
ffn_dim: 2048
dropout: 0.1
max_steps: 16
task_text_len: 128
obs_text_len: 256
vocab_size: 8192 or 16384
```

这个规模对 4090 很轻。真正耗时的部分通常不是模型训练，而是 rollout 中反复启动 pytest 子进程。

如果后续想进一步增强，可以把 task/source/test output 文本编码器换成 CodeBERT、MiniLM 或其他 pretrained encoder，但第一版不要引入这个依赖。先用本项目内可控的 tokenizer + embedding + transformer，把训练闭环跑通。

## 3. 整体网络结构

推荐新增模块：

```text
src/agentic_code_rl/policy.py
```

核心类：

```text
SimpleTextTokenizer
PolicyFeatureEncoder
TrajectoryTransformerPolicy
PolicyCheckpoint
action_mask_for_memory()
tool_input_for_action()
```

网络结构：

```text
Task text tokens
  -> text embedding
  -> mean/attention pooling
  -> task token

Last observation text tokens
  -> text embedding
  -> mean/attention pooling
  -> observation token

Global numeric features
  -> linear projection
  -> global token

Step history features
  -> action embedding + status embedding + position embedding + numeric projection
  -> step tokens

[task token, observation token, global token, step_1 ... step_T, decision token]
  -> TransformerEncoder
  -> decision token hidden state
  -> policy head -> action logits [7]
  -> value head -> scalar value
```

伪代码：

```python
class TrajectoryTransformerPolicy(nn.Module):
    def __init__(self, config):
        self.text_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.action_embedding = nn.Embedding(num_actions + 1, config.d_model)
        self.status_embedding = nn.Embedding(num_statuses, config.d_model)
        self.position_embedding = nn.Embedding(config.max_steps + 8, config.d_model)
        self.step_numeric_proj = nn.Linear(step_feature_dim, config.d_model)
        self.global_proj = nn.Linear(global_feature_dim, config.d_model)
        self.decision_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.transformer = nn.TransformerEncoder(...)
        self.policy_head = nn.Linear(config.d_model, num_actions)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(self, batch):
        task_token = pool(self.text_embedding(batch.task_tokens))
        obs_token = pool(self.text_embedding(batch.observation_tokens))
        global_token = self.global_proj(batch.global_features)

        step_tokens = (
            self.action_embedding(batch.history_actions)
            + self.status_embedding(batch.history_statuses)
            + self.position_embedding(batch.history_positions)
            + self.step_numeric_proj(batch.history_numeric_features)
        )

        sequence = concat([
            task_token,
            obs_token,
            global_token,
            step_tokens,
            decision_token,
        ])

        hidden = self.transformer(sequence, src_key_padding_mask=batch.padding_mask)
        decision_hidden = hidden[:, -1]
        logits = self.policy_head(decision_hidden)
        value = self.value_head(decision_hidden).squeeze(-1)
        masked_logits = logits.masked_fill(~batch.action_mask, -1e9)
        return masked_logits, value
```

## 4. Policy 输入具体长什么样

一次 policy forward 对应 episode 中的一个 decision point。

推荐 batch schema：

```python
PolicyBatch(
    task_tokens: LongTensor,              # [B, L_task]
    observation_tokens: LongTensor,       # [B, L_obs]
    global_features: FloatTensor,         # [B, F_global]
    history_actions: LongTensor,          # [B, T]
    history_statuses: LongTensor,         # [B, T]
    history_positions: LongTensor,        # [B, T]
    history_numeric_features: FloatTensor,# [B, T, F_step]
    history_padding_mask: BoolTensor,     # [B, T]
    action_mask: BoolTensor,              # [B, 7]
)
```

### 4.1 `task_tokens`

来自 task prompt 和 metadata，例如：

```text
Fix prime detection for edge cases.
function: is_prime
target_file: src/buggy_lib.py
tags: logic edge-case
```

shape：

```text
[batch_size, 128]
```

第一版 tokenizer 可以用简单 regex/hash tokenizer：

```text
lowercase -> split on non-alnum/underscore -> hash into vocab_size
```

这不是为了深度理解代码，只是让 policy 能感知任务类型和函数名提示。

### 4.2 `observation_tokens`

来自当前 memory observation 的短文本，重点包含最近工具输出摘要。例如：

```text
Task: Fix prime detection for edge cases.
Recent tool history:
- search_code: README.md:5 Target function: is_prime
- read_file: def is_prime(n): ...
```

shape：

```text
[batch_size, 256]
```

注意：这里不能包含 hidden test 内容。可以包含 public test 输出、read_file 内容摘要和工具错误信息。

### 4.3 `global_features`

这是当前状态的结构化全局特征。推荐第一版包含：

```text
step_index / max_steps
remaining_steps / max_steps

action count normalized:
  list_files_count / max_steps
  read_file_count / max_steps
  search_code_count / max_steps
  apply_patch_count / max_steps
  run_tests_count / max_steps
  inspect_failure_count / max_steps
  finish_count / max_steps

boolean flags:
  has_listed_files
  has_searched_code
  has_read_target
  has_applied_patch
  has_run_public_tests
  has_inspected_failure
  last_tool_ok
  last_tool_invalid
  last_public_test_passed
  public_failure_improved
  has_function_name
  has_target_file

normalized counters:
  last_public_failure_count / 10
  patches_applied / max_steps
  invalid_tool_calls / max_steps
  syntax_or_import_errors / max_steps
```

建议 `F_global` 第一版大约 28-40 维。具体维度由 `PolicyFeatureEncoder.feature_schema` 固定保存到 checkpoint。

### 4.4 `history_actions`

过去每一步执行过的 action id：

```text
PAD = 0
list_files = 1
read_file = 2
search_code = 3
apply_patch = 4
run_tests = 5
inspect_failure = 6
finish = 7
```

shape：

```text
[batch_size, max_steps]
```

例如 task_001 在准备 `apply_patch` 前：

```text
[list_files, search_code, read_file, PAD, PAD, ...]
```

### 4.5 `history_statuses`

每一步工具调用状态：

```text
PAD = 0
OK = 1
FAILED = 2
INVALID = 3
TEST_PASSED = 4
TEST_FAILED = 5
```

shape：

```text
[batch_size, max_steps]
```

### 4.6 `history_numeric_features`

每个历史 step 的数值特征。推荐包含：

```text
reward_delta
tool_ok
tool_invalid
is_public_test
public_test_passed
failure_count / 10
passed_count / 10
patches_applied_after_step / max_steps
```

shape：

```text
[batch_size, max_steps, F_step]
```

第一版 `F_step` 大约 8-12 维。

### 4.7 `action_mask`

合法动作 mask：

```text
[batch_size, 7]
```

顺序对应：

```text
[list_files, read_file, search_code, apply_patch, run_tests, inspect_failure, finish]
```

mask 只防止明显非法动作，不要把专家策略硬编码进去。

推荐规则：

```text
run_tests 永远只允许 public scope
inspect_failure 只有 last_test_result 存在时可选
apply_patch 第一版最多允许一次
finish 在第 0 步不允许
hidden/all test 不是 action 输出的一部分，永远不可选
```

可选：第一步强制或强烈偏向 `list_files`。如果强制，会减少探索但也减少无意义轨迹；如果不强制，SFT 通常会学到第一步选 `list_files`。

## 5. Policy 输出具体长什么样

模型输出：

```python
PolicyOutput(
    logits: FloatTensor,        # [B, 7], mask 后的 action logits
    value: FloatTensor,         # [B]
)
```

训练或采样时：

```python
dist = Categorical(logits=logits)
action_id = dist.sample()
logprob = dist.log_prob(action_id)
entropy = dist.entropy()
```

推理时可以用：

```python
action_id = logits.argmax(dim=-1)
```

`LearnedPolicyAgent.decide()` 最终返回：

```python
AgentDecision(
    action="run_tests",
    tool_input={"scope": "public"},
    rationale="Policy selected action.",
    policy_logprob=-0.73,
)
```

建议扩展 `TrajectoryStep.metadata`，记录：

```text
policy_value
policy_entropy
action_mask
checkpoint_path
temperature
```

如果暂时不改 schema，也至少要在 rollout buffer 里记录这些训练字段。

## 6. Tool Input 如何生成

policy 只输出 action，`tool_input` 用规则生成。

```text
list_files      -> {}
read_file       -> {"path": task.metadata["target_file"] or "src/buggy_lib.py"}
search_code     -> {"query": task.metadata["function_name"] or "def "}
apply_patch     -> expert patch provider
run_tests       -> {"scope": "public"}
inspect_failure -> {}
finish          -> {}
```

`apply_patch` 的 payload provider 必须和 policy 分开。未来替换 patch generator 时，只替换这里，不改 policy 输出空间。

## 7. 一个具体样例

以 `task_001` 为例，在执行完：

```text
list_files -> search_code -> read_file
```

下一步需要决策时，输入大致是：

```text
task_tokens:
  "fix prime detection for edge cases function is_prime target_file src buggy_lib py tags logic edge case"

observation_tokens:
  "recent tool history list_files readme src buggy_lib tests test_public search_code is_prime read_file def is_prime n ..."

global_features:
  step_index = 3 / 12
  remaining_steps = 9 / 12
  list_files_count = 1 / 12
  search_code_count = 1 / 12
  read_file_count = 1 / 12
  apply_patch_count = 0
  run_tests_count = 0
  has_read_target = 1
  has_applied_patch = 0
  has_run_public_tests = 0
  last_tool_ok = 1
  last_tool_invalid = 0

history_actions:
  [list_files, search_code, read_file, PAD, ...]

action_mask:
  list_files: maybe true
  read_file: true
  search_code: true
  apply_patch: true
  run_tests: true
  inspect_failure: false
  finish: true or false depending on policy rule
```

输出 logits 可能是：

```text
list_files:      -1.8
read_file:       -0.4
search_code:     -0.7
apply_patch:      2.6
run_tests:        0.1
inspect_failure: -inf
finish:          -1.2
```

softmax 后最高的是 `apply_patch`，agent 调用 expert patch provider，工具层执行 patch。

## 8. 阶段 0：环境准备

默认训练机器：

```text
GPU: RTX 4090 24GB
Python: >= 3.11
Torch: >= 2.2
```

验收标准：

```text
python -m pytest -q
python -m agentic_code_rl benchmark create --out data/tasks --count 30 --overwrite
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
```

scripted baseline 应该在 synthetic tasks 上 hidden pass rate 为 1.0。后续所有训练算法都和 scripted/react/sft/ppo/grpo 共用同一个 `eval` 和 `report`。

## 9. 阶段 1：SFT Warm Start

SFT 必须做。它的作用是让 policy 先学会基本工具顺序，否则 PPO 从随机策略开始会浪费大量 rollout：

```text
反复 list_files
没有测试结果就 inspect_failure
过早 finish
没有读文件就 apply_patch
```

专家轨迹来自 `ScriptedAgent`：

```text
list_files
search_code
read_file
apply_patch
run_tests
finish
```

训练样本构造：

```text
for each task:
  memory = empty
  for expert_action in expert_sequence:
    x = encode(memory, task)
    y = expert_action
    append (x, y)
    memory = memory + simulated/executed expert step
```

更稳的做法是直接用 `run_episode(..., ScriptedAgent())` 生成真实 trajectory，然后从 trajectory prefixes 构造 SFT 样本。这样 observation、reward_delta、metadata 都和真实 runner 一致。

SFT loss：

```text
loss_policy = CE(masked_logits, expert_action)
```

可选 value warmup：

```text
return_t = discounted future reward from scripted trajectory
loss_value = MSE(value_t, return_t)
loss = loss_policy + value_coef * loss_value
```

输出：

```text
runs/checkpoints/sft.json
runs/checkpoints/sft.pt
runs/training/sft/replay_buffer.json
runs/training/sft/sft_metrics.json
```

`sft.pt` 必须保存：

```text
model_state_dict
model_config
action_list
tokenizer_config
feature_schema
training_target
patch_generation
scripted_patch
```

验收标准：

```text
train-sft 生成 .json 和 .pt
LearnedPolicyAgent 能加载 sft.pt
eval --agent sft --checkpoint runs/checkpoints/sft.pt 能跑
trajectory 中能看到 policy_logprob
SFT agent 的动作序列接近 scripted
hidden tests 仍只在 final evaluation 运行
```

## 10. 阶段 2：PPO Rollout Training

PPO 在 SFT checkpoint 基础上继续训练。PPO 必须调用真实 `run_episode()`，不能再只是静态修改 action scores。

PPO rollout 数据每一步至少保存：

```text
task_id
state_features or encoded batch reference
action
action_mask
old_logprob
value
reward_delta
done
metadata
```

episode 级保存：

```text
final_reward
public_passed
hidden_passed
tool_calls
invalid_tool_calls
syntax_or_import_errors
```

reward 分配：

```text
step_rewards = trajectory.steps[*].reward_delta
terminal_bonus = trajectory.final_reward - sum(step_rewards)
last_step_reward += terminal_bonus
```

这样不会重复计算 reward，同时把 hidden-test final reward 分配到轨迹末尾。

PPO advantage 第一版：

```text
return_t = discounted sum from t
advantage_t = return_t - value_t
advantage = normalize(advantage)
```

后续可以升级 GAE：

```text
delta_t = reward_t + gamma * value_{t+1} - value_t
gae_t = delta_t + gamma * lambda * gae_{t+1}
```

PPO loss：

```text
ratio = exp(new_logprob - old_logprob)
unclipped = ratio * advantage
clipped = clip(ratio, 1 - clip_range, 1 + clip_range) * advantage
policy_loss = -mean(min(unclipped, clipped))

value_loss = mean((new_value - return)^2)
entropy_loss = -mean(entropy)

loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

推荐配置：

```text
epochs: 10
rollout_tasks_per_epoch: 30
ppo_update_epochs: 4
minibatch_size: 128
gamma: 0.99
gae_lambda: 0.95 optional
clip_range: 0.2
value_coef: 0.5
entropy_coef: 0.01
learning_rate: 3e-4
temperature: 1.0
```

输出：

```text
runs/checkpoints/ppo.json
runs/checkpoints/ppo.pt
runs/training/ppo/rollouts.json
runs/training/ppo/ppo_metrics.json
```

metrics：

```text
mean_episode_reward
hidden_pass_rate
public_pass_rate
avg_tool_calls
invalid_tool_rate
syntax_error_rate
policy_loss
value_loss
entropy
clip_fraction
approx_kl
```

验收标准：

```text
train-ppo 真实产生 rollouts
rollouts.json 中能看到真实 actions 和 rewards
ppo.pt 可被 LearnedPolicyAgent 加载
eval --agent ppo 可生成可比较 metrics
hidden tests 没有进入 agent observation
```

## 11. 阶段 3：GRPO Group Rollout Training

GRPO 对同一个 task 采样 K 条 trajectory，用组内相对 reward 作为 advantage。它不需要 value critic，但可以继续使用同一个带 value head 的网络，只是不训练或弱化 value loss。

流程：

```text
for each task:
  rollout_1 = sample(policy, task, temperature, epsilon)
  rollout_2 = sample(policy, task, temperature, epsilon)
  ...
  rollout_K = sample(policy, task, temperature, epsilon)

  rewards = [final_reward_1, ..., final_reward_K]
  group_mean = mean(rewards)
  group_std = std(rewards)
  relative_advantage_i = (reward_i - group_mean) / (group_std + eps)
```

每条 trajectory 的每个 action 使用同一个 trajectory-level relative advantage：

```text
advantage(step in rollout_i) = relative_advantage_i
```

GRPO loss 可以复用 PPO clipping：

```text
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(
  ratio * relative_advantage,
  clip(ratio, 1 - clip_range, 1 + clip_range) * relative_advantage
))
```

推荐配置：

```text
group_size: 4 or 8
tasks_per_epoch: 30
update_epochs: 2
temperature: 1.0
epsilon: 0.1
clip_range: 0.2
entropy_coef: 0.01
learning_rate: 1e-4
```

输出：

```text
runs/checkpoints/grpo.json
runs/checkpoints/grpo.pt
runs/training/grpo/group_rollouts.json
runs/training/grpo/grpo_metrics.json
```

`group_rollouts.json` 至少包含：

```text
task_id
group_id
rollout_id
actions
old_logprobs
final_reward
relative_advantage
hidden_passed
public_passed
tool_calls
winner
failure_type optional
```

验收标准：

```text
同一 task 有 K 条采样轨迹
group artifact 能看出 winner/loser
grpo.pt 可 eval
report 能比较 sft/ppo/grpo 的 hidden pass rate 和 tool cost
```

## 12. LearnedPolicyAgent 推理逻辑

`LearnedPolicyAgent` 应支持：

```text
checkpoint.json fallback
checkpoint.pt torch policy
device cuda/cpu
temperature
epsilon
deterministic eval
stochastic training
```

推理流程：

```text
memory + task
  -> PolicyFeatureEncoder
  -> PolicyBatch
  -> TrajectoryTransformerPolicy.forward()
  -> masked logits
  -> sample or argmax action
  -> build tool_input
  -> AgentDecision(action, tool_input, policy_logprob)
```

eval 默认用 deterministic：

```text
argmax(masked_logits)
```

PPO/GRPO rollout 默认用 stochastic：

```text
Categorical(logits=masked_logits / temperature).sample()
epsilon-greedy optional
```

## 13. Checkpoint 规格

`.json` 用于可读 metadata 和无 torch fallback：

```json
{
  "algorithm": "ppo",
  "training_target": "tool_policy",
  "patch_generation": "expert_patch_provider",
  "scripted_patch": true,
  "torch_checkpoint": "runs/checkpoints/ppo.pt",
  "action_scores": {},
  "metrics": {}
}
```

`.pt` 用于真实推理和继续训练：

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),  # optional for resume
    "model_config": {...},
    "action_list": ACTIONS,
    "tokenizer_config": {...},
    "feature_schema": {...},
    "training_config": {...},
    "metadata": {
        "algorithm": "ppo",
        "training_target": "tool_policy",
        "patch_generation": "expert_patch_provider",
        "scripted_patch": True,
    },
}
```

## 14. Hidden-Test 边界

训练不能破坏 hidden-test 隔离。

必须保持：

```text
ToolLayer(... allow_hidden_tests=False)
run_tests action always uses {"scope": "public"}
hidden tests only run in runner final evaluation
feature encoder never reads hidden test files
observation_tokens never include hidden test source
rollout buffer may store final_reward/hidden_passed only after episode ends
```

也就是说，hidden reward 可以作为训练标签，但不能成为 agent 决策时的 observation。

## 15. 最小实现顺序

推荐按照这个顺序实现：

```text
1. 新增 policy.py
   - tokenizer
   - feature encoder
   - action mask
   - TrajectoryTransformerPolicy
   - checkpoint save/load

2. 重写 train_sft
   - 用 ScriptedAgent 生成真实 trajectories
   - 从 trajectory prefixes 构造 supervised samples
   - 训练 transformer policy
   - 写 sft.json / sft.pt

3. 改 LearnedPolicyAgent
   - 支持加载 .pt
   - 支持 cuda
   - 支持 deterministic/stochastic
   - 返回 policy_logprob

4. 实现 PPO rollout collection
   - 调用 run_episode
   - 保存 old_logprob/value/reward/action_mask
   - 先确保 rollouts artifact 正确

5. 实现 PPO update
   - return/advantage
   - clipped objective
   - value loss
   - entropy

6. 实现 GRPO group sampling
   - 同一 task 采样 K 条 trajectory
   - 计算 group-relative advantage
   - 写 group_rollouts.json

7. 实现 GRPO update
   - PPO-like clipping
   - trajectory-level advantage

8. 扩展 report
   - 显示 training_target
   - 显示 patch_generation
   - 比较 sft/ppo/grpo
```

## 16. 必须新增测试

推荐测试：

```text
test_policy_encoder_shapes
test_action_mask_never_allows_hidden_tests
test_transformer_policy_forward_shapes
test_sft_writes_pt_checkpoint
test_learned_policy_agent_loads_pt_and_decides
test_learned_policy_agent_records_logprob
test_ppo_collects_real_rollouts
test_ppo_writes_trainable_checkpoint
test_grpo_writes_group_rollouts
test_training_does_not_expose_hidden_tests_to_tools
```

测试仍然用 `tmp_path` 生成小 benchmark，避免提交 `data/` 和 `runs/`。

## 17. 现在不要做的事

P0/P1 暂时不要做：

```text
训练 patch generator
让 policy 输出代码 diff
把 expert patch 写进 TaskSpec
让 hidden tests 进入 observation
接 SWE-bench Lite
引入大模型 fine-tuning
把 runner 改成依赖 torch
```

runner 仍然应该是纯 evaluation/runtime 逻辑。torch 相关代码放在 policy/training/agent loading 路径里。

## 18. 后续 patch generation 路线

当 tool policy 稳定后，再进入 patch generator 路线：

```text
1. ReAct/API patch generator 替代 expert patch provider
2. 收集 successful/failed patch traces
3. 用 expert diffs 做 patch SFT
4. 用 chosen/rejected patch 或 trajectory 做 DPO
5. 最后做 end-to-end PPO/GRPO
```

成熟系统会分成两层：

```text
Tool Policy:
  decides when to read/search/patch/test/finish

Patch Generator:
  decides what exact code edit to apply
```

两层共享同一个 SWE-bench-style evaluation harness 和 hidden-test reward，但训练数据、模型结构和优化目标应该分开设计。
