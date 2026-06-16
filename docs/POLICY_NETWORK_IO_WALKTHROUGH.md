# Policy Network I/O Walkthrough

这份文档只解释一件事：高层 tool policy 的输入输出到底长什么样，它们如何从 `task_001` 的原始 episode 状态一步步变成 Transformer 内部张量，又如何从网络输出变回 `AgentDecision`。

本文对应当前实现：

```text
src/agentic_code_rl/policy.py
src/agentic_code_rl/agents.py:LearnedPolicyAgent
src/agentic_code_rl/training.py
```

网络结构图见 [POLICY_NETWORK_ARCHITECTURE.md](POLICY_NETWORK_ARCHITECTURE.md)。

本文不讨论 patch 生成训练。当前 policy 只输出 tool action，不输出代码 diff。

## 1. 网络解决的问题

policy 学的是：

```text
pi(action | task, memory, tool history)
```

动作空间固定为 7 个：

```text
0 list_files
1 read_file
2 search_code
3 apply_patch
4 run_tests
5 inspect_failure
6 finish
```

当前实现中 `PAD_ACTION_ID = 7`，只用于 history padding，不是可选动作。

网络输出：

```text
logits: [B, 7]
value:  [B]
```

其中：

- `logits` 表示 7 个 tool action 的未归一化分数。
- `value` 是 PPO critic，用来估计当前状态的 expected return。

## 2. 使用的网络结构

当前设计是 Trajectory Transformer，不是 LLM，不生成代码。

默认 4090 配置来自 `configs/*.yaml`：

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
```

整体结构：

```text
task_tokens
  -> text_embedding
  -> mean pooling
  -> task_token: [B, 1, 512]

observation_tokens
  -> text_embedding
  -> mean pooling
  -> observation_token: [B, 1, 512]

global_features
  -> linear projection
  -> global_token: [B, 1, 512]

history step features
  -> action embedding + status embedding + position embedding + numeric projection
  -> step_tokens: [B, 16, 512]

decision_token
  -> learned parameter
  -> [B, 1, 512]

concat:
  [task_token, observation_token, global_token, step_tokens, decision_token]
  -> sequence: [B, 20, 512]

TransformerEncoder
  -> hidden: [B, 20, 512]
  -> take final decision token hidden: [B, 512]

policy_head
  -> logits: [B, 7]

value_head
  -> value: [B]
```

为什么 sequence 长度是 20：

```text
1 task token
+ 1 observation token
+ 1 global token
+ 16 history step tokens
+ 1 decision token
= 20
```

## 3. task_001 原始输入

`data/tasks/task_001.json`：

```json
{
  "id": "task_001",
  "repo_template": "task_001",
  "prompt": "Fix prime detection for edge cases.",
  "public_tests": ["tests/test_public.py"],
  "hidden_tests": ["tests/test_hidden.py"],
  "max_steps": 12,
  "tags": ["logic", "edge-case"],
  "metadata": {
    "function_name": "is_prime",
    "target_file": "src/buggy_lib.py",
    "source_case": "prime_edges"
  }
}
```

visible source：

```python
def is_prime(n):
    if n == 2:
        return True
    for divisor in range(2, n):
        if n % divisor == 0:
            return False
    return True
```

public test：

```python
from buggy_lib import is_prime

def test_common_primes_and_composites():
    assert is_prime(2)
    assert is_prime(3)
    assert not is_prime(4)
```

hidden test 不会进入 policy 输入。hidden pass/fail 只会在 episode 结束后作为训练 reward 标签进入 rollout buffer。

## 4. 选择一个具体决策点

下面用这个决策点做例子：

```text
agent 已经执行了三步：

1. list_files
2. search_code("is_prime")
3. read_file("src/buggy_lib.py")

现在要决定第 4 步做什么。
```

此时合理动作通常是：

```text
apply_patch
```

但网络并不知道规则答案。它只看到当前 task、memory、tool history，然后输出 7 个 action 的 logits。

## 5. Raw Memory 长什么样

当前 memory 中有 3 个 `TrajectoryStep`。

### Step 1

```python
TrajectoryStep(
    observation="Task: Fix prime detection for edge cases.",
    action="list_files",
    tool_input={},
    tool_output="README.md\nsrc/buggy_lib.py\ntests/test_public.py",
    reward_delta=-0.01,
    metadata={"ok": True, "file_count": 3},
)
```

### Step 2

```python
TrajectoryStep(
    observation="...",
    action="search_code",
    tool_input={"query": "is_prime"},
    tool_output=(
        "README.md:5: Target function: `is_prime`.\n"
        "src/buggy_lib.py:1: def is_prime(n):\n"
        "tests/test_public.py:4:     assert is_prime(2)"
    ),
    reward_delta=-0.01,
    metadata={"ok": True, "query": "is_prime", "matches": 3},
)
```

### Step 3

```python
TrajectoryStep(
    observation="...",
    action="read_file",
    tool_input={"path": "src/buggy_lib.py"},
    tool_output="def is_prime(n):\n    if n == 2:\n        return True\n    ...",
    reward_delta=-0.01,
    metadata={"ok": True, "path": "src/buggy_lib.py"},
)
```

这些 step 是 policy encoder 的主要输入。它们来自 episode 内公开交互，不包含 hidden test 内容。

## 6. PolicyFeatureEncoder 的输出

调用：

```python
encoder = PolicyFeatureEncoder(PolicyConfig(max_steps=16))
encoded = encoder.encode(task, steps)
```

得到：

```python
EncodedPolicyInput(
    task_tokens=...,
    observation_tokens=...,
    global_features=...,
    history_actions=...,
    history_statuses=...,
    history_positions=...,
    history_numeric_features=...,
    history_padding_mask=...,
    action_mask=...,
)
```

下面逐项展开。

## 7. task_text -> task_tokens

encoder 先把 `TaskSpec` 拼成 task text：

```text
Fix prime detection for edge cases.
function: is_prime
target_file: src/buggy_lib.py
tags: logic edge-case
```

tokenizer 做：

```text
lowercase
-> regex split
-> 每个 token 用 blake2b hash 映射到 [2, vocab_size)
-> 不足 128 的部分补 0
```

所以：

```text
task_tokens: [128]
```

这里的 `[128]` 是固定长度，不是说原始 task 一定有 128 个词。

含义是：

```text
最多保留 128 个 tokenizer 切出来的 token
如果少于 128 个，就在后面补 padding 0
如果多于 128 个，就截断到前 128 个
```

以 `task_001` 为例，regex split 之后实际只有 23 个有效 token：

```text
fix
prime
detection
for
edge
cases
.
function
:
is_prime
target_file
:
src
/
buggy_lib
.
py
tags
:
logic
edge
-
case
```

所以完整的 `task_tokens` 是：

```text
23 个非 0 token id + 105 个 padding 0 = 128 个整数
```

文档里只展示前 24 个位置，是为了让例子短一点。第 24 个位置已经是第一个 padding：

```text
[49, 3864, 7843, 1484, 2298, 3706, 2446, 934,
 4737, 395, 6119, 4737, 2146, 512, 749, 2446,
 2992, 3157, 4737, 3995, 2298, 1583, 5256, 0]
```

最后的 `0` 是 padding。后面第 25 到第 128 个位置也都是 `0`。

这些 token id 本身没有人工语义。比如 `49` 不等于 “fix” 的某种人工编号；它只是 `fix` 经过 hash 后落到 vocabulary 里的一个整数槽位。语义来自训练后 `text_embedding` 这一层的参数。

进入 batch 后：

```text
task_tokens: LongTensor [B, 128]
```

进入网络后：

```text
text_embedding(task_tokens): [B, 128, 512]
non-pad mean pooling:        [B, 1, 512]
```

这个 `[B, 1, 512]` 就是 `task_token`。

## 8. observation_text -> observation_tokens

如果外部没有显式传入 observation，encoder 会用最近 3 个 step 构造 observation：

```text
Task: Fix prime detection for edge cases.
Recent tool history:
- list_files: README.md src/buggy_lib.py tests/test_public.py
- search_code: README.md:5: Target function: `is_prime`. src/buggy_lib.py:1: def is_prime(n): tests/test_public.py:4: assert is_prime(2)
- read_file: def is_prime(n): if n == 2: return True ...
```

然后 tokenizer 同样 hash 到固定长度：

```text
observation_tokens: [256]
```

实际前 32 个 token id 类似：

```text
[624, 4737, 49, 3864, 7843, 1484, 2298, 3706,
 2446, 4173, 3034, 6070, 4737, 1583, 7210, 4737,
 4108, 2446, 5277, 2146, 512, 749, 2446, 2992,
 2108, 512, 6808, 2446, 2992, 1583, 5712, 4737]
```

进入 batch：

```text
observation_tokens: LongTensor [B, 256]
```

进入网络：

```text
text_embedding(observation_tokens): [B, 256, 512]
non-pad mean pooling:              [B, 1, 512]
```

这个 `[B, 1, 512]` 是 `observation_token`。

## 9. global_features

`global_features` 是当前状态的结构化摘要。当前 schema 一共 25 维：

```text
0  step_index_norm
1  remaining_steps_norm
2  list_files_count_norm
3  read_file_count_norm
4  search_code_count_norm
5  apply_patch_count_norm
6  run_tests_count_norm
7  inspect_failure_count_norm
8  finish_count_norm
9  has_listed_files
10 has_searched_code
11 has_read_target
12 has_applied_patch
13 has_run_public_tests
14 has_inspected_failure
15 last_tool_ok
16 last_tool_invalid
17 last_public_test_passed
18 public_failure_improved
19 has_function_name
20 has_target_file
21 last_public_failure_count_norm
22 patches_applied_norm
23 invalid_tool_calls_norm
24 syntax_or_import_errors_norm
```

注意：归一化分母用 `task.max_steps`。`task_001.max_steps = 12`，虽然 Transformer history 固定长度是 16。

在当前三步之后：

```text
step_index_norm = 3 / 12 = 0.25
remaining_steps_norm = 9 / 12 = 0.75
list_files_count_norm = 1 / 12 = 0.0833
read_file_count_norm = 1 / 12 = 0.0833
search_code_count_norm = 1 / 12 = 0.0833
apply_patch_count_norm = 0
run_tests_count_norm = 0
inspect_failure_count_norm = 0
finish_count_norm = 0
has_listed_files = 1
has_searched_code = 1
has_read_target = 1
has_applied_patch = 0
has_run_public_tests = 0
has_inspected_failure = 0
last_tool_ok = 1
last_tool_invalid = 0
last_public_test_passed = 0
public_failure_improved = 0
has_function_name = 1
has_target_file = 1
last_public_failure_count_norm = 0
patches_applied_norm = 0
invalid_tool_calls_norm = 0
syntax_or_import_errors_norm = 0
```

实际向量：

```text
[
  0.25, 0.75,
  0.0833, 0.0833, 0.0833, 0.0, 0.0, 0.0, 0.0,
  1.0, 1.0, 1.0,
  0.0, 0.0, 0.0,
  1.0, 0.0, 0.0, 0.0,
  1.0, 1.0,
  0.0, 0.0, 0.0, 0.0
]
```

进入 batch：

```text
global_features: FloatTensor [B, 25]
```

进入网络：

```text
global_proj(global_features): [B, 512]
unsqueeze:                    [B, 1, 512]
```

这个 `[B, 1, 512]` 是 `global_token`。

## 10. history_actions

history 长度固定为 `PolicyConfig.max_steps = 16`。

当前只有 3 个真实历史 step，所以前 13 个位置是 padding，最后 3 个位置是真实动作：

```text
history_actions:
[
  7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7,
  0, 2, 1
]
```

解释：

```text
7 = PAD_ACTION_ID
0 = list_files
2 = search_code
1 = read_file
```

为什么真实 step 放在最后：

```text
clipped_steps = steps[-max_steps:]
start = max_steps - len(clipped_steps)
```

也就是说 history 是右对齐的。这样最近的 step 总是在靠近 decision token 的位置。

进入 batch：

```text
history_actions: LongTensor [B, 16]
```

进入网络：

```text
action_embedding(history_actions): [B, 16, 512]
```

## 11. history_statuses

状态 id：

```text
0 pad
1 ok
2 failed
3 invalid
4 test_passed
5 test_failed
```

当前三个历史工具调用都成功，所以：

```text
history_statuses:
[
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0,
  1, 1, 1
]
```

进入 batch：

```text
history_statuses: LongTensor [B, 16]
```

进入网络：

```text
status_embedding(history_statuses): [B, 16, 512]
```

## 12. history_positions

位置 id 固定是：

```text
history_positions:
[
  0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15
]
```

进入 batch：

```text
history_positions: LongTensor [B, 16]
```

进入网络：

```text
position_embedding(history_positions): [B, 16, 512]
```

padding 位置也会有 position embedding，但后面 Transformer 的 padding mask 会让这些位置不参与 attention。

## 13. history_numeric_features

每个历史 step 有 8 个数值特征：

```text
0 reward_delta
1 tool_ok
2 tool_invalid
3 is_public_test
4 public_test_passed
5 failure_count_norm
6 passed_count_norm
7 patch_count_seen_norm
```

当前三个真实 step 都是普通文件/搜索工具，不是测试，也没有 patch，所以每个真实 step 是：

```text
[-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

完整 shape：

```text
history_numeric_features: [16, 8]
```

前 13 行 padding 全 0，最后 3 行是：

```text
[
  [-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
```

进入 batch：

```text
history_numeric_features: FloatTensor [B, 16, 8]
```

进入网络：

```text
step_numeric_proj(history_numeric_features): [B, 16, 512]
```

## 14. history_padding_mask

padding mask 中：

```text
True  = padding，需要 mask 掉
False = 真实 step
```

当前：

```text
history_padding_mask:
[
  True, True, True, True, True, True, True, True,
  True, True, True, True, True,
  False, False, False
]
```

进入 batch：

```text
history_padding_mask: BoolTensor [B, 16]
```

Transformer 最终需要完整 sequence 的 padding mask：

```text
sequence = [task, observation, global, 16 history steps, decision]
```

所以代码会拼出：

```text
prefix_mask = [False, False, False]
history_padding_mask = [True x13, False x3]
suffix_mask = [False]

full_padding_mask: [B, 20]
```

完整含义：

```text
task token:        False
observation token: False
global token:      False
history pad x13:   True
history real x3:   False
decision token:    False
```

## 15. action_mask

`action_mask` 是当前可选动作，shape：

```text
action_mask: BoolTensor [B, 7]
```

动作顺序：

```text
[list_files, read_file, search_code, apply_patch, run_tests, inspect_failure, finish]
```

当前三步后：

```text
{
  "list_files": true,
  "read_file": true,
  "search_code": true,
  "apply_patch": true,
  "run_tests": true,
  "inspect_failure": false,
  "finish": true
}
```

为什么 `inspect_failure=false`：

```text
还没有执行过 run_tests，所以没有 last test result 可以 inspect。
```

为什么 `apply_patch=true`：

```text
当前还没有 apply_patch，第一版最多允许 patch 一次。
```

注意：`hidden/all tests` 不是 policy action，也不可能通过 action mask 打开。`run_tests` 的 tool input 永远由规则生成成：

```json
{"scope": "public"}
```

## 16. to_batch 后的完整输入

单样本时 `B=1`。

```python
batch = encoder.to_batch([encoded], device="cuda")
```

得到：

```text
task_tokens:              LongTensor  [1, 128]
observation_tokens:       LongTensor  [1, 256]
candidate_tokens:         LongTensor  [1, K, L_patch]
candidate_features:       FloatTensor [1, K, F_candidate]
candidate_mask:           BoolTensor  [1, K]
global_features:          FloatTensor [1, 25]
history_actions:          LongTensor  [1, 16]
history_statuses:         LongTensor  [1, 16]
history_positions:        LongTensor  [1, 16]
history_numeric_features: FloatTensor [1, 16, 8]
history_padding_mask:     BoolTensor  [1, 16]
action_mask:              BoolTensor  [1, 7]
```

训练时 `B` 是 minibatch size，例如 `B=128`：

```text
task_tokens:              [128, 128]
observation_tokens:       [128, 256]
candidate_tokens:         [128, K, L_patch]
candidate_features:       [128, K, F_candidate]
candidate_mask:           [128, K]
global_features:          [128, 25]
history_actions:          [128, 16]
history_statuses:         [128, 16]
history_positions:        [128, 16]
history_numeric_features: [128, 16, 8]
history_padding_mask:     [128, 16]
action_mask:              [128, 7]
```

## 17. Transformer 内部怎么处理

forward 入口：

```python
action_logits, patch_candidate_logits, value = model(batch)
```

### 17.1 文本 token

```python
task_token = self._pool_text(batch["task_tokens"])
observation_token = self._pool_text(batch["observation_tokens"])
```

内部：

```text
task_tokens [B,128]
-> text_embedding
-> [B,128,512]
-> mask pad id 0
-> mean pooling
-> [B,1,512]

observation_tokens [B,256]
-> text_embedding
-> [B,256,512]
-> mask pad id 0
-> mean pooling
-> [B,1,512]
```

### 17.2 全局特征

```python
global_token = self.global_proj(batch["global_features"]).unsqueeze(1)
```

形状：

```text
[B,25] -> Linear(25,512) -> [B,512] -> [B,1,512]
```

### 17.3 历史 step token

代码：

```python
step_tokens = (
    action_embedding(history_actions)
    + status_embedding(history_statuses)
    + position_embedding(history_positions)
    + step_numeric_proj(history_numeric_features)
)
```

每一项形状：

```text
action_embedding:   [B,16,512]
status_embedding:   [B,16,512]
position_embedding: [B,16,512]
numeric_projection: [B,16,512]
```

相加后：

```text
step_tokens: [B,16,512]
```

这一步的含义是：每个历史 step 的 token 同时包含：

- 这个 step 做了哪个 action。
- 这个 step 成功、失败、invalid、test pass/fail。
- 这个 step 在 history 里的位置。
- 这个 step 的 reward、测试失败数、是否 public test 等数值信息。

### 17.4 decision token

```python
decision = self.decision_token.expand(batch_size, -1, -1)
```

形状：

```text
decision_token parameter: [1,1,512]
expanded decision:        [B,1,512]
```

这个 token 类似分类任务里的 `[CLS]`。Transformer 会让它 attend 到 task、observation、global 和 history，最后用它来输出当前决策。

### 17.5 拼成完整 sequence

```python
sequence = torch.cat([
    task_token,
    observation_token,
    global_token,
    step_tokens,
    decision,
], dim=1)
```

形状：

```text
[B,1,512]
[B,1,512]
[B,1,512]
[B,16,512]
[B,1,512]
--------------
[B,20,512]
```

### 17.6 TransformerEncoder

```python
hidden = self.transformer(sequence, src_key_padding_mask=padding_mask)
```

输入：

```text
sequence:     [B,20,512]
padding_mask: [B,20]
```

输出：

```text
hidden: [B,20,512]
```

然后取最后一个 token：

```python
decision_hidden = hidden[:, -1, :]
```

形状：

```text
decision_hidden: [B,512]
```

这就是当前状态的最终表示。

## 18. 输出 logits 和 value

```python
logits = self.policy_head(decision_hidden)
value = self.value_head(decision_hidden).squeeze(-1)
```

形状：

```text
logits before mask: [B,7]
value:              [B]
```

然后应用 action mask：

```python
masked_logits = logits.masked_fill(~action_mask, -1e9)
```

对于当前例子，`inspect_failure` 会被设成极小值：

```text
list_files:      raw logit
read_file:       raw logit
search_code:     raw logit
apply_patch:     raw logit
run_tests:       raw logit
inspect_failure: -1e9
finish:          raw logit
```

所以即使模型想选 `inspect_failure`，也选不到。

## 19. logits 如何变成动作

eval 默认 deterministic：

```python
action_id = argmax(masked_logits)
```

PPO/GRPO rollout 默认 stochastic：

```python
dist = Categorical(logits=masked_logits / temperature)
action_id = dist.sample()
logprob = dist.log_prob(action_id)
entropy = dist.entropy()
```

假设当前输出类似：

```text
list_files:      -1.8
read_file:       -0.4
search_code:     -0.7
apply_patch:      2.6
run_tests:        0.1
inspect_failure: -1e9
finish:          -1.2
```

那么 deterministic eval 会选：

```text
action_id = 3
action = apply_patch
```

同时记录：

```text
policy_logprob = log softmax(masked_logits)[3]
policy_entropy = entropy(masked distribution)
policy_value = value[0]
```

## 20. action 如何变成 AgentDecision

policy 只输出 action，不输出 tool input。tool input 由规则生成：

```python
tool_input_for_action(task, "apply_patch")
```

当前 `task_001` 会根据：

```json
{
  "source_case": "prime_edges"
}
```

从 patch candidate provider 取到被选中的 payload：

```json
{
  "path": "src/buggy_lib.py",
  "find": "def is_prime(n):\n    if n == 2:\n        return True\n    ...",
  "replace": "def is_prime(n):\n    if n < 2:\n        return False\n    ..."
}
```

最后返回：

```python
AgentDecision(
    action="apply_patch",
    tool_input={
        "path": "src/buggy_lib.py",
        "find": "...",
        "replace": "...",
    },
    rationale="Torch policy selected action.",
    policy_logprob=-0.13,
    metadata={
        "policy_value": 1.21,
        "policy_entropy": 0.42,
        "action_mask": [True, True, True, True, True, False, True],
        "checkpoint_path": "runs/checkpoints/sft.pt",
        "temperature": 1.0,
        "action_id": 3,
    },
)
```

数值只是示意，实际取决于训练后的模型参数。

## 21. Runner 如何使用这个输出

`runner.py` 做：

```python
decision = agent.decide(memory)
result = tools.call(decision.action, decision.tool_input)
reward_delta = _reward_delta(...)
memory.steps.append(TrajectoryStep(...))
```

此时工具层真正执行：

```text
ToolLayer.apply_patch(payload)
```

如果 patch 成功，trajectory step 里会记录：

```text
action: apply_patch
tool_input: patch payload
tool_output: unified diff
reward_delta: -0.01
policy_logprob: 来自网络
metadata:
  ok: true
  path: src/buggy_lib.py
  policy_value: ...
  policy_entropy: ...
  action_mask: ...
```

这些 metadata 后续 PPO/GRPO 会用来计算 loss。

## 22. SFT 阶段样本怎么构造

SFT 不直接手写状态，而是先用 `ScriptedAgent` 跑真实 episode：

```text
list_files
search_code
read_file
apply_patch
run_tests
finish
```

然后从 trajectory prefixes 构造样本。

对于 `task_001`，SFT 样本大致是：

```text
sample 0:
  input = empty memory
  label = list_files

sample 1:
  input = [list_files]
  label = search_code

sample 2:
  input = [list_files, search_code]
  label = read_file

sample 3:
  input = [list_files, search_code, read_file]
  label = apply_patch

sample 4:
  input = [list_files, search_code, read_file, apply_patch]
  label = run_tests

sample 5:
  input = [list_files, search_code, read_file, apply_patch, run_tests]
  label = finish
```

每个 sample 都会经过同一个 encoder，得到 `PolicyBatch`。训练目标：

```text
cross_entropy(masked_logits, expert_action)
```

可选 value warmup：

```text
MSE(value, discounted_return)
```

## 23. PPO 阶段样本怎么用

PPO rollout 不是 expert label，而是模型自己采样：

```text
state_t -> policy -> action_t -> ToolLayer -> reward_t -> next state
```

每一步保存：

```text
encoded state
action_id
old_logprob
value
reward
action_mask
```

episode 结束后：

```text
terminal_bonus = trajectory.final_reward - sum(step_reward_delta)
last_step_reward += terminal_bonus
```

然后计算：

```text
return_t
advantage_t = return_t - value_t
```

PPO 更新时重新 forward 同样的 encoded state：

```text
new_logits, new_value = model(encoded_state)
new_logprob = log_prob(action_id)
ratio = exp(new_logprob - old_logprob)
clipped PPO loss
value loss
entropy bonus
```

## 24. GRPO 阶段样本怎么用

GRPO 对同一个 task 采样 K 条 trajectory。

例如 `task_001`：

```text
rollout 1 final_reward = 1.74
rollout 2 final_reward = 0.18
rollout 3 final_reward = -0.25
rollout 4 final_reward = 1.12
```

计算：

```text
group_mean = mean(rewards)
relative_advantage_i = (reward_i - group_mean) / (group_std + eps)
```

rollout 1 的每一步 action 都用同一个正 advantage；rollout 3 的每一步 action 都用负 advantage。

GRPO 的输入 batch 和 PPO 一样，区别是 advantage 来自同题组内相对 reward，不来自 critic。

## 25. Hidden Tests 在哪里

hidden tests 不出现在：

```text
task_tokens
observation_tokens
global_features
history_actions
history_statuses
history_numeric_features
action_mask
```

hidden tests 只在 runner final evaluation 之后影响：

```text
trajectory.final_reward
trajectory.hidden_passed
```

这些字段可以作为训练标签进入 PPO/GRPO 的 return 或 group reward，但不会成为 agent 当步决策的 observation。

这是评测边界，不能改。

## 26. 一句话总结

对于 `task_001`，网络看到的是：

```text
任务文字 + 最近公开工具历史 + 25 维状态摘要 + 16 步历史 token
```

网络内部把它们变成：

```text
[task, observation, global, step_1...step_16, decision] = [B,20,512]
```

Transformer 输出：

```text
7 个 tool action logits + 1 个 value
```

最后规则层把 action 变成实际 tool input。当前阶段训练的是“什么时候读、搜、patch、测试、结束”，不是“patch 具体怎么写”。
