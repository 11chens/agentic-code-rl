# PPO、GRPO、DPO 原理与 Agentic Code RL 对应关系

这篇文档解释 PPO、GRPO、DPO 的核心思想，并说明它们放到 `agentic-code-rl` 这种代码修复评测框架和 tool-using agent runtime 里时，到底在优化什么。这里不把 MDP loop 或 PPO/GRPO 算法本身称为 harness；harness 只指受控评测和工具执行边界。

## 1. 先统一问题形式

在传统机器人 RL 中，你可能熟悉：

```text
observation -> policy -> continuous action -> environment -> reward
```

在本项目里，对应形式是：

```text
observation/memory -> policy -> tool action -> code workspace -> reward
```

一次 trajectory 可能是：

```text
list_files
-> search_code("is_prime")
-> read_file("src/buggy_lib.py")
-> apply_patch(...)
-> run_tests("public")
-> finish
```

reward 来自：

```text
+ hidden tests pass
+ public tests pass
+ failure count improves
- tool call cost
- invalid action
- syntax/import error
```

所以这里的 policy 不一定是大 LLM 本体，也可以是一个小型高层策略：

```text
π(action | memory, task, tool history)
```

其中 action 是：

```text
list_files / read_file / search_code / apply_patch / run_tests / inspect_failure / finish
```

## 2. PPO：限制更新幅度的 on-policy policy gradient

### 2.1 PPO 想解决什么

普通 policy gradient 的问题是：如果一次更新太大，policy 可能崩。

PPO 的核心是：

> 鼓励高 advantage 动作，但限制新旧 policy 概率比不要偏离太多。

### 2.2 核心公式直觉

定义新旧 policy 概率比：

```text
r_t(θ) = πθ(a_t | s_t) / π_old(a_t | s_t)
```

如果某个动作 advantage 为正，说明它比预期好，应提高概率。

如果 advantage 为负，说明它比预期差，应降低概率。

但 PPO 不允许概率变化太猛：

```text
L_PPO = min(
  r_t(θ) * A_t,
  clip(r_t(θ), 1 - ε, 1 + ε) * A_t
)
```

直觉：

```text
如果新 policy 把好动作概率提高太多，clip 会截断收益。
如果新 policy 把坏动作概率降低太多，也会被限制。
```

### 2.3 PPO 需要什么

PPO 通常需要：

- on-policy rollouts
- action logprob
- reward
- advantage estimate
- value function / critic

advantage 常见形式：

```text
A_t = G_t - V(s_t)
```

其中：

- `G_t`：从 t 开始的实际回报
- `V(s_t)`：critic 估计的状态价值

### 2.4 放到代码修复 Agent 中

状态可以是：

```json
{
  "step": 4,
  "has_read_target": true,
  "has_patch": false,
  "last_test_failed": false,
  "action_counts": {
    "read_file": 1,
    "search_code": 1
  }
}
```

动作可以是：

```text
apply_patch
```

如果后续 hidden tests pass，`apply_patch` 这一步可能得到正 advantage。

如果后续语法错误，`apply_patch` 这一步应该得到负 advantage。

### 2.5 PPO 在本项目里的合理角色

适合训练：

```text
高层 tool/action policy
```

也就是学会：

- 什么时候读文件
- 什么时候搜索
- 什么时候 patch
- 什么时候测试
- 测试失败后是否 inspect_failure
- 什么时候 finish

不建议一开始直接训练大 LLM。成本太高，工程复杂。

## 3. GRPO：用组内相对奖励替代 critic 的 PPO-like 方法

### 3.1 GRPO 想解决什么

PPO 的一个工程负担是 critic/value model。

在 LLM 场景中，训练 value model 很贵，也不稳定。

GRPO 的核心思想是：

> 对同一个任务采样一组输出或轨迹，用组内相对 reward 估计 advantage，不单独训练 critic。

### 3.2 核心流程

对同一个 task 采样 K 条 trajectory：

```text
τ1 reward = 1.4
τ2 reward = 0.3
τ3 reward = -0.2
τ4 reward = 0.8
```

计算组内均值：

```text
mean = (1.4 + 0.3 - 0.2 + 0.8) / 4 = 0.575
```

再计算相对 advantage：

```text
A_i = (R_i - mean) / std
```

直觉：

```text
同一题里，比其他轨迹好的轨迹被增强；
比其他轨迹差的轨迹被削弱。
```

### 3.3 GRPO 和 PPO 的关系

GRPO 通常仍保留 PPO-like clipping：

```text
限制新旧 policy 概率比
```

但 advantage 不来自 critic，而来自 group-relative reward。

所以可以理解为：

```text
PPO = critic-based advantage + clipped update
GRPO = group-relative advantage + clipped update
```

### 3.4 放到代码修复 Agent 中

任务：

```text
Fix parse_int_list so it trims whitespace and ignores empty fields.
```

采样 4 条轨迹：

```text
τ1:
search_code -> read_file -> apply correct patch -> run_tests -> finish
reward = 1.4

τ2:
read_file -> patch only strip whitespace -> run_tests -> finish
reward = 0.3

τ3:
read_file -> bad syntax patch -> run_tests -> finish
reward = -0.2

τ4:
list_files -> finish
reward = -0.2
```

GRPO 会增强 τ1 中动作序列的概率，降低 τ3/τ4 这类轨迹概率。

### 3.5 GRPO 的优点

- 不需要单独 value model。
- 很适合有 verifier 的任务。
- 很适合同一 task 多采样。
- 对代码、数学、工具调用任务都自然。

### 3.6 GRPO 的缺点

- 需要对同一任务采样多条轨迹，rollout 成本高。
- 如果组内轨迹都很差，relative advantage 可能噪声大。
- 只知道整条轨迹好坏，不一定知道哪一步关键。
- 仍然可能 reward hacking。

### 3.7 在本项目里的合理角色

适合做：

```text
同一 bug task 多次 rollout
比较不同 tool-use trajectory
用 hidden tests + cost 得到 group reward
更新高层 action policy
```

## 4. DPO：直接从偏好对中学习，不显式跑 RL rollout

### 4.1 DPO 想解决什么

RLHF 通常流程是：

```text
收集偏好数据
训练 reward model
用 PPO 优化 policy
```

DPO 简化了这个过程：

> 不训练显式 reward model，直接用 preferred/rejected pairs 优化 policy。

### 4.2 偏好数据形式

单轮 LLM 里是：

```text
prompt x
preferred response y_w
rejected response y_l
```

代码 agent 里可以是：

```text
task x
preferred trajectory τ_good
rejected trajectory τ_bad
```

### 4.3 核心直觉

DPO 让 policy 满足：

```text
πθ(τ_good | task) > πθ(τ_bad | task)
```

并相对 reference policy 做 KL 约束。

不用显式 reward model，也不用在线 rollouts。

### 4.4 放到代码修复 Agent 中

同一个任务有两条轨迹：

Good：

```text
run_tests -> inspect_failure -> read target -> apply minimal patch -> run_tests -> finish
hidden pass
```

Bad：

```text
read wrong file -> apply syntax error patch -> run_tests -> finish
hidden fail
```

构造成偏好对：

```json
{
  "task_id": "task_001",
  "chosen": ["run_tests", "inspect_failure", "read_file", "apply_patch", "run_tests", "finish"],
  "rejected": ["read_file", "apply_patch", "run_tests", "finish"]
}
```

DPO 会让模型更偏向 chosen trajectory。

### 4.5 DPO 的优点

- 不需要在线环境交互。
- 不需要训练 reward model。
- 适合利用已有 logs。
- 训练流程相对稳定。

### 4.6 DPO 的缺点

- 依赖偏好数据质量。
- 不直接探索新策略。
- 如果 chosen/rejected 都来自弱策略，提升有限。
- 对长 trajectory 的 token-level attribution 不一定清晰。

### 4.7 在本项目里的合理角色

可以用于：

- 从 scripted success vs bad patch failure 构造偏好对。
- 从 GRPO group rollout 中选 winner/loser。
- 从失败恢复轨迹中学习 reflection 策略。

例如：

```text
chosen: failed test -> inspect_failure -> correct patch
rejected: failed test -> random patch -> fail again
```

## 5. 三者对比

| 维度 | PPO | GRPO | DPO |
|---|---|---|---|
| 数据来源 | on-policy rollout | 同任务 group rollouts | 离线偏好对 |
| 是否需要环境交互 | 需要 | 需要 | 不一定需要 |
| 是否需要 critic | 通常需要 | 不需要 | 不需要 |
| 是否直接用 reward | 是 | 是 | 间接用偏好 |
| 是否适合 verifier 任务 | 适合 | 很适合 | 适合离线化 |
| 长时序 credit assignment | 依赖 advantage/critic | 轨迹级相对比较 | 偏好级比较 |
| 工程成本 | 中高 | 中高 | 中 |
| 探索能力 | 有 | 有 | 弱 |

## 6. 在 agentic-code-rl 中的推荐实现顺序

### 第一步：SFT baseline

先用 scripted expert trajectories 学基本行为：

```text
list -> search -> read -> patch -> test -> finish
```

目的：

```text
让 policy 学会基本工具顺序。
```

### 第二步：PPO high-level tool policy

训练小型 policy：

```text
π(action | memory_features)
```

让它学：

- patch 后测试
- 失败后 inspect
- 不要过早 finish
- 控制工具成本

### 第三步：GRPO group rollout

每个 task 采样 K 条 trajectory：

```text
K = 4 或 8
```

用：

```text
hidden pass + public progress - cost
```

做组内相对 advantage。

### 第四步：DPO trajectory preference

把历史轨迹构造成偏好对：

```text
success trajectory > failure trajectory
recovered trajectory > repeated failure trajectory
low-cost success > high-cost success
```

用于 offline policy improvement。

## 7. 一个完整例子：同一任务下三种算法怎么用

任务：

```text
Fix binary_search so it returns the correct index or -1.
```

### PPO 视角

一次 rollout 中：

```text
search_code -> read_file -> apply_patch -> run_tests -> finish
```

如果成功，计算每一步 advantage，更新每一步动作概率。

### GRPO 视角

采样多条：

```text
τ1 correct patch, reward 1.4
τ2 off-by-one patch, reward 0.3
τ3 syntax error, reward -0.2
τ4 premature finish, reward -0.2
```

用组内相对 reward 增强 τ1。

### DPO 视角

构造偏好对：

```text
chosen = τ1
rejected = τ3
```

直接训练 policy 偏向 chosen。

## 8. 最重要的理解

PPO、GRPO、DPO 不是三个互斥选择。

在一个成熟 Agentic RL 项目中，它们可以组成流水线：

```text
SFT 建立基本工具能力
PPO 做在线高层策略优化
GRPO 做同题多轨迹相对优化
DPO 利用历史轨迹做离线偏好改进
```

对于 `agentic-code-rl`，最合理的长期路线是：

```text
scripted expert -> SFT
real rollout -> PPO
same-task multi-rollout -> GRPO
trajectory logs -> DPO
```
