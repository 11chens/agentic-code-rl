# Agentic Code RL

Agentic Code RL is a compact research-style project for evaluating and training tool-using code repair agents. The core harness is SWE-bench-style: each episode copies a small buggy Python repository into an isolated workspace, exposes only public tests to the agent, applies patches through guarded tools, and reserves hidden tests for final grading.

The trainable component is a small Transformer tool policy over discrete actions. It learns when to call tools such as `read_file`, `search_code`, `apply_patch`, `run_tests`, and `finish`. The current policy does not generate code diffs itself; `apply_patch` still uses the synthetic expert patch provider.

For a cold-start handoff to another agent, read [docs/COLD_START_HANDOFF.md](docs/COLD_START_HANDOFF.md). For training commands, read [docs/TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md).

## What It Demonstrates

- Agent architecture: planner, memory, tool calls, reflection, trajectory logging.
- RL surfaces: scripted SFT warm start, rollout PPO, and group-relative GRPO over tool actions.
- Evaluation harness: isolated workspaces, hidden-test grading, patch validity, tool cost, pass rate, runtime, trace artifacts.
- Engineering: reproducible synthetic benchmark, PyTorch checkpoints, JSON metadata artifacts, CLI, tests, and reports.

## Quick Start

Use Python 3.11+.

```bash
python -m pip install -e '.[train,dev]'
python -m agentic_code_rl benchmark create --out data/tasks
python -m agentic_code_rl run --task data/tasks/task_001.json --agent scripted
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
python -m agentic_code_rl report --run runs/latest
```

For the full GPU training pipeline:

```bash
bash scripts/gpu_preflight.sh
bash scripts/train_policy.sh
```

The training script runs:

```text
benchmark create -> pytest -> train-sft -> train-ppo -> train-grpo -> eval -> report
```

Generated benchmark data, workspaces, checkpoints, and reports are written under `data/` and `runs/`; both are ignored by Git.

Detailed training setup: [docs/TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md).

## CLI

```bash
python -m agentic_code_rl benchmark create --out data/tasks
python -m agentic_code_rl run --task data/tasks/task_001.json --agent react
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
python -m agentic_code_rl eval --config configs/eval.yaml --agent grpo
python -m agentic_code_rl report --run runs/<run_id>
```

## Reward

```text
+1.00 hidden tests all pass
+0.40 public tests all pass
+0.20 public test failure count improves
-0.01 each tool call
-0.05 invalid tool call
-0.10 patch creates syntax/import failure
-0.20 finish while hidden tests fail
```

## Project Layout

```text
src/agentic_code_rl/
  benchmark.py      synthetic code-repair task generation
  environment.py    isolated workspaces and private hidden-test runner
  tools.py          guarded code-repair tool layer
  agents.py         scripted, ReAct fallback, and learned-policy agents
  policy.py         Transformer tool-policy feature encoder and model
  runner.py         episode loop, memory, reward, trajectory artifacts
  training.py       SFT/PPO/GRPO training entrypoints
  evaluation.py     batch evaluation and summary metrics
  reporting.py      Markdown report generation
```

## Documentation

- [Project design](docs/PROJECT_DESIGN.md)
- [System flow walkthrough](docs/SYSTEM_FLOW_WALKTHROUGH.md)
- [Training quickstart](docs/TRAINING_QUICKSTART.md)
- [Training policy handoff](docs/TRAINING_POLICY_HANDOFF.md)
- [Policy network architecture](docs/POLICY_NETWORK_ARCHITECTURE.md)
- [Policy network I/O walkthrough](docs/POLICY_NETWORK_IO_WALKTHROUGH.md)

## Resume Bullets

- Built a SWE-bench-style code-repair evaluation harness with isolated workspaces, guarded file search, controlled patch application, public-test interaction, private hidden-test grading, and trajectory logging.
- Implemented public/hidden-test reward shaping and comparable ReAct/SFT/PPO/GRPO entrypoints over pass rate, tool cost, invalid patch rate, and runtime.
- Designed a reproducible synthetic benchmark with path guards, hidden-test isolation, separate expert traces, and resumeable training artifacts.
