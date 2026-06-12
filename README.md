# Agentic Code RL

Agentic Code RL is a compact research-style project for evaluating and training tool-using code repair agents. The core harness is SWE-bench-style: each episode copies a small buggy Python repository into an isolated workspace, exposes only public tests to the agent, applies patches through guarded tools, and reserves hidden tests for final grading.

The project is intentionally not embodied: it targets Agentic RL concepts that can be verified on a normal laptop.

For a cold-start handoff to another agent, read [docs/COLD_START_HANDOFF.md](docs/COLD_START_HANDOFF.md).

## What It Demonstrates

- Agent architecture: planner, memory, tool calls, reflection, trajectory logging.
- RL surfaces: SFT baseline plus lightweight PPO/GRPO training scaffolds.
- Evaluation harness: isolated workspaces, hidden-test grading, patch validity, tool cost, pass rate, runtime, trace artifacts.
- Engineering: reproducible synthetic benchmark, CLI, JSON artifacts, tests, and reports.

## Quick Start

Use Python 3.11+.

```powershell
cd D:\Vibethon\agentic-code-rl
python -m pip install -e .[dev]
python -m agentic_code_rl benchmark create --out data/tasks
python -m agentic_code_rl run --task data/tasks/task_001.json --agent scripted
python -m agentic_code_rl eval --config configs/eval.yaml --agent scripted
python -m agentic_code_rl report --run runs/latest
```

If this laptop cannot train with PyTorch, commit and push the repository, then run the training extras on your development machine:

```powershell
git init
git add .
git commit -m "Initial agentic code RL project"
git remote add origin https://github.com/<you>/agentic-code-rl.git
git push -u origin main
```

On the development machine:

```powershell
git clone https://github.com/<you>/agentic-code-rl.git
cd agentic-code-rl
python -m pip install -e .[train,llm,dev]
python -m agentic_code_rl benchmark create --out data/tasks
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
```

Training commands work without mutating the benchmark. Install `.[train]` to use PyTorch; without PyTorch, the commands write deterministic JSON fallback checkpoints so the CLI remains inspectable.

```powershell
python -m pip install -e .[train,dev]
python -m agentic_code_rl train-sft --config configs/sft.yaml
python -m agentic_code_rl train-ppo --config configs/ppo.yaml
python -m agentic_code_rl train-grpo --config configs/grpo.yaml
```

## CLI

```powershell
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
  runner.py         episode loop, memory, reward, trajectory artifacts
  training.py       SFT/PPO/GRPO-style training entrypoints
  evaluation.py     batch evaluation and summary metrics
  reporting.py      Markdown report generation
```

## Resume Bullets

- Built a SWE-bench-style code-repair evaluation harness with isolated workspaces, guarded file search, controlled patch application, public-test interaction, private hidden-test grading, and trajectory logging.
- Implemented public/hidden-test reward shaping and comparable ReAct/SFT/PPO-style/GRPO-style entrypoints over pass rate, tool cost, invalid patch rate, and runtime.
- Designed a reproducible synthetic benchmark with path guards, hidden-test isolation, separate expert traces, and resumeable training artifacts.
