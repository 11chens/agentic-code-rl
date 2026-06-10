# Agentic Code RL

Agentic Code RL is a compact research-style project for training and evaluating a tool-using code repair agent. Each episode copies a small buggy Python repository into an isolated workspace. The agent reads files, searches code, applies a patch, runs public tests, reflects on failures, and stops when it believes the task is solved. Hidden tests are reserved for final reward and evaluation.

The project is intentionally not embodied: it targets Agentic RL concepts that can be verified on a normal laptop.

For a cold-start handoff to another agent, read [docs/COLD_START_HANDOFF.md](docs/COLD_START_HANDOFF.md).

## What It Demonstrates

- Agent architecture: planner, memory, tool calls, reflection, trajectory logging.
- RL surfaces: SFT baseline, PPO-style clipped objective, GRPO-style group-relative objective.
- Evaluation: public vs hidden tests, patch validity, tool cost, pass rate, runtime, trace artifacts.
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
  environment.py    isolated workspaces and public/hidden test runner
  tools.py          guarded code-repair tool layer
  agents.py         scripted, ReAct fallback, and learned-policy agents
  runner.py         episode loop, memory, reward, trajectory artifacts
  training.py       SFT/PPO/GRPO-style training entrypoints
  evaluation.py     batch evaluation and summary metrics
  reporting.py      Markdown report generation
```

## Resume Bullets

- Built an Agentic RL code-repair environment where agents solve Python bugs through guarded file search, patch generation, test execution, failure reflection, and trajectory logging.
- Implemented public/hidden-test reward shaping and compared ReAct, SFT, PPO-style, and GRPO-style policies on pass rate, tool cost, invalid patch rate, and runtime.
- Designed a reproducible synthetic benchmark with isolated workspaces, path guards, deterministic expert traces, and resumeable training artifacts.
