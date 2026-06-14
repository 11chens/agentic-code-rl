# Project Design

## Architecture

This project uses "harness" narrowly. The harness is the code-repair evaluation and tool-execution boundary: isolated workspace setup, guarded file operations, controlled patch application, public/hidden test separation, timeout handling, and standard artifacts. The MDP-style loop and PPO/GRPO updates are training logic, not separate harness concepts.

```text
TaskSpec
  -> EpisodeWorkspace
  -> ToolLayer
  -> Agent Memory
  -> Agent Decision
  -> ToolResult
  -> Reward + Trajectory
  -> Public/Hidden Evaluation
```

The language model surface is intentionally narrow. API models can be added at the `ReactAgent.decide()` boundary, while the trainable component remains a small discrete policy over tool actions.

## Hidden Test Boundary

During an episode, `run_tests` only exposes public tests. Hidden tests are stored outside the visible workspace, are filtered from file tools, and are executed by the runner after the agent finishes or reaches `max_steps`. This keeps final grading separate from interaction feedback.

## Expert Trace Boundary

Task JSON files do not contain expert patches. Synthetic expert patches are written as separate benchmark artifacts for scripted smoke tests and SFT data generation. ReAct/API agents should not depend on these artifacts.

## Training Story

- SFT learns the scripted expert tool sequence.
- PPO runs rollout training over the same discrete tool-action space with a clipped objective.
- GRPO runs same-task group-relative rollout training over tool actions.

The default trainable model is a compact Transformer encoder, not an LLM. It outputs action logits and a scalar value estimate; it does not generate patch text. `apply_patch` still receives patch payloads from the synthetic expert patch provider.

Training writes both readable JSON metadata and PyTorch `.pt` checkpoints when `torch` is installed. The JSON fallback path is kept only so the CLI remains inspectable in non-training environments.
