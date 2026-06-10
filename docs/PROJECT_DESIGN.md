# Project Design

## Architecture

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

During an episode, `run_tests` only exposes public tests. Hidden tests are executed by the runner after the agent finishes or reaches `max_steps`. This keeps final reward separate from interaction feedback.

## Training Story

- SFT learns the scripted expert sequence.
- PPO-style training writes a clipped-objective checkpoint over the same discrete action space.
- GRPO-style training samples a group of trajectory variants and boosts actions from the best relative trajectory.

The default implementation is deliberately lightweight so it can run locally without a GPU. Installing `.[train]` enables PyTorch training artifacts in addition to JSON checkpoints.
