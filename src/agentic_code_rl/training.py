from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
import math
import random

from .agents import ScriptedAgent, save_policy_checkpoint
from .config import load_config
from .policy import (
    ACTION_TO_ID,
    ACTIONS,
    PolicyConfig,
    PolicyFeatureEncoder,
    TrajectoryTransformerPolicy,
    choose_action_from_logits,
    default_device,
    load_torch_policy_checkpoint,
    save_torch_policy_checkpoint,
    tool_input_for_action,
    torch_available,
)
from .runner import run_episode
from .schemas import AgentDecision, TaskSpec, Trajectory, TrajectoryStep, load_task, write_json


@dataclass(slots=True)
class TrainSample:
    encoded: Any
    action_id: int
    old_logprob: float
    value: float
    reward: float
    ret: float
    advantage: float
    task_id: str


class TorchRolloutAgent:
    def __init__(
        self,
        model: Any,
        encoder: PolicyFeatureEncoder,
        device: str,
        name: str = "policy_rollout",
        temperature: float = 1.0,
        epsilon: float = 0.05,
        deterministic: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon
        self.deterministic = deterministic

    def decide(self, memory: Any) -> AgentDecision:
        import torch

        encoded = self.encoder.encode(memory.task, memory.steps, memory.observation())
        batch = self.encoder.to_batch([encoded], device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(batch)
        action_id, logprob, entropy = choose_action_from_logits(
            logits,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon=self.epsilon if not self.deterministic else 0.0,
        )
        action = ACTIONS[action_id]
        return AgentDecision(
            action,
            tool_input_for_action(memory.task, action),
            rationale="Training policy rollout selected action.",
            policy_logprob=logprob,
            metadata={
                "policy_value": float(value.squeeze(0).detach().cpu().item()),
                "policy_entropy": entropy,
                "action_mask": encoded.action_mask,
                "temperature": self.temperature,
                "action_id": action_id,
            },
        )


def train_sft(config_path: Path | None) -> Path:
    config = _training_config(config_path, "sft")
    task_paths = _task_paths(Path(config["tasks_dir"]), int(config.get("limit", 0) or 0))
    counts = _expert_action_counts(task_paths)
    scores = _scores_from_counts(counts)
    scores = _merge_resume_scores(scores, config)
    checkpoint = Path(config["checkpoint"])

    if torch_available():
        torch_status = _train_sft_torch(task_paths, config, checkpoint)
    else:
        torch_status = _try_torch_import()

    torch_path = checkpoint.with_suffix(".pt")
    metadata = _checkpoint_metadata(
        algorithm="sft",
        tasks=len(task_paths),
        torch_status=torch_status,
        torch_checkpoint=str(torch_path) if torch_path.exists() else None,
    )
    save_policy_checkpoint(checkpoint, scores, metadata)
    _write_replay_buffer(Path(config["output_dir"]) / "replay_buffer.json", task_paths)
    write_json(
        Path(config["output_dir"]) / "sft_metrics.json",
        {"action_counts": dict(counts), "checkpoint": str(checkpoint), "torch_status": torch_status},
    )
    return checkpoint


def train_ppo(config_path: Path | None) -> Path:
    config = _training_config(config_path, "ppo")
    task_paths = _task_paths(Path(config["tasks_dir"]), int(config.get("limit", 0) or 0))
    checkpoint = Path(config["checkpoint"])

    if not torch_available():
        return _train_ppo_fallback(config, task_paths, checkpoint)

    import torch

    device = _training_device(config)
    model, encoder, loaded_from = _load_or_init_policy(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-4)))
    epochs = int(config.get("epochs", 3))
    rollout_tasks_per_epoch = int(config.get("rollout_tasks_per_epoch", len(task_paths)) or len(task_paths))
    update_epochs = int(config.get("ppo_update_epochs", 4))
    batch_size = int(config.get("minibatch_size", 128))
    gamma = float(config.get("gamma", 0.99))
    clip_range = float(config.get("clip_range", 0.2))
    value_coef = float(config.get("value_coef", 0.5))
    entropy_coef = float(config.get("entropy_coef", 0.01))
    all_artifacts: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, Any]] = []

    for epoch in range(epochs):
        selected = _sample_task_paths(task_paths, rollout_tasks_per_epoch)
        samples, artifacts = _collect_rollouts(
            task_paths=selected,
            repos_dir=Path(config["repos_dir"]),
            runs_dir=Path(config["output_dir"]) / "rollout_runs",
            model=model,
            encoder=encoder,
            device=device,
            temperature=float(config.get("temperature", 1.0)),
            epsilon=float(config.get("epsilon", 0.05)),
            gamma=gamma,
            run_prefix=f"ppo-e{epoch + 1}",
            test_timeout_sec=int(config.get("test_timeout_sec", 10)),
        )
        all_artifacts.extend(artifacts)
        metrics = _ppo_update(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            samples=samples,
            device=device,
            update_epochs=update_epochs,
            batch_size=batch_size,
            clip_range=clip_range,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        metrics.update(_artifact_summary(artifacts))
        metrics["epoch"] = epoch + 1
        epoch_metrics.append(metrics)

    torch_path = checkpoint.with_suffix(".pt")
    metadata = _checkpoint_metadata(
        algorithm="ppo",
        tasks=len(task_paths),
        torch_status=f"torch trained: {torch_path}",
        torch_checkpoint=str(torch_path),
        extra={"loaded_from": loaded_from, "objective": "real rollout clipped PPO over tool actions"},
    )
    save_torch_policy_checkpoint(torch_path, model, encoder, optimizer, dict(config), metadata)
    scores = _scores_from_rollout_artifacts(all_artifacts) or _scores_from_counts(_expert_action_counts(task_paths))
    save_policy_checkpoint(checkpoint, scores, metadata)
    write_json(Path(config["output_dir"]) / "rollouts.json", {"rollouts": all_artifacts})
    write_json(
        Path(config["output_dir"]) / "ppo_metrics.json",
        {"checkpoint": str(checkpoint), "torch_checkpoint": str(torch_path), "epochs": epoch_metrics},
    )
    return checkpoint


def train_grpo(config_path: Path | None) -> Path:
    config = _training_config(config_path, "grpo")
    task_paths = _task_paths(Path(config["tasks_dir"]), int(config.get("limit", 0) or 0))
    checkpoint = Path(config["checkpoint"])

    if not torch_available():
        return _train_grpo_fallback(config, task_paths, checkpoint)

    import torch

    device = _training_device(config)
    model, encoder, loaded_from = _load_or_init_policy(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    epochs = int(config.get("epochs", 3))
    group_size = int(config.get("group_size", 4))
    tasks_per_epoch = int(config.get("tasks_per_epoch", len(task_paths)) or len(task_paths))
    update_epochs = int(config.get("update_epochs", 2))
    batch_size = int(config.get("minibatch_size", 128))
    clip_range = float(config.get("clip_range", 0.2))
    entropy_coef = float(config.get("entropy_coef", 0.01))
    all_groups: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, Any]] = []

    for epoch in range(epochs):
        selected = _sample_task_paths(task_paths, tasks_per_epoch)
        samples: list[TrainSample] = []
        group_artifacts: list[dict[str, Any]] = []
        for group_index, task_path in enumerate(selected, start=1):
            group_samples, group = _collect_grpo_group(
                task_path=task_path,
                repos_dir=Path(config["repos_dir"]),
                runs_dir=Path(config["output_dir"]) / "group_runs",
                model=model,
                encoder=encoder,
                device=device,
                group_size=group_size,
                temperature=float(config.get("temperature", 1.0)),
                epsilon=float(config.get("epsilon", 0.1)),
                group_id=f"e{epoch + 1}-g{group_index}",
                test_timeout_sec=int(config.get("test_timeout_sec", 10)),
            )
            samples.extend(group_samples)
            group_artifacts.append(group)
        all_groups.extend(group_artifacts)
        metrics = _grpo_update(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            samples=samples,
            device=device,
            update_epochs=update_epochs,
            batch_size=batch_size,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
        )
        flat_rollouts = [rollout for group in group_artifacts for rollout in group["rollouts"]]
        metrics.update(_artifact_summary(flat_rollouts))
        metrics["epoch"] = epoch + 1
        epoch_metrics.append(metrics)

    torch_path = checkpoint.with_suffix(".pt")
    metadata = _checkpoint_metadata(
        algorithm="grpo",
        tasks=len(task_paths),
        torch_status=f"torch trained: {torch_path}",
        torch_checkpoint=str(torch_path),
        extra={
            "loaded_from": loaded_from,
            "objective": "real same-task group-relative rollout over tool actions",
            "group_size": group_size,
        },
    )
    save_torch_policy_checkpoint(torch_path, model, encoder, optimizer, dict(config), metadata)
    flat_artifacts = [rollout for group in all_groups for rollout in group["rollouts"]]
    scores = _scores_from_rollout_artifacts(flat_artifacts) or _scores_from_counts(_expert_action_counts(task_paths))
    save_policy_checkpoint(checkpoint, scores, metadata)
    write_json(Path(config["output_dir"]) / "group_rollouts.json", {"groups": all_groups})
    write_json(
        Path(config["output_dir"]) / "grpo_metrics.json",
        {"checkpoint": str(checkpoint), "torch_checkpoint": str(torch_path), "epochs": epoch_metrics},
    )
    return checkpoint


def _train_sft_torch(task_paths: list[Path], config: dict[str, Any], checkpoint: Path) -> str:
    import torch
    import torch.nn.functional as F

    device = _training_device(config)
    encoder = PolicyFeatureEncoder(config=_policy_config_from_config(config))
    model = TrajectoryTransformerPolicy(
        encoder.config,
        global_feature_dim=encoder.global_feature_dim,
        step_feature_dim=encoder.step_feature_dim,
        num_actions=len(ACTIONS),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-4)))
    trajectories = _scripted_trajectories(task_paths, Path(config["repos_dir"]), Path(config["output_dir"]) / "sft_runs")
    samples: list[TrainSample] = []
    gamma = float(config.get("gamma", 0.99))
    for task_path, trajectory in trajectories:
        task = load_task(task_path)
        samples.extend(_samples_from_trajectory(task, trajectory, encoder, gamma=gamma, use_recorded_advantage=False))
    if not samples:
        raise RuntimeError("No SFT samples were produced")

    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("minibatch_size", 128))
    value_coef = float(config.get("value_coef", 0.1))
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(samples)
        for chunk in _chunks(samples, batch_size):
            batch = encoder.to_batch([sample.encoded for sample in chunk], device=device)
            actions = torch.tensor([sample.action_id for sample in chunk], dtype=torch.long, device=device)
            returns = torch.tensor([sample.ret for sample in chunk], dtype=torch.float32, device=device)
            logits, values = model(batch)
            policy_loss = F.cross_entropy(logits, actions)
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

    torch_path = checkpoint.with_suffix(".pt")
    metadata = _checkpoint_metadata(
        algorithm="sft",
        tasks=len(task_paths),
        torch_status=f"torch trained: {torch_path}",
        torch_checkpoint=str(torch_path),
        extra={"objective": "supervised imitation of scripted tool trajectories", "loss": last_loss},
    )
    save_torch_policy_checkpoint(torch_path, model, encoder, optimizer, dict(config), metadata)
    return f"torch trained: {torch_path}"


def _collect_rollouts(
    task_paths: list[Path],
    repos_dir: Path,
    runs_dir: Path,
    model: Any,
    encoder: PolicyFeatureEncoder,
    device: str,
    temperature: float,
    epsilon: float,
    gamma: float,
    run_prefix: str,
    test_timeout_sec: int,
) -> tuple[list[TrainSample], list[dict[str, Any]]]:
    samples: list[TrainSample] = []
    artifacts: list[dict[str, Any]] = []
    for index, task_path in enumerate(task_paths, start=1):
        agent = TorchRolloutAgent(
            model=model,
            encoder=encoder,
            device=device,
            name="ppo",
            temperature=temperature,
            epsilon=epsilon,
            deterministic=False,
        )
        trajectory = run_episode(
            task_path=task_path,
            repos_dir=repos_dir,
            runs_dir=runs_dir,
            agent=agent,
            run_id=f"{run_prefix}-{index:04d}-{task_path.stem}",
            test_timeout_sec=test_timeout_sec,
        )
        task = load_task(task_path)
        samples.extend(_samples_from_trajectory(task, trajectory, encoder, gamma=gamma, use_recorded_advantage=True))
        artifacts.append(_trajectory_artifact(trajectory))
    return samples, artifacts


def _collect_grpo_group(
    task_path: Path,
    repos_dir: Path,
    runs_dir: Path,
    model: Any,
    encoder: PolicyFeatureEncoder,
    device: str,
    group_size: int,
    temperature: float,
    epsilon: float,
    group_id: str,
    test_timeout_sec: int,
) -> tuple[list[TrainSample], dict[str, Any]]:
    trajectories: list[Trajectory] = []
    rollouts: list[dict[str, Any]] = []
    for rollout_index in range(group_size):
        agent = TorchRolloutAgent(
            model=model,
            encoder=encoder,
            device=device,
            name="grpo",
            temperature=temperature,
            epsilon=epsilon,
            deterministic=False,
        )
        trajectory = run_episode(
            task_path=task_path,
            repos_dir=repos_dir,
            runs_dir=runs_dir,
            agent=agent,
            run_id=f"grpo-{group_id}-r{rollout_index + 1}-{task_path.stem}",
            test_timeout_sec=test_timeout_sec,
        )
        trajectories.append(trajectory)
    rewards = [trajectory.final_reward for trajectory in trajectories]
    group_mean = mean(rewards) if rewards else 0.0
    variance = mean([(reward - group_mean) ** 2 for reward in rewards]) if rewards else 0.0
    group_std = math.sqrt(variance)
    winner_reward = max(rewards) if rewards else 0.0
    samples: list[TrainSample] = []
    task = load_task(task_path)
    for index, trajectory in enumerate(trajectories):
        advantage = (trajectory.final_reward - group_mean) / (group_std + 1e-6)
        samples.extend(
            _samples_from_trajectory(
                task,
                trajectory,
                encoder,
                gamma=1.0,
                use_recorded_advantage=True,
                override_advantage=advantage,
            )
        )
        artifact = _trajectory_artifact(trajectory)
        artifact["relative_advantage"] = advantage
        artifact["winner"] = trajectory.final_reward == winner_reward
        rollouts.append(artifact)
    return samples, {
        "task_id": task.id,
        "group_id": group_id,
        "mean_reward": group_mean,
        "std_reward": group_std,
        "rollouts": rollouts,
    }


def _samples_from_trajectory(
    task: TaskSpec,
    trajectory: Trajectory,
    encoder: PolicyFeatureEncoder,
    gamma: float,
    use_recorded_advantage: bool,
    override_advantage: float | None = None,
) -> list[TrainSample]:
    if not trajectory.steps:
        return []
    rewards = _trajectory_step_rewards(trajectory)
    returns = _discounted_returns(rewards, gamma)
    values = [float(step.metadata.get("policy_value", 0.0) or 0.0) for step in trajectory.steps]
    raw_advantages = [returns[index] - values[index] for index in range(len(returns))]
    samples: list[TrainSample] = []
    for index, step in enumerate(trajectory.steps):
        if step.action not in ACTION_TO_ID:
            continue
        prefix = trajectory.steps[:index]
        encoded = encoder.encode(task, prefix, step.observation)
        advantage = override_advantage if override_advantage is not None else raw_advantages[index]
        samples.append(
            TrainSample(
                encoded=encoded,
                action_id=ACTION_TO_ID[step.action],
                old_logprob=float(step.policy_logprob if step.policy_logprob is not None else 0.0),
                value=values[index],
                reward=rewards[index],
                ret=returns[index],
                advantage=advantage if use_recorded_advantage else returns[index],
                task_id=trajectory.task_id,
            )
        )
    return samples


def _ppo_update(
    model: Any,
    encoder: PolicyFeatureEncoder,
    optimizer: Any,
    samples: list[TrainSample],
    device: str,
    update_epochs: int,
    batch_size: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
) -> dict[str, float]:
    if not samples:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
    import torch
    import torch.nn.functional as F

    _normalize_advantages(samples)
    model.train()
    metrics: list[dict[str, float]] = []
    for _ in range(update_epochs):
        random.shuffle(samples)
        for chunk in _chunks(samples, batch_size):
            batch = encoder.to_batch([sample.encoded for sample in chunk], device=device)
            actions = torch.tensor([sample.action_id for sample in chunk], dtype=torch.long, device=device)
            old_logprobs = torch.tensor([sample.old_logprob for sample in chunk], dtype=torch.float32, device=device)
            returns = torch.tensor([sample.ret for sample in chunk], dtype=torch.float32, device=device)
            advantages = torch.tensor([sample.advantage for sample in chunk], dtype=torch.float32, device=device)
            logits, values = model(batch)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            approx_kl = torch.mean(old_logprobs - new_logprobs)
            metrics.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "clip_fraction": float(clip_fraction.detach().cpu().item()),
                    "approx_kl": float(approx_kl.detach().cpu().item()),
                }
            )
    return _mean_metrics(metrics)


def _grpo_update(
    model: Any,
    encoder: PolicyFeatureEncoder,
    optimizer: Any,
    samples: list[TrainSample],
    device: str,
    update_epochs: int,
    batch_size: int,
    clip_range: float,
    entropy_coef: float,
) -> dict[str, float]:
    if not samples:
        return {"policy_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
    import torch

    model.train()
    metrics: list[dict[str, float]] = []
    for _ in range(update_epochs):
        random.shuffle(samples)
        for chunk in _chunks(samples, batch_size):
            batch = encoder.to_batch([sample.encoded for sample in chunk], device=device)
            actions = torch.tensor([sample.action_id for sample in chunk], dtype=torch.long, device=device)
            old_logprobs = torch.tensor([sample.old_logprob for sample in chunk], dtype=torch.float32, device=device)
            advantages = torch.tensor([sample.advantage for sample in chunk], dtype=torch.float32, device=device)
            logits, _values = model(batch)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            loss = policy_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            approx_kl = torch.mean(old_logprobs - new_logprobs)
            metrics.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "clip_fraction": float(clip_fraction.detach().cpu().item()),
                    "approx_kl": float(approx_kl.detach().cpu().item()),
                }
            )
    return _mean_metrics(metrics)


def _training_config(config_path: Path | None, algorithm: str) -> dict[str, Any]:
    config = {
        "tasks_dir": "data/tasks",
        "repos_dir": "data/repos",
        "output_dir": f"runs/training/{algorithm}",
        "checkpoint": f"runs/checkpoints/{algorithm}.json",
        "limit": 0,
        "epochs": 3,
        "learning_rate": 0.0003,
        "clip_range": 0.2,
        "group_size": 4,
        "resume_from": None,
        "test_timeout_sec": 10,
        "device": None,
    }
    config.update(load_config(config_path))
    if not config.get("repos_dir"):
        config["repos_dir"] = str(Path(config["tasks_dir"]).parent / "repos")
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    return config


def _policy_config_from_config(config: dict[str, Any]) -> PolicyConfig:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}
    for key in ["vocab_size", "task_text_len", "observation_text_len", "max_steps", "d_model", "num_layers", "num_heads", "ffn_dim", "dropout"]:
        if key in config and key not in model_config:
            model_config[key] = config[key]
    return PolicyConfig.from_dict(model_config)


def _training_device(config: dict[str, Any]) -> str:
    requested = config.get("device")
    if not requested:
        return default_device()
    if str(requested) == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requested device=cuda, but torch.cuda.is_available() is False. "
                "Fix the NVIDIA driver/CUDA runtime or set device: cpu explicitly for CPU debugging."
            )
    return str(requested)


def _task_paths(tasks_dir: Path, limit: int) -> list[Path]:
    paths = sorted(tasks_dir.glob("task_*.json"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No task_*.json files found in {tasks_dir}")
    return paths


def _expert_action_counts(task_paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task_path in task_paths:
        task = load_task(task_path)
        for action in ["list_files", "search_code", "read_file", "apply_patch", "run_tests", "finish"]:
            counts[action] += 1
        if not task.metadata.get("function_name"):
            counts["search_code"] -= 1
    return counts


def _scores_from_counts(counts: Counter[str]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {action: counts.get(action, 0) / total for action in ACTIONS}


def _merge_resume_scores(scores: dict[str, float], config: dict[str, Any]) -> dict[str, float]:
    resume_from = config.get("resume_from")
    if not resume_from:
        return scores
    path = Path(str(resume_from))
    if not path.exists():
        return scores
    try:
        from .schemas import read_json

        data = read_json(path)
        old_scores = data.get("action_scores", {})
        return {
            action: (float(scores.get(action, 0.0)) + float(old_scores.get(action, 0.0))) / 2.0
            for action in ACTIONS
        }
    except Exception:
        return scores


def _write_replay_buffer(path: Path, task_paths: list[Path]) -> None:
    trajectories = []
    sequence = ["list_files", "search_code", "read_file", "apply_patch", "run_tests", "finish"]
    for task_path in task_paths:
        task = load_task(task_path)
        trajectories.append(
            {
                "task_id": task.id,
                "actions": sequence,
                "final_reward": 1.4,
                "source": "scripted_expert",
            }
        )
    write_json(path, {"trajectories": trajectories})


def _try_torch_import() -> str:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return f"torch unavailable: {type(exc).__name__}: {exc}"
    return "torch available"


def _train_ppo_fallback(config: dict[str, Any], task_paths: list[Path], checkpoint: Path) -> Path:
    counts = _expert_action_counts(task_paths)
    scores = _scores_from_counts(counts)
    scores = _merge_resume_scores(scores, config)
    for action in ["run_tests", "inspect_failure", "finish"]:
        scores[action] += 0.1
    metadata = _checkpoint_metadata(
        algorithm="ppo",
        tasks=len(task_paths),
        torch_status=_try_torch_import(),
        extra={"objective": "fallback action-score checkpoint; torch unavailable"},
    )
    save_policy_checkpoint(checkpoint, scores, metadata)
    _write_replay_buffer(Path(config["output_dir"]) / "replay_buffer.json", task_paths)
    write_json(Path(config["output_dir"]) / "ppo_metrics.json", {"checkpoint": str(checkpoint), "action_scores": scores})
    return checkpoint


def _train_grpo_fallback(config: dict[str, Any], task_paths: list[Path], checkpoint: Path) -> Path:
    group_size = int(config.get("group_size", 4))
    counts = _expert_action_counts(task_paths)
    scores = _scores_from_counts(counts)
    scores = _merge_resume_scores(scores, config)
    scripted_reward = 1.4
    partial_rewards = [0.35, 0.15, -0.2][: max(group_size - 1, 0)]
    group = [scripted_reward, *partial_rewards]
    group_mean = sum(group) / len(group)
    advantage = scripted_reward - group_mean
    for action in ["apply_patch", "run_tests", "finish"]:
        scores[action] += advantage / 10.0
    metadata = _checkpoint_metadata(
        algorithm="grpo",
        tasks=len(task_paths),
        torch_status=_try_torch_import(),
        extra={
            "objective": "fallback group-relative action-score checkpoint; torch unavailable",
            "group_size": group_size,
            "group_advantage": advantage,
        },
    )
    save_policy_checkpoint(checkpoint, scores, metadata)
    _write_replay_buffer(Path(config["output_dir"]) / "replay_buffer.json", task_paths)
    write_json(Path(config["output_dir"]) / "grpo_metrics.json", {"checkpoint": str(checkpoint), "action_scores": scores})
    return checkpoint


def _checkpoint_metadata(
    algorithm: str,
    tasks: int,
    torch_status: str,
    torch_checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "algorithm": algorithm,
        "training_target": "tool_policy",
        "patch_generation": "expert_patch_provider",
        "scripted_patch": True,
        "torch_status": torch_status,
        "torch_checkpoint": torch_checkpoint,
        "tasks": tasks,
    }
    metadata.update(extra or {})
    return metadata


def _scripted_trajectories(task_paths: list[Path], repos_dir: Path, runs_dir: Path) -> list[tuple[Path, Trajectory]]:
    trajectories: list[tuple[Path, Trajectory]] = []
    for index, task_path in enumerate(task_paths, start=1):
        trajectory = run_episode(
            task_path=task_path,
            repos_dir=repos_dir,
            runs_dir=runs_dir,
            agent=ScriptedAgent(),
            run_id=f"sft-{index:04d}-{task_path.stem}",
        )
        trajectories.append((task_path, trajectory))
    return trajectories


def _load_or_init_policy(config: dict[str, Any], device: str) -> tuple[Any, PolicyFeatureEncoder, str]:
    resume_from = config.get("resume_from")
    if resume_from:
        torch_path = _resolve_torch_checkpoint(Path(str(resume_from)))
        if torch_path and torch_path.exists():
            model, encoder, _payload = load_torch_policy_checkpoint(torch_path, device)
            model.to(device)
            return model, encoder, str(torch_path)
    encoder = PolicyFeatureEncoder(config=_policy_config_from_config(config))
    model = TrajectoryTransformerPolicy(
        encoder.config,
        global_feature_dim=encoder.global_feature_dim,
        step_feature_dim=encoder.step_feature_dim,
        num_actions=len(ACTIONS),
    ).to(device)
    return model, encoder, "fresh"


def _resolve_torch_checkpoint(path: Path) -> Path | None:
    if path.suffix == ".pt":
        return path
    if path.exists():
        try:
            from .schemas import read_json

            data = read_json(path)
            torch_checkpoint = data.get("torch_checkpoint") or data.get("metadata", {}).get("torch_checkpoint")
            if torch_checkpoint:
                candidate = Path(str(torch_checkpoint))
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    candidate = path.with_suffix(".pt")
    return candidate if candidate.exists() else None


def _sample_task_paths(task_paths: list[Path], count: int) -> list[Path]:
    if count >= len(task_paths):
        return list(task_paths)
    return random.sample(task_paths, count)


def _trajectory_step_rewards(trajectory: Trajectory) -> list[float]:
    rewards = [float(step.reward_delta) for step in trajectory.steps]
    if rewards:
        terminal_bonus = trajectory.final_reward - sum(rewards)
        rewards[-1] += terminal_bonus
    return rewards


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def _normalize_advantages(samples: list[TrainSample]) -> None:
    if not samples:
        return
    values = [sample.advantage for sample in samples]
    avg = mean(values)
    variance = mean([(value - avg) ** 2 for value in values])
    std = math.sqrt(variance)
    for sample in samples:
        sample.advantage = (sample.advantage - avg) / (std + 1e-6)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(size, 1)
    return [items[index : index + size] for index in range(0, len(items), size)]


def _trajectory_artifact(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "task_id": trajectory.task_id,
        "agent": trajectory.agent,
        "actions": [step.action for step in trajectory.steps],
        "old_logprobs": [step.policy_logprob for step in trajectory.steps],
        "reward_deltas": [step.reward_delta for step in trajectory.steps],
        "final_reward": trajectory.final_reward,
        "hidden_passed": trajectory.hidden_passed,
        "public_passed": trajectory.public_passed,
        "tool_calls": trajectory.metrics.get("tool_calls", len(trajectory.steps)),
        "invalid_tool_calls": trajectory.metrics.get("invalid_tool_calls", 0),
        "syntax_or_import_errors": trajectory.metrics.get("syntax_or_import_errors", 0),
    }


def _artifact_summary(artifacts: list[dict[str, Any]]) -> dict[str, float]:
    if not artifacts:
        return {
            "mean_episode_reward": 0.0,
            "hidden_pass_rate": 0.0,
            "public_pass_rate": 0.0,
            "avg_tool_calls": 0.0,
            "invalid_tool_rate": 0.0,
            "syntax_error_rate": 0.0,
        }
    return {
        "mean_episode_reward": mean(float(item.get("final_reward", 0.0)) for item in artifacts),
        "hidden_pass_rate": mean(1.0 if item.get("hidden_passed") else 0.0 for item in artifacts),
        "public_pass_rate": mean(1.0 if item.get("public_passed") else 0.0 for item in artifacts),
        "avg_tool_calls": mean(float(item.get("tool_calls", 0.0)) for item in artifacts),
        "invalid_tool_rate": mean(1.0 if item.get("invalid_tool_calls", 0) else 0.0 for item in artifacts),
        "syntax_error_rate": mean(1.0 if item.get("syntax_or_import_errors", 0) else 0.0 for item in artifacts),
    }


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    return {key: mean(float(item.get(key, 0.0)) for item in metrics) for key in keys}


def _scores_from_rollout_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for artifact in artifacts:
        for action in artifact.get("actions", []):
            counts[str(action)] += 1
    if not counts:
        return {}
    return _scores_from_counts(counts)
