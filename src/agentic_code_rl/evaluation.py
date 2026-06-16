from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean
import shutil

from .agents import create_agent
from .config import load_config
from .runner import run_episode
from .schemas import Trajectory, read_json, write_json


def evaluate(config_path: Path | None, agent_name: str, checkpoint: Path | None = None) -> Path:
    config = _default_eval_config()
    config.update(load_config(config_path))
    tasks_dir = Path(config["tasks_dir"])
    repos_dir = Path(config["repos_dir"])
    runs_dir = Path(config["runs_dir"])
    run_id = str(config.get("run_id") or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{agent_name}")
    limit = int(config.get("limit", 0) or 0)
    timeout = int(config.get("test_timeout_sec", 10))

    task_paths = sorted(path for path in tasks_dir.glob("task_*.json"))
    if limit:
        task_paths = task_paths[:limit]
    eval_root = runs_dir / run_id
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    trajectories: list[Trajectory] = []
    for task_path in task_paths:
        agent = create_agent(agent_name, checkpoint=checkpoint)
        trajectory = run_episode(
            task_path=task_path,
            repos_dir=repos_dir,
            runs_dir=eval_root,
            agent=agent,
            run_id=task_path.stem,
            test_timeout_sec=timeout,
        )
        trajectories.append(trajectory)

    summary = summarize_trajectories(trajectories)
    summary["agent"] = agent_name
    summary["run_id"] = run_id
    summary["task_count"] = len(trajectories)
    summary.update(_checkpoint_summary(agent_name, checkpoint))
    write_json(eval_root / "eval_summary.json", summary)
    _update_latest(runs_dir, eval_root)
    return eval_root


def summarize_trajectories(trajectories: list[Trajectory]) -> dict[str, float]:
    if not trajectories:
        return {
            "pass_at_1": 0.0,
            "hidden_pass_rate": 0.0,
            "public_pass_rate": 0.0,
            "avg_tool_calls": 0.0,
            "avg_steps": 0.0,
            "invalid_patch_rate": 0.0,
            "syntax_error_rate": 0.0,
            "patch_candidate_accuracy": 0.0,
            "oracle_candidate_selection_rate": 0.0,
            "avg_api_cost_usd": 0.0,
            "avg_duration_sec": 0.0,
        }
    successes = [1.0 if item.success else 0.0 for item in trajectories]
    public = [1.0 if item.public_passed else 0.0 for item in trajectories]
    tool_calls = [float(item.metrics.get("tool_calls", len(item.steps))) for item in trajectories]
    invalid_patch = [
        1.0 if item.metrics.get("patches_applied", 0) == 0 or item.metrics.get("invalid_tool_calls", 0) else 0.0
        for item in trajectories
    ]
    syntax_errors = [1.0 if item.metrics.get("syntax_or_import_errors", 0) else 0.0 for item in trajectories]
    durations = [float(item.metrics.get("duration_sec", 0.0)) for item in trajectories]
    costs = [float(item.metrics.get("api_cost_usd", 0.0)) for item in trajectories]
    patch_metrics = _patch_candidate_metrics(trajectories)
    return {
        "pass_at_1": mean(successes),
        "hidden_pass_rate": mean(successes),
        "public_pass_rate": mean(public),
        "avg_tool_calls": mean(tool_calls),
        "avg_steps": mean(tool_calls),
        "invalid_patch_rate": mean(invalid_patch),
        "syntax_error_rate": mean(syntax_errors),
        "patch_candidate_accuracy": patch_metrics["patch_candidate_accuracy"],
        "oracle_candidate_selection_rate": patch_metrics["oracle_candidate_selection_rate"],
        "avg_api_cost_usd": mean(costs),
        "avg_duration_sec": mean(durations),
    }


def _patch_candidate_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
    labeled_steps = []
    oracle_steps = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            if step.action != "apply_patch":
                continue
            metadata = step.metadata
            if metadata.get("patch_candidate_label") is not None or metadata.get("patch_candidate_is_correct") is not None:
                labeled_steps.append(1.0 if metadata.get("patch_candidate_is_correct") else 0.0)
            if metadata.get("patch_candidate_id") is not None:
                oracle_steps.append(1.0 if metadata.get("patch_candidate_id") == "expert_correct" else 0.0)
    return {
        "patch_candidate_accuracy": mean(labeled_steps) if labeled_steps else 0.0,
        "oracle_candidate_selection_rate": mean(oracle_steps) if oracle_steps else 0.0,
    }


def _default_eval_config() -> dict[str, object]:
    return {
        "tasks_dir": "data/tasks",
        "repos_dir": "data/repos",
        "runs_dir": "runs",
        "limit": 0,
        "test_timeout_sec": 10,
    }


def _update_latest(runs_dir: Path, eval_root: Path) -> None:
    latest = runs_dir / "latest"
    if latest.exists():
        if latest.is_dir():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(eval_root, latest)


def _checkpoint_summary(agent_name: str, checkpoint: Path | None) -> dict[str, object]:
    if checkpoint is None and agent_name in {"sft", "ppo", "grpo", "learned"}:
        checkpoint = Path("runs") / "checkpoints" / f"{agent_name}.json"
    if checkpoint is None or not checkpoint.exists() or checkpoint.suffix != ".json":
        return {}
    try:
        data = read_json(checkpoint)
    except Exception:
        return {"checkpoint": str(checkpoint)}
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "checkpoint": str(checkpoint),
        "training_target": data.get("training_target") or metadata.get("training_target"),
        "patch_generation": data.get("patch_generation") or metadata.get("patch_generation"),
        "scripted_patch": data.get("scripted_patch", metadata.get("scripted_patch")),
        "torch_checkpoint": data.get("torch_checkpoint") or metadata.get("torch_checkpoint"),
    }
