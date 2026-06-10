from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean
import shutil

from .agents import create_agent
from .config import load_config
from .runner import run_episode
from .schemas import Trajectory, write_json


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
    return {
        "pass_at_1": mean(successes),
        "hidden_pass_rate": mean(successes),
        "public_pass_rate": mean(public),
        "avg_tool_calls": mean(tool_calls),
        "avg_steps": mean(tool_calls),
        "invalid_patch_rate": mean(invalid_patch),
        "syntax_error_rate": mean(syntax_errors),
        "avg_api_cost_usd": mean(costs),
        "avg_duration_sec": mean(durations),
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
