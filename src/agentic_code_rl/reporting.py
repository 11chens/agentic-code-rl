from __future__ import annotations

from pathlib import Path

from .schemas import read_json


def write_report(run: Path, out: Path | None = None) -> Path:
    run = run.resolve()
    if run.name == "latest" and not run.exists():
        run = Path("runs/latest").resolve()
    if (run / "eval_summary.json").exists():
        summary = read_json(run / "eval_summary.json")
        body = _eval_report(run, summary)
    elif (run / "trajectory.json").exists():
        trajectory = read_json(run / "trajectory.json")
        body = _trajectory_report(run, trajectory)
    else:
        raise FileNotFoundError(f"No eval_summary.json or trajectory.json found in {run}")
    output = out or (run / "report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    return output


def _eval_report(run: Path, summary: dict[str, object]) -> str:
    rows = [
        ("Agent", summary.get("agent", "")),
        ("Task count", summary.get("task_count", 0)),
        ("Checkpoint", summary.get("checkpoint", "")),
        ("Training target", summary.get("training_target", "")),
        ("Patch generation", summary.get("patch_generation", "")),
        ("Scripted patch", summary.get("scripted_patch", "")),
        ("Torch checkpoint", summary.get("torch_checkpoint", "")),
        ("pass@1", f"{float(summary.get('pass_at_1', 0.0)):.3f}"),
        ("Hidden pass rate", f"{float(summary.get('hidden_pass_rate', 0.0)):.3f}"),
        ("Public pass rate", f"{float(summary.get('public_pass_rate', 0.0)):.3f}"),
        ("Avg tool calls", f"{float(summary.get('avg_tool_calls', 0.0)):.2f}"),
        ("Invalid patch rate", f"{float(summary.get('invalid_patch_rate', 0.0)):.3f}"),
        ("Syntax error rate", f"{float(summary.get('syntax_error_rate', 0.0)):.3f}"),
        ("Patch candidate accuracy", f"{float(summary.get('patch_candidate_accuracy', 0.0)):.3f}"),
        ("Oracle candidate selection rate", f"{float(summary.get('oracle_candidate_selection_rate', 0.0)):.3f}"),
        ("Avg duration sec", f"{float(summary.get('avg_duration_sec', 0.0)):.2f}"),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return f"""# Agentic Code RL Evaluation Report

Run: `{run}`

| Metric | Value |
| --- | --- |
{table}

## Interpretation

This report separates public-test progress from hidden-test success. A strong agent should raise hidden pass rate while keeping average tool calls and invalid patches low.
"""


def _trajectory_report(run: Path, trajectory: dict[str, object]) -> str:
    steps = trajectory.get("steps", [])
    lines = [
        "# Agentic Code RL Episode Report",
        "",
        f"Run: `{run}`",
        "",
        f"- Task: `{trajectory.get('task_id')}`",
        f"- Agent: `{trajectory.get('agent')}`",
        f"- Success: `{trajectory.get('success')}`",
        f"- Final reward: `{trajectory.get('final_reward')}`",
        "",
        "## Steps",
        "",
    ]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. `{step.get('action')}` reward `{step.get('reward_delta')}`")
    lines.append("")
    return "\n".join(lines)
