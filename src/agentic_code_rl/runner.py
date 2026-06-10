from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

from .agents import Agent, Memory
from .environment import EpisodeWorkspace
from .schemas import TaskSpec, Trajectory, TrajectoryStep, load_task, write_json
from .tools import ToolContext, ToolLayer


@dataclass(slots=True)
class RewardState:
    final_reward: float = 0.0
    last_public_failures: int | None = None
    public_passed: bool = False
    invalid_tools: int = 0
    syntax_or_import_errors: int = 0
    finish_called: bool = False


def run_episode(
    task_path: Path,
    repos_dir: Path,
    runs_dir: Path,
    agent: Agent,
    run_id: str | None = None,
    test_timeout_sec: int = 10,
) -> Trajectory:
    task = load_task(task_path)
    run_root = runs_dir / (run_id or _default_run_id(task.id, agent.name))
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    workspace = EpisodeWorkspace.create(task, repos_dir=repos_dir, runs_dir=run_root)
    context = ToolContext(workspace)
    tools = ToolLayer(context, test_timeout_sec=test_timeout_sec, allow_hidden_tests=False)
    memory = Memory(task)
    reward_state = RewardState()
    started = perf_counter()

    for _ in range(task.max_steps):
        observation = memory.observation()
        decision = agent.decide(memory)
        result = tools.call(decision.action, decision.tool_input)
        reward_delta = _reward_delta(decision.action, result.metadata, result.invalid, reward_state)
        step = TrajectoryStep(
            observation=observation,
            action=decision.action,
            tool_input=decision.tool_input,
            tool_output=result.output,
            reward_delta=reward_delta,
            policy_logprob=decision.policy_logprob,
            rationale=decision.rationale,
            metadata={"ok": result.ok, **result.metadata},
        )
        memory.steps.append(step)
        if decision.action == "finish":
            reward_state.finish_called = True
            break

    public_result = workspace.run_tests(task.public_tests, scope="public-final", timeout_sec=test_timeout_sec)
    hidden_result = workspace.run_tests(task.hidden_tests, scope="hidden-final", timeout_sec=test_timeout_sec)
    reward_state.public_passed = public_result.passed
    final_reward = reward_state.final_reward
    if public_result.passed:
        final_reward += 0.4
    if hidden_result.passed:
        final_reward += 1.0
    elif reward_state.finish_called:
        final_reward -= 0.2

    metrics: dict[str, Any] = {
        "duration_sec": perf_counter() - started,
        "tool_calls": len(memory.steps),
        "patches_applied": context.patches_applied,
        "invalid_tool_calls": reward_state.invalid_tools,
        "syntax_or_import_errors": reward_state.syntax_or_import_errors,
        "public_failure_count": public_result.failure_count,
        "hidden_failure_count": hidden_result.failure_count,
        "public_passed": public_result.passed,
        "hidden_passed": hidden_result.passed,
        "api_cost_usd": 0.0,
    }
    trajectory = Trajectory(
        task_id=task.id,
        agent=agent.name,
        steps=memory.steps,
        final_reward=round(final_reward, 4),
        success=hidden_result.passed,
        public_passed=public_result.passed,
        hidden_passed=hidden_result.passed,
        metrics=metrics,
    )
    write_json(run_root / "trajectory.json", trajectory)
    write_json(run_root / "summary.json", {"task": task, "trajectory": trajectory, "run_dir": str(run_root)})
    return trajectory


def _reward_delta(action: str, metadata: dict[str, Any], invalid: bool, state: RewardState) -> float:
    reward = -0.01
    if invalid:
        state.invalid_tools += 1
        reward -= 0.05
    if action == "run_tests":
        if metadata.get("scope") == "public":
            failures = int(metadata.get("failure_count", 1))
            if metadata.get("passed"):
                state.public_passed = True
                reward += 0.4
            elif state.last_public_failures is not None and failures < state.last_public_failures:
                reward += 0.2
            state.last_public_failures = failures
            output_hint = str(metadata.get("returncode", ""))
            if int(metadata.get("returncode", 0)) in {2, 4} or output_hint in {"2", "4"}:
                state.syntax_or_import_errors += 1
                reward -= 0.1
    state.final_reward += reward
    return round(reward, 4)


def _default_run_id(task_id: str, agent_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{agent_name}-{task_id}"
