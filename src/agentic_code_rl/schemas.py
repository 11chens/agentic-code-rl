from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
import json


ACTIONS = [
    "list_files",
    "read_file",
    "search_code",
    "apply_patch",
    "run_tests",
    "inspect_failure",
    "finish",
]


@dataclass(slots=True)
class TaskSpec:
    id: str
    repo_template: str
    prompt: str
    public_tests: list[str]
    hidden_tests: list[str]
    max_steps: int = 12
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestRunResult:
    scope: str
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    failure_count: int
    passed_count: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        return "\n".join(part for part in [self.stdout, self.stderr] if part)


@dataclass(slots=True)
class ToolResult:
    action: str
    ok: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    invalid: bool = False


@dataclass(slots=True)
class AgentDecision:
    action: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    policy_logprob: float | None = None


@dataclass(slots=True)
class TrajectoryStep:
    observation: str
    action: str
    tool_input: dict[str, Any]
    tool_output: str
    reward_delta: float
    policy_logprob: float | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    task_id: str
    agent: str
    steps: list[TrajectoryStep]
    final_reward: float
    success: bool
    public_passed: bool
    hidden_passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain(inner) for key, inner in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(inner) for key, inner in value.items()}
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_task(path: Path) -> TaskSpec:
    data = read_json(path)
    return TaskSpec(
        id=data["id"],
        repo_template=data["repo_template"],
        prompt=data["prompt"],
        public_tests=list(data["public_tests"]),
        hidden_tests=list(data["hidden_tests"]),
        max_steps=int(data.get("max_steps", 12)),
        tags=list(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
    )


def save_task(path: Path, task: TaskSpec) -> None:
    write_json(path, task)
