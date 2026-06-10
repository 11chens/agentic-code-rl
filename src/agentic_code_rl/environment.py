from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import os
import shutil
import subprocess
import sys
import tempfile

from .schemas import TaskSpec, TestRunResult


class WorkspaceError(RuntimeError):
    pass


@dataclass(slots=True)
class EpisodeWorkspace:
    task: TaskSpec
    repo_source: Path
    root: Path

    @classmethod
    def create(cls, task: TaskSpec, repos_dir: Path, runs_dir: Path | None = None) -> "EpisodeWorkspace":
        repo_source = (repos_dir / task.repo_template).resolve()
        if not repo_source.exists():
            raise FileNotFoundError(f"Repo template not found: {repo_source}")
        base = runs_dir.resolve() if runs_dir else Path(tempfile.mkdtemp(prefix="agentic-code-rl-"))
        base.mkdir(parents=True, exist_ok=True)
        workspace_root = base / "workspace"
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        shutil.copytree(repo_source, workspace_root)
        return cls(task=task, repo_source=repo_source, root=workspace_root.resolve())

    def resolve_path(self, relative: str | Path) -> Path:
        raw = Path(relative)
        if raw.is_absolute():
            raise WorkspaceError(f"Absolute paths are not allowed: {relative}")
        resolved = (self.root / raw).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {relative}") from exc
        return resolved

    def list_files(self) -> list[str]:
        files: list[str] = []
        for path in self.root.rglob("*"):
            if path.is_file() and not _is_ignored(path):
                files.append(path.relative_to(self.root).as_posix())
        return sorted(files)

    def read_text(self, relative: str | Path, max_chars: int = 12000) -> str:
        path = self.resolve_path(relative)
        if not path.exists() or not path.is_file():
            raise WorkspaceError(f"File not found: {relative}")
        text = path.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + "\n...<truncated>..."
        return text

    def write_text(self, relative: str | Path, text: str) -> None:
        path = self.resolve_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_tests(self, tests: list[str], scope: str, timeout_sec: int = 10) -> TestRunResult:
        args = [sys.executable, "-m", "pytest", "-q", *tests]
        env = os.environ.copy()
        src_path = str(self.root / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        started = perf_counter()
        try:
            completed = subprocess.run(
                args,
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
            duration = perf_counter() - started
            output = completed.stdout + "\n" + completed.stderr
            failure_count = _parse_failure_count(output, completed.returncode)
            passed_count = _parse_passed_count(output)
            return TestRunResult(
                scope=scope,
                passed=completed.returncode == 0,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=duration,
                failure_count=failure_count,
                passed_count=passed_count,
            )
        except subprocess.TimeoutExpired as exc:
            duration = perf_counter() - started
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return TestRunResult(
                scope=scope,
                passed=False,
                returncode=124,
                stdout=stdout,
                stderr=stderr + f"\nTimed out after {timeout_sec}s",
                duration_sec=duration,
                failure_count=1,
                passed_count=0,
                timed_out=True,
            )


def _is_ignored(path: Path) -> bool:
    ignored_names = {"__pycache__", ".pytest_cache", ".git"}
    return any(part in ignored_names for part in path.parts)


def _parse_failure_count(output: str, returncode: int) -> int:
    if returncode == 0:
        return 0
    for token in output.replace(",", " ").split():
        if token.isdigit():
            # Pytest summary starts with a count in common failure cases.
            return int(token)
    return 1


def _parse_passed_count(output: str) -> int:
    words = output.replace(",", " ").split()
    for index, word in enumerate(words):
        if word == "passed" and index > 0 and words[index - 1].isdigit():
            return int(words[index - 1])
    return 0
