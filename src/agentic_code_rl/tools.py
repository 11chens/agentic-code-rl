from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import difflib
import re

from .environment import EpisodeWorkspace, WorkspaceError
from .schemas import TestRunResult, ToolResult


@dataclass(slots=True)
class ToolContext:
    workspace: EpisodeWorkspace
    last_test_result: TestRunResult | None = None
    public_test_history: list[TestRunResult] = field(default_factory=list)
    patches_applied: int = 0


class ToolLayer:
    def __init__(self, context: ToolContext, test_timeout_sec: int = 30, allow_hidden_tests: bool = False):
        self.context = context
        self.test_timeout_sec = test_timeout_sec
        self.allow_hidden_tests = allow_hidden_tests

    def call(self, action: str, tool_input: dict[str, Any] | None = None) -> ToolResult:
        tool_input = tool_input or {}
        try:
            if action == "list_files":
                return self.list_files()
            if action == "read_file":
                return self.read_file(str(tool_input.get("path", "")))
            if action == "search_code":
                return self.search_code(str(tool_input.get("query", "")))
            if action == "apply_patch":
                return self.apply_patch(tool_input)
            if action == "run_tests":
                scope = str(tool_input.get("scope", "public"))
                return self.run_tests(scope)
            if action == "inspect_failure":
                return self.inspect_failure()
            if action == "finish":
                return ToolResult(action=action, ok=True, output="Agent finished episode.")
            return ToolResult(action=action, ok=False, output=f"Unknown action: {action}", invalid=True)
        except WorkspaceError as exc:
            return ToolResult(action=action, ok=False, output=str(exc), invalid=True)
        except Exception as exc:  # Defensive boundary for agent-selected tools.
            return ToolResult(action=action, ok=False, output=f"{type(exc).__name__}: {exc}", invalid=True)

    def list_files(self) -> ToolResult:
        files = self.context.workspace.list_files()
        return ToolResult("list_files", True, "\n".join(files), {"file_count": len(files)})

    def read_file(self, path: str) -> ToolResult:
        if not path:
            return ToolResult("read_file", False, "Missing path", invalid=True)
        text = self.context.workspace.read_text(path)
        return ToolResult("read_file", True, text, {"path": path})

    def search_code(self, query: str) -> ToolResult:
        if not query:
            return ToolResult("search_code", False, "Missing query", invalid=True)
        matches: list[str] = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for relative in self.context.workspace.list_files():
            if not relative.endswith((".py", ".md", ".txt")):
                continue
            text = self.context.workspace.read_text(relative, max_chars=50000)
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{relative}:{lineno}: {line}")
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        return ToolResult("search_code", True, "\n".join(matches) or "No matches", {"query": query, "matches": len(matches)})

    def apply_patch(self, payload: dict[str, Any]) -> ToolResult:
        path = str(payload.get("path", ""))
        if not path:
            return ToolResult("apply_patch", False, "Missing path", invalid=True)
        file_path = self.context.workspace.resolve_path(path)
        if not file_path.exists():
            return ToolResult("apply_patch", False, f"File not found: {path}", invalid=True)
        old = file_path.read_text(encoding="utf-8")
        if "content" in payload:
            new = str(payload["content"])
        else:
            find = str(payload.get("find", ""))
            replace = str(payload.get("replace", ""))
            if not find:
                return ToolResult("apply_patch", False, "Missing find/content patch payload", invalid=True)
            if find not in old:
                return ToolResult("apply_patch", False, "Patch find text did not match file", invalid=True)
            new = old.replace(find, replace, 1)
        file_path.write_text(new, encoding="utf-8")
        self.context.patches_applied += 1
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return ToolResult("apply_patch", True, diff or "Patch applied with no textual diff", {"path": path})

    def run_tests(self, scope: str = "public") -> ToolResult:
        if scope in {"hidden", "all"}:
            if not self.allow_hidden_tests:
                return ToolResult(
                    "run_tests",
                    False,
                    "Hidden and all-test runs are reserved for final evaluation.",
                    invalid=True,
                )
            tests = (
                self.context.workspace.task.hidden_tests
                if scope == "hidden"
                else [*self.context.workspace.task.public_tests, *self.context.workspace.task.hidden_tests]
            )
        else:
            scope = "public"
            tests = self.context.workspace.task.public_tests
        result = self.context.workspace.run_tests(tests, scope=scope, timeout_sec=self.test_timeout_sec)
        self.context.last_test_result = result
        if scope == "public":
            self.context.public_test_history.append(result)
        return ToolResult(
            "run_tests",
            result.passed,
            _summarize_test_result(result),
            {
                "scope": scope,
                "passed": result.passed,
                "returncode": result.returncode,
                "failure_count": result.failure_count,
                "passed_count": result.passed_count,
                "timed_out": result.timed_out,
            },
        )

    def inspect_failure(self) -> ToolResult:
        result = self.context.last_test_result
        if result is None:
            return ToolResult("inspect_failure", False, "No test result is available yet.", invalid=True)
        output = result.output.strip()
        lines = output.splitlines()
        excerpt = "\n".join(lines[-80:]) if lines else "No output"
        return ToolResult("inspect_failure", True, excerpt, {"scope": result.scope, "returncode": result.returncode})


def _summarize_test_result(result: TestRunResult) -> str:
    status = "PASSED" if result.passed else "FAILED"
    output = result.output.strip()
    tail = "\n".join(output.splitlines()[-30:]) if output else ""
    return (
        f"{status} {result.scope} tests in {result.duration_sec:.2f}s "
        f"(returncode={result.returncode}, failures={result.failure_count}, passed={result.passed_count})"
        + (f"\n{tail}" if tail else "")
    )
