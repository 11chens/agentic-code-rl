from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import json
import os
import random

from .schemas import ACTIONS, AgentDecision, TaskSpec, TrajectoryStep, read_json


@dataclass(slots=True)
class Memory:
    task: TaskSpec
    steps: list[TrajectoryStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def observation(self) -> str:
        if not self.steps:
            return f"Task: {self.task.prompt}"
        recent = self.steps[-3:]
        lines = [f"Task: {self.task.prompt}", "Recent tool history:"]
        for step in recent:
            output = step.tool_output.replace("\n", " ")
            lines.append(f"- {step.action}: {output[:400]}")
        return "\n".join(lines)

    def action_counts(self) -> dict[str, int]:
        counts = {action: 0 for action in ACTIONS}
        for step in self.steps:
            counts[step.action] = counts.get(step.action, 0) + 1
        return counts

    def has_action(self, action: str) -> bool:
        return any(step.action == action for step in self.steps)


class Agent(Protocol):
    name: str

    def decide(self, memory: Memory) -> AgentDecision:
        ...


class ScriptedAgent:
    name = "scripted"

    def decide(self, memory: Memory) -> AgentDecision:
        counts = memory.action_counts()
        task = memory.task
        target_file = str(task.metadata.get("target_file", "src/buggy_lib.py"))
        function_name = str(task.metadata.get("function_name", ""))
        expert_patch = _expert_patch_for_task(task)
        if counts["list_files"] == 0:
            return AgentDecision("list_files", rationale="Inspect repository layout.")
        if counts["search_code"] == 0 and function_name:
            return AgentDecision("search_code", {"query": function_name}, rationale="Find the target function.")
        if counts["read_file"] == 0:
            return AgentDecision("read_file", {"path": target_file}, rationale="Read target source file.")
        if counts["apply_patch"] == 0 and expert_patch:
            return AgentDecision("apply_patch", expert_patch, rationale="Apply known expert repair.")
        if counts["run_tests"] == 0:
            return AgentDecision("run_tests", {"scope": "public"}, rationale="Verify public tests.")
        return AgentDecision("finish", rationale="Stop after verification.")


class ReactAgent:
    name = "react"

    def __init__(self, fallback: Agent | None = None, model: str | None = None):
        self.fallback = fallback or ScriptedAgent()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def decide(self, memory: Memory) -> AgentDecision:
        if not os.getenv("OPENAI_API_KEY"):
            decision = self.fallback.decide(memory)
            decision.rationale = f"Fallback scripted path: {decision.rationale}"
            return decision
        try:
            return self._llm_decide(memory)
        except Exception as exc:
            decision = self.fallback.decide(memory)
            decision.rationale = f"LLM planner failed ({type(exc).__name__}); fallback scripted path: {decision.rationale}"
            return decision

    def _llm_decide(self, memory: Memory) -> AgentDecision:
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        prompt = {
            "task": memory.task.prompt,
            "allowed_actions": ACTIONS,
            "tool_contracts": {
                "read_file": {"path": "workspace-relative path"},
                "search_code": {"query": "text query"},
                "apply_patch": {"path": "file", "find": "old text", "replace": "new text"},
                "run_tests": {"scope": "public"},
            },
            "observation": memory.observation(),
            "target_hint": {
                "file": memory.task.metadata.get("target_file"),
                "function": memory.task.metadata.get("function_name"),
            },
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code repair planner. Return strict JSON with keys "
                        "action, tool_input, rationale. Use only public tests."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        action = str(data.get("action", ""))
        if action not in ACTIONS or action == "run_tests" and data.get("tool_input", {}).get("scope") == "hidden":
            raise ValueError(f"Invalid LLM action: {action}")
        tool_input = data.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        return AgentDecision(action, tool_input, str(data.get("rationale", "LLM selected action.")))


class LearnedPolicyAgent:
    def __init__(self, checkpoint: Path, name: str = "learned", epsilon: float = 0.05):
        self.name = name
        self.checkpoint = checkpoint
        self.epsilon = epsilon
        self.fallback = ScriptedAgent()
        data = read_json(checkpoint) if checkpoint.exists() else {}
        self.action_scores: dict[str, float] = {
            action: float(data.get("action_scores", {}).get(action, 0.0)) for action in ACTIONS
        }
        self.scripted_patch = bool(data.get("scripted_patch", True))

    def decide(self, memory: Memory) -> AgentDecision:
        if self.scripted_patch:
            scripted = self.fallback.decide(memory)
            if scripted.action == "apply_patch":
                return scripted
        valid_actions = self._valid_actions(memory)
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            action = max(valid_actions, key=lambda item: self.action_scores.get(item, 0.0))
        return self._decision_for_action(action, memory)

    def _valid_actions(self, memory: Memory) -> list[str]:
        if not memory.has_action("list_files"):
            return ["list_files"]
        actions = ["read_file", "search_code", "run_tests", "inspect_failure", "finish"]
        if not memory.has_action("apply_patch"):
            actions.append("apply_patch")
        return actions

    def _decision_for_action(self, action: str, memory: Memory) -> AgentDecision:
        task = memory.task
        target_file = str(task.metadata.get("target_file", "src/buggy_lib.py"))
        function_name = str(task.metadata.get("function_name", ""))
        if action == "read_file":
            return AgentDecision(action, {"path": target_file}, rationale="Policy selected source inspection.")
        if action == "search_code":
            return AgentDecision(action, {"query": function_name or "def "}, rationale="Policy selected code search.")
        if action == "apply_patch":
            patch = _expert_patch_for_task(task)
            return AgentDecision(action, patch, rationale="Policy selected patch action.")
        if action == "run_tests":
            return AgentDecision(action, {"scope": "public"}, rationale="Policy selected public test run.")
        return AgentDecision(action, rationale="Policy selected action.")


def create_agent(name: str, checkpoint: Path | None = None) -> Agent:
    normalized = name.lower()
    if normalized == "scripted":
        return ScriptedAgent()
    if normalized == "react":
        return ReactAgent()
    if normalized in {"sft", "ppo", "grpo", "learned"}:
        ckpt = checkpoint or Path("runs") / "checkpoints" / f"{normalized}.json"
        return LearnedPolicyAgent(ckpt, name=normalized)
    raise ValueError(f"Unknown agent: {name}")


def _expert_patch_for_task(task: TaskSpec) -> dict[str, str]:
    source_case = str(task.metadata.get("source_case", ""))
    if not source_case:
        return {}
    from .benchmark import expert_patch_for_case

    return expert_patch_for_case(source_case)


def save_policy_checkpoint(path: Path, action_scores: dict[str, float], metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "action_scores": {action: float(action_scores.get(action, 0.0)) for action in ACTIONS},
        "scripted_patch": True,
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
