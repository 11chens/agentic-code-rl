from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any
import json

from .schemas import TaskSpec, read_json, write_json


DEFAULT_CANDIDATES_DIR = Path("data") / "patch_candidates"
PATCH_CANDIDATE_FEATURES = [
    "has_find",
    "has_replace",
    "has_content",
    "payload_text_len_norm",
    "payload_key_count_norm",
]


@dataclass(frozen=True, slots=True)
class PatchCandidate:
    id: str
    source: str
    payload: dict[str, Any]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class PatchCandidateSet:
    task_id: str
    target_file: str
    oracle_candidate_id: str | None
    candidates: list[PatchCandidate]


class PatchCandidateProvider:
    def __init__(self, candidates_dir: Path | str | None = None, max_candidates: int | None = None) -> None:
        self.candidates_dir = Path(candidates_dir) if candidates_dir is not None else DEFAULT_CANDIDATES_DIR
        self.max_candidates = max_candidates

    def candidate_set(self, task: TaskSpec) -> PatchCandidateSet:
        path = self.candidates_dir / task.id / "candidates.json"
        if path.exists():
            return _candidate_set_from_data(read_json(path))
        source_case = str(task.metadata.get("source_case", ""))
        if source_case:
            data = synthetic_patch_candidates_from_source_case(task.id, source_case)
            return _candidate_set_from_data(data)
        return PatchCandidateSet(
            task_id=task.id,
            target_file=str(task.metadata.get("target_file", "src/buggy_lib.py")),
            oracle_candidate_id=None,
            candidates=[],
        )

    def candidates_for_policy(self, task: TaskSpec) -> list[PatchCandidate]:
        candidates = self.candidate_set(task).candidates
        if self.max_candidates is None:
            return candidates
        return candidates[: max(int(self.max_candidates), 0)]

    def candidate_ids(self, task: TaskSpec) -> list[str]:
        return [candidate.id for candidate in self.candidates_for_policy(task)]

    def payload(self, task: TaskSpec, candidate_id: str | None) -> dict[str, Any]:
        if candidate_id is None:
            return {}
        for candidate in self.candidates_for_policy(task):
            if candidate.id == candidate_id:
                return dict(candidate.payload)
        return {}

    def oracle_candidate_id(self, task: TaskSpec) -> str | None:
        return self.candidate_set(task).oracle_candidate_id

    def oracle_payload(self, task: TaskSpec) -> dict[str, Any]:
        return self.payload(task, self.oracle_candidate_id(task))

    def label(self, task: TaskSpec, candidate_id: str | None) -> str | None:
        if candidate_id is None:
            return None
        for candidate in self.candidate_set(task).candidates:
            if candidate.id == candidate_id:
                return candidate.label
        return None

    def candidate_index(self, task: TaskSpec, candidate_id: str | None) -> int | None:
        if candidate_id is None:
            return None
        ids = self.candidate_ids(task)
        try:
            return ids.index(candidate_id)
        except ValueError:
            return None


def write_patch_candidates(candidates_dir: Path, task_id: str, case: Any) -> Path:
    data = synthetic_patch_candidates_from_case(task_id, case)
    path = candidates_dir / task_id / "candidates.json"
    write_json(path, data)
    return path


def synthetic_patch_candidates_from_source_case(task_id: str, source_case: str) -> dict[str, Any]:
    from .benchmark import CASE_LIBRARY

    for case in CASE_LIBRARY:
        if case.slug == source_case:
            return synthetic_patch_candidates_from_case(task_id, case)
    return {
        "task_id": task_id,
        "target_file": "src/buggy_lib.py",
        "oracle_candidate_id": None,
        "candidates": [],
    }


def synthetic_patch_candidates_from_case(task_id: str, case: Any) -> dict[str, Any]:
    target_file = "src/buggy_lib.py"
    buggy_source = _clean(case.buggy_source)
    fixed_source = _clean(case.fixed_source)
    candidates = [
        {
            "id": "expert_correct",
            "source": "synthetic_expert",
            "payload": {"path": target_file, "find": buggy_source, "replace": fixed_source},
            "label": "correct",
        },
        {
            "id": "no_op",
            "source": "synthetic_baseline",
            "payload": {"path": target_file, "find": buggy_source, "replace": buggy_source},
            "label": "no_op",
        },
        {
            "id": "syntax_error",
            "source": "synthetic_negative",
            "payload": {"path": target_file, "content": "def broken(:\n"},
            "label": "syntax_error",
        },
        {
            "id": "partial_fix",
            "source": "synthetic_negative",
            "payload": {"path": target_file, "find": buggy_source, "replace": _partial_fix_source(fixed_source, buggy_source)},
            "label": "partial_fix",
        },
        {
            "id": "wrong_logic",
            "source": "synthetic_negative",
            "payload": {"path": target_file, "content": _wrong_logic_source(str(case.function_name))},
            "label": "wrong_logic",
        },
    ]
    return {
        "task_id": task_id,
        "target_file": target_file,
        "oracle_candidate_id": "expert_correct",
        "candidates": candidates,
    }


def candidate_text(candidate: PatchCandidate) -> str:
    visible = {
        "id": candidate.id,
        "source": candidate.source,
        "payload": candidate.payload,
    }
    return json.dumps(visible, sort_keys=True, ensure_ascii=False)


def candidate_feature_vector(candidate: PatchCandidate) -> list[float]:
    payload = candidate.payload
    text = candidate_text(candidate)
    return [
        1.0 if "find" in payload else 0.0,
        1.0 if "replace" in payload else 0.0,
        1.0 if "content" in payload else 0.0,
        min(len(text), 2000) / 2000.0,
        min(len(payload), 8) / 8.0,
    ]


def policy_visible_candidate(candidate: PatchCandidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "source": candidate.source,
        "payload": dict(candidate.payload),
    }


def _candidate_set_from_data(data: dict[str, Any]) -> PatchCandidateSet:
    candidates = [
        PatchCandidate(
            id=str(item.get("id", "")),
            source=str(item.get("source", "")),
            payload=dict(item.get("payload", {})),
            label=str(item["label"]) if "label" in item else None,
        )
        for item in data.get("candidates", [])
        if isinstance(item, dict)
    ]
    return PatchCandidateSet(
        task_id=str(data.get("task_id", "")),
        target_file=str(data.get("target_file", "src/buggy_lib.py")),
        oracle_candidate_id=data.get("oracle_candidate_id"),
        candidates=candidates,
    )


def _partial_fix_source(fixed_source: str, fallback_source: str) -> str:
    replacements = [
        ("return False", "return True"),
        ("return True", "return False"),
        ("return -1", "return None"),
        ("return []", "return None"),
        ("raise ValueError", "return None  # ValueError"),
    ]
    for old, new in replacements:
        if old in fixed_source:
            mutated = fixed_source.replace(old, new, 1)
            if mutated != fixed_source:
                return mutated
    return fallback_source


def _wrong_logic_source(function_name: str) -> str:
    safe_name = function_name or "buggy_function"
    return f"def {safe_name}(*args, **kwargs):\n    return None\n"


def _clean(text: str) -> str:
    return dedent(text).strip() + "\n"
