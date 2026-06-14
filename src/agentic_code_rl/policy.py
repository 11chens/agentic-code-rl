from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import random
import re

from .schemas import ACTIONS, TaskSpec, TrajectoryStep

try:  # Keep the package importable without the train extra.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - exercised in no-torch environments.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


ACTION_TO_ID = {action: index for index, action in enumerate(ACTIONS)}
ID_TO_ACTION = {index: action for action, index in ACTION_TO_ID.items()}
PAD_ACTION_ID = len(ACTIONS)

STATUS_TO_ID = {
    "pad": 0,
    "ok": 1,
    "failed": 2,
    "invalid": 3,
    "test_passed": 4,
    "test_failed": 5,
}


@dataclass(slots=True)
class SimpleTextTokenizer:
    vocab_size: int = 8192
    pad_id: int = 0
    unk_id: int = 1

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[^\s]", text.lower())
        ids = [self._token_id(token) for token in tokens[:max_length]]
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def config(self) -> dict[str, int]:
        return {"vocab_size": self.vocab_size, "pad_id": self.pad_id, "unk_id": self.unk_id}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SimpleTextTokenizer":
        config = config or {}
        return cls(
            vocab_size=int(config.get("vocab_size", 8192)),
            pad_id=int(config.get("pad_id", 0)),
            unk_id=int(config.get("unk_id", 1)),
        )

    def _token_id(self, token: str) -> int:
        if self.vocab_size <= 2:
            return self.unk_id
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, "little")
        return 2 + value % (self.vocab_size - 2)


@dataclass(slots=True)
class PolicyConfig:
    vocab_size: int = 8192
    task_text_len: int = 128
    observation_text_len: int = 256
    max_steps: int = 16
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PolicyConfig":
        data = data or {}
        return cls(
            vocab_size=int(data.get("vocab_size", 8192)),
            task_text_len=int(data.get("task_text_len", 128)),
            observation_text_len=int(data.get("observation_text_len", 256)),
            max_steps=int(data.get("max_steps", 16)),
            d_model=int(data.get("d_model", 512)),
            num_layers=int(data.get("num_layers", 6)),
            num_heads=int(data.get("num_heads", 8)),
            ffn_dim=int(data.get("ffn_dim", 2048)),
            dropout=float(data.get("dropout", 0.1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "task_text_len": self.task_text_len,
            "observation_text_len": self.observation_text_len,
            "max_steps": self.max_steps,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
        }


@dataclass(slots=True)
class EncodedPolicyInput:
    task_tokens: list[int]
    observation_tokens: list[int]
    global_features: list[float]
    history_actions: list[int]
    history_statuses: list[int]
    history_positions: list[int]
    history_numeric_features: list[list[float]]
    history_padding_mask: list[bool]
    action_mask: list[bool]


@dataclass(slots=True)
class PolicyFeatureEncoder:
    config: PolicyConfig = field(default_factory=PolicyConfig)
    tokenizer: SimpleTextTokenizer | None = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = SimpleTextTokenizer(vocab_size=self.config.vocab_size)

    @property
    def global_feature_dim(self) -> int:
        return len(self.feature_schema()["global_features"])

    @property
    def step_feature_dim(self) -> int:
        return len(self.feature_schema()["step_features"])

    def feature_schema(self) -> dict[str, Any]:
        return {
            "actions": ACTIONS,
            "statuses": STATUS_TO_ID,
            "global_features": [
                "step_index_norm",
                "remaining_steps_norm",
                *[f"{action}_count_norm" for action in ACTIONS],
                "has_listed_files",
                "has_searched_code",
                "has_read_target",
                "has_applied_patch",
                "has_run_public_tests",
                "has_inspected_failure",
                "last_tool_ok",
                "last_tool_invalid",
                "last_public_test_passed",
                "public_failure_improved",
                "has_function_name",
                "has_target_file",
                "last_public_failure_count_norm",
                "patches_applied_norm",
                "invalid_tool_calls_norm",
                "syntax_or_import_errors_norm",
            ],
            "step_features": [
                "reward_delta",
                "tool_ok",
                "tool_invalid",
                "is_public_test",
                "public_test_passed",
                "failure_count_norm",
                "passed_count_norm",
                "patch_count_seen_norm",
            ],
        }

    def encode(
        self,
        task: TaskSpec,
        steps: list[TrajectoryStep],
        observation: str | None = None,
    ) -> EncodedPolicyInput:
        task_text = _task_text(task)
        observation_text = observation if observation is not None else _observation_from_steps(task, steps)
        max_steps = self.config.max_steps
        clipped_steps = steps[-max_steps:]
        counts = {action: 0 for action in ACTIONS}
        for step in steps:
            if step.action in counts:
                counts[step.action] += 1

        last_step = steps[-1] if steps else None
        previous_public_failures = [
            int(step.metadata.get("failure_count", 0))
            for step in steps
            if step.action == "run_tests" and step.metadata.get("scope") == "public"
        ]
        last_public_failure_count = previous_public_failures[-1] if previous_public_failures else 0
        public_failure_improved = 0.0
        if len(previous_public_failures) >= 2 and previous_public_failures[-1] < previous_public_failures[-2]:
            public_failure_improved = 1.0

        max_task_steps = max(int(task.max_steps or max_steps), 1)
        invalid_count = sum(1 for step in steps if step.metadata.get("ok") is False or step.metadata.get("invalid"))
        syntax_count = sum(
            1
            for step in steps
            if step.action == "run_tests" and int(step.metadata.get("returncode", 0) or 0) in {2, 4}
        )
        patches_applied = counts.get("apply_patch", 0)
        global_features = [
            min(len(steps), max_task_steps) / max_task_steps,
            max(max_task_steps - len(steps), 0) / max_task_steps,
            *[counts[action] / max_task_steps for action in ACTIONS],
            1.0 if counts["list_files"] else 0.0,
            1.0 if counts["search_code"] else 0.0,
            1.0 if counts["read_file"] else 0.0,
            1.0 if counts["apply_patch"] else 0.0,
            1.0 if counts["run_tests"] else 0.0,
            1.0 if counts["inspect_failure"] else 0.0,
            1.0 if last_step and last_step.metadata.get("ok") is True else 0.0,
            1.0 if last_step and (last_step.metadata.get("ok") is False or last_step.metadata.get("invalid")) else 0.0,
            1.0 if _last_public_test_passed(steps) else 0.0,
            public_failure_improved,
            1.0 if task.metadata.get("function_name") else 0.0,
            1.0 if task.metadata.get("target_file") else 0.0,
            min(last_public_failure_count, 10) / 10.0,
            patches_applied / max_task_steps,
            invalid_count / max_task_steps,
            syntax_count / max_task_steps,
        ]

        history_actions = [PAD_ACTION_ID] * max_steps
        history_statuses = [STATUS_TO_ID["pad"]] * max_steps
        history_positions = list(range(max_steps))
        history_numeric_features = [[0.0] * self.step_feature_dim for _ in range(max_steps)]
        history_padding_mask = [True] * max_steps
        patch_count_seen = 0
        start = max_steps - len(clipped_steps)
        for offset, step in enumerate(clipped_steps):
            index = start + offset
            if step.action == "apply_patch":
                patch_count_seen += 1
            history_actions[index] = ACTION_TO_ID.get(step.action, PAD_ACTION_ID)
            history_statuses[index] = _status_id(step)
            history_padding_mask[index] = False
            failure_count = min(int(step.metadata.get("failure_count", 0) or 0), 10) / 10.0
            passed_count = min(int(step.metadata.get("passed_count", 0) or 0), 10) / 10.0
            is_public_test = step.action == "run_tests" and step.metadata.get("scope") == "public"
            history_numeric_features[index] = [
                float(step.reward_delta),
                1.0 if step.metadata.get("ok") is True else 0.0,
                1.0 if step.metadata.get("ok") is False or step.metadata.get("invalid") else 0.0,
                1.0 if is_public_test else 0.0,
                1.0 if is_public_test and step.metadata.get("passed") else 0.0,
                failure_count,
                passed_count,
                patch_count_seen / max_task_steps,
            ]

        return EncodedPolicyInput(
            task_tokens=self.tokenizer.encode(task_text, self.config.task_text_len),
            observation_tokens=self.tokenizer.encode(observation_text, self.config.observation_text_len),
            global_features=global_features,
            history_actions=history_actions,
            history_statuses=history_statuses,
            history_positions=history_positions,
            history_numeric_features=history_numeric_features,
            history_padding_mask=history_padding_mask,
            action_mask=action_mask_for_steps(task, steps),
        )

    def to_batch(self, encoded: list[EncodedPolicyInput], device: str | None = None) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor batches")
        target_device = torch.device(device or "cpu")
        return {
            "task_tokens": torch.tensor([item.task_tokens for item in encoded], dtype=torch.long, device=target_device),
            "observation_tokens": torch.tensor(
                [item.observation_tokens for item in encoded], dtype=torch.long, device=target_device
            ),
            "global_features": torch.tensor(
                [item.global_features for item in encoded], dtype=torch.float32, device=target_device
            ),
            "history_actions": torch.tensor(
                [item.history_actions for item in encoded], dtype=torch.long, device=target_device
            ),
            "history_statuses": torch.tensor(
                [item.history_statuses for item in encoded], dtype=torch.long, device=target_device
            ),
            "history_positions": torch.tensor(
                [item.history_positions for item in encoded], dtype=torch.long, device=target_device
            ),
            "history_numeric_features": torch.tensor(
                [item.history_numeric_features for item in encoded], dtype=torch.float32, device=target_device
            ),
            "history_padding_mask": torch.tensor(
                [item.history_padding_mask for item in encoded], dtype=torch.bool, device=target_device
            ),
            "action_mask": torch.tensor([item.action_mask for item in encoded], dtype=torch.bool, device=target_device),
        }


if nn is not None:

    class TrajectoryTransformerPolicy(nn.Module):
        def __init__(
            self,
            config: PolicyConfig,
            global_feature_dim: int,
            step_feature_dim: int,
            num_actions: int = len(ACTIONS),
        ) -> None:
            super().__init__()
            self.config = config
            self.global_feature_dim = global_feature_dim
            self.step_feature_dim = step_feature_dim
            self.num_actions = num_actions
            self.text_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
            self.action_embedding = nn.Embedding(num_actions + 1, config.d_model)
            self.status_embedding = nn.Embedding(len(STATUS_TO_ID), config.d_model)
            self.position_embedding = nn.Embedding(config.max_steps + 8, config.d_model)
            self.global_proj = nn.Linear(global_feature_dim, config.d_model)
            self.step_numeric_proj = nn.Linear(step_feature_dim, config.d_model)
            self.decision_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.policy_head = nn.Linear(config.d_model, num_actions)
            self.value_head = nn.Linear(config.d_model, 1)
            nn.init.normal_(self.decision_token, mean=0.0, std=0.02)

        def forward(self, batch: dict[str, Any]) -> tuple[Any, Any]:
            task_token = self._pool_text(batch["task_tokens"])
            observation_token = self._pool_text(batch["observation_tokens"])
            global_token = self.global_proj(batch["global_features"]).unsqueeze(1)
            step_tokens = (
                self.action_embedding(batch["history_actions"])
                + self.status_embedding(batch["history_statuses"])
                + self.position_embedding(batch["history_positions"])
                + self.step_numeric_proj(batch["history_numeric_features"])
            )
            batch_size = batch["task_tokens"].shape[0]
            decision = self.decision_token.expand(batch_size, -1, -1)
            sequence = torch.cat([task_token, observation_token, global_token, step_tokens, decision], dim=1)
            prefix_mask = torch.zeros((batch_size, 3), dtype=torch.bool, device=sequence.device)
            suffix_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=sequence.device)
            padding_mask = torch.cat([prefix_mask, batch["history_padding_mask"], suffix_mask], dim=1)
            hidden = self.transformer(sequence, src_key_padding_mask=padding_mask)
            decision_hidden = hidden[:, -1, :]
            logits = self.policy_head(decision_hidden)
            action_mask = batch["action_mask"]
            masked_logits = logits.masked_fill(~action_mask, -1e9)
            value = self.value_head(decision_hidden).squeeze(-1)
            return masked_logits, value

        def _pool_text(self, token_ids: Any) -> Any:
            embeddings = self.text_embedding(token_ids)
            mask = (token_ids != 0).unsqueeze(-1)
            masked = embeddings * mask
            denom = mask.sum(dim=1).clamp_min(1)
            return (masked.sum(dim=1) / denom).unsqueeze(1)

else:

    class TrajectoryTransformerPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for TrajectoryTransformerPolicy")


def torch_available() -> bool:
    return torch is not None and nn is not None


def default_device(prefer_cuda: bool = True) -> str:
    if torch is None:
        return "cpu"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_torch_policy_checkpoint(
    path: Path,
    model: Any,
    encoder: PolicyFeatureEncoder,
    optimizer: Any | None = None,
    training_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to save torch checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": encoder.config.to_dict(),
        "action_list": ACTIONS,
        "tokenizer_config": encoder.tokenizer.config() if encoder.tokenizer else {},
        "feature_schema": encoder.feature_schema(),
        "training_config": training_config or {},
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_torch_policy_checkpoint(path: Path, device: str | None = None) -> tuple[Any, PolicyFeatureEncoder, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load torch checkpoints")
    payload = torch.load(path, map_location=torch.device(device or "cpu"))
    config = PolicyConfig.from_dict(payload.get("model_config", {}))
    tokenizer = SimpleTextTokenizer.from_config(payload.get("tokenizer_config", {}))
    encoder = PolicyFeatureEncoder(config=config, tokenizer=tokenizer)
    model = TrajectoryTransformerPolicy(
        config,
        global_feature_dim=encoder.global_feature_dim,
        step_feature_dim=encoder.step_feature_dim,
        num_actions=len(ACTIONS),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device or "cpu"))
    return model, encoder, payload


def action_mask_for_steps(task: TaskSpec, steps: list[TrajectoryStep]) -> list[bool]:
    del task
    mask = {action: True for action in ACTIONS}
    if not steps:
        mask["finish"] = False
        mask["inspect_failure"] = False
    if not any(step.action == "run_tests" for step in steps):
        mask["inspect_failure"] = False
    if any(step.action == "apply_patch" for step in steps):
        mask["apply_patch"] = False
    if any(step.action == "finish" for step in steps):
        return [action == "finish" for action in ACTIONS]
    if not any(mask.values()):
        mask["finish"] = True
    return [mask[action] for action in ACTIONS]


def tool_input_for_action(task: TaskSpec, action: str) -> dict[str, Any]:
    if action == "read_file":
        return {"path": str(task.metadata.get("target_file", "src/buggy_lib.py"))}
    if action == "search_code":
        return {"query": str(task.metadata.get("function_name", "")) or "def "}
    if action == "apply_patch":
        source_case = str(task.metadata.get("source_case", ""))
        if not source_case:
            return {}
        from .benchmark import expert_patch_for_case

        return expert_patch_for_case(source_case)
    if action == "run_tests":
        return {"scope": "public"}
    return {}


def choose_action_from_logits(
    logits: Any,
    deterministic: bool,
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> tuple[int, float, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required to sample policy actions")
    squeezed = logits.squeeze(0)
    if epsilon > 0.0 and random.random() < epsilon:
        valid = torch.nonzero(squeezed > -1e8, as_tuple=False).flatten().tolist()
        action_id = int(random.choice(valid))
        dist = torch.distributions.Categorical(logits=squeezed / max(temperature, 1e-6))
    else:
        dist = torch.distributions.Categorical(logits=squeezed / max(temperature, 1e-6))
        action_id = int(torch.argmax(squeezed).item()) if deterministic else int(dist.sample().item())
    action_tensor = torch.tensor(action_id, device=squeezed.device)
    logprob = float(dist.log_prob(action_tensor).detach().cpu().item())
    entropy = float(dist.entropy().detach().cpu().item())
    return action_id, logprob, entropy


def _task_text(task: TaskSpec) -> str:
    tags = " ".join(task.tags)
    function_name = str(task.metadata.get("function_name", ""))
    target_file = str(task.metadata.get("target_file", ""))
    return f"{task.prompt}\nfunction: {function_name}\ntarget_file: {target_file}\ntags: {tags}"


def _observation_from_steps(task: TaskSpec, steps: list[TrajectoryStep]) -> str:
    if not steps:
        return f"Task: {task.prompt}"
    recent = steps[-3:]
    lines = [f"Task: {task.prompt}", "Recent tool history:"]
    for step in recent:
        output = step.tool_output.replace("\n", " ")
        lines.append(f"- {step.action}: {output[:400]}")
    return "\n".join(lines)


def _status_id(step: TrajectoryStep) -> int:
    if step.metadata.get("invalid"):
        return STATUS_TO_ID["invalid"]
    if step.action == "run_tests":
        return STATUS_TO_ID["test_passed"] if step.metadata.get("passed") else STATUS_TO_ID["test_failed"]
    if step.metadata.get("ok") is True:
        return STATUS_TO_ID["ok"]
    return STATUS_TO_ID["failed"]


def _last_public_test_passed(steps: list[TrajectoryStep]) -> bool:
    for step in reversed(steps):
        if step.action == "run_tests" and step.metadata.get("scope") == "public":
            return bool(step.metadata.get("passed"))
    return False
