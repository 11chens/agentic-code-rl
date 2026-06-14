from __future__ import annotations

from pathlib import Path

import pytest

from agentic_code_rl.agents import AgentDecision, LearnedPolicyAgent, Memory, ScriptedAgent
from agentic_code_rl.benchmark import create_benchmark
from agentic_code_rl.environment import EpisodeWorkspace, WorkspaceError
from agentic_code_rl.evaluation import evaluate
from agentic_code_rl.policy import (
    ACTIONS,
    PolicyConfig,
    PolicyFeatureEncoder,
    TrajectoryTransformerPolicy,
    save_torch_policy_checkpoint,
    torch_available,
)
from agentic_code_rl.reporting import write_report
from agentic_code_rl.runner import run_episode
from agentic_code_rl.schemas import load_task, read_json
from agentic_code_rl.tools import ToolContext, ToolLayer
from agentic_code_rl.training import train_ppo, train_sft


def test_benchmark_create_writes_tasks_and_repos(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=3)

    assert len(task_paths) == 3
    assert (tmp_path / "tasks" / "manifest.json").exists()
    assert (tmp_path / "repos" / "task_001" / "src" / "buggy_lib.py").exists()
    assert (tmp_path / "repos" / "task_001" / "tests" / "test_public.py").exists()
    assert not (tmp_path / "repos" / "task_001" / "tests" / "test_hidden.py").exists()
    assert (tmp_path / "hidden_tests" / "task_001" / "tests" / "test_hidden.py").exists()
    assert (tmp_path / "expert_patches" / "task_001" / "patch.json").exists()
    task = load_task(task_paths[0])
    assert task.public_tests == ["tests/test_public.py"]
    assert task.hidden_tests == ["tests/test_hidden.py"]
    assert task.metadata["target_file"] == "src/buggy_lib.py"
    assert "expert_patch" not in task.metadata


def test_workspace_path_guard_and_public_hidden_boundary(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)
    task = load_task(task_paths[0])
    workspace = EpisodeWorkspace.create(task, repos_dir=tmp_path / "repos", runs_dir=tmp_path / "run")

    files = workspace.list_files()
    assert "tests/test_public.py" in files
    assert "tests/test_hidden.py" not in files
    assert workspace.run_tests(task.public_tests, scope="public").passed
    assert not workspace.run_tests(task.hidden_tests, scope="hidden").passed
    try:
        workspace.read_text("../outside.py")
    except WorkspaceError as exc:
        assert "escapes workspace" in str(exc)
    else:
        raise AssertionError("Path traversal should be rejected")

    tools = ToolLayer(ToolContext(workspace), allow_hidden_tests=False)
    result = tools.call("run_tests", {"scope": "hidden"})
    assert result.invalid
    assert "Hidden" in result.output
    all_result = tools.call("run_tests", {"scope": "all"})
    assert all_result.invalid

    hidden_read = tools.call("read_file", {"path": "tests/test_hidden.py"})
    assert hidden_read.invalid
    search = tools.call("search_code", {"query": "is_prime"})
    assert "test_hidden.py" not in search.output


def test_scripted_episode_repairs_task_and_logs_trajectory(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)

    trajectory = run_episode(
        task_path=task_paths[0],
        repos_dir=tmp_path / "repos",
        runs_dir=tmp_path / "runs",
        agent=ScriptedAgent(),
        run_id="scripted-task",
    )

    assert trajectory.success
    assert trajectory.hidden_passed
    assert trajectory.public_passed
    assert trajectory.final_reward > 1.0
    assert (tmp_path / "runs" / "scripted-task" / "trajectory.json").exists()


class BadPatchAgent:
    name = "bad_patch"

    def __init__(self) -> None:
        self.index = 0

    def decide(self, memory):  # noqa: ANN001
        sequence = [
            AgentDecision("read_file", {"path": "src/buggy_lib.py"}),
            AgentDecision("apply_patch", {"path": "src/buggy_lib.py", "content": "def broken(:\n"}),
            AgentDecision("run_tests", {"scope": "public"}),
            AgentDecision("finish"),
        ]
        decision = sequence[min(self.index, len(sequence) - 1)]
        self.index += 1
        return decision


def test_syntax_error_patch_is_penalized(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)

    trajectory = run_episode(
        task_path=task_paths[0],
        repos_dir=tmp_path / "repos",
        runs_dir=tmp_path / "runs",
        agent=BadPatchAgent(),
        run_id="bad-patch-task",
    )

    assert not trajectory.success
    assert trajectory.metrics["syntax_or_import_errors"] >= 1
    assert trajectory.final_reward < 0.5


def test_training_writes_checkpoint_replay_and_resume(tmp_path: Path) -> None:
    create_benchmark(tmp_path / "tasks", count=2)
    sft_config = tmp_path / "sft.yaml"
    sft_config.write_text(
        f"""
tasks_dir: {tmp_path / "tasks"}
output_dir: {tmp_path / "training" / "sft"}
checkpoint: {tmp_path / "checkpoints" / "sft.json"}
limit: 2
epochs: 1
""".strip(),
        encoding="utf-8",
    )
    ppo_config = tmp_path / "ppo.yaml"
    ppo_config.write_text(
        f"""
tasks_dir: {tmp_path / "tasks"}
output_dir: {tmp_path / "training" / "ppo"}
checkpoint: {tmp_path / "checkpoints" / "ppo.json"}
resume_from: {tmp_path / "checkpoints" / "sft.json"}
limit: 2
epochs: 1
""".strip(),
        encoding="utf-8",
    )

    sft_checkpoint = train_sft(sft_config)
    ppo_checkpoint = train_ppo(ppo_config)

    assert sft_checkpoint.exists()
    assert ppo_checkpoint.exists()
    assert (tmp_path / "training" / "sft" / "replay_buffer.json").exists()
    ppo_payload = read_json(ppo_checkpoint)
    assert ppo_payload["metadata"]["algorithm"] == "ppo"
    assert ppo_payload["training_target"] == "tool_policy"
    assert ppo_payload["patch_generation"] == "expert_patch_provider"


def test_eval_and_report(tmp_path: Path) -> None:
    create_benchmark(tmp_path / "tasks", count=2)
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"""
tasks_dir: {tmp_path / "tasks"}
repos_dir: {tmp_path / "repos"}
runs_dir: {tmp_path / "runs"}
limit: 2
test_timeout_sec: 30
""".strip(),
        encoding="utf-8",
    )

    run_dir = evaluate(config, "scripted")
    report = write_report(run_dir)
    summary = read_json(run_dir / "eval_summary.json")

    assert summary["task_count"] == 2
    assert summary["pass_at_1"] == 1.0
    assert report.exists()
    assert "Evaluation Report" in report.read_text(encoding="utf-8")


def test_policy_encoder_shapes_and_action_mask(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)
    task = load_task(task_paths[0])
    encoder = PolicyFeatureEncoder(PolicyConfig(max_steps=12, d_model=64, num_layers=1, num_heads=4, ffn_dim=128))

    encoded = encoder.encode(task, [])

    assert len(encoded.task_tokens) == 128
    assert len(encoded.observation_tokens) == 256
    assert len(encoded.history_actions) == 12
    assert len(encoded.history_statuses) == 12
    assert len(encoded.history_numeric_features) == 12
    assert len(encoded.action_mask) == len(ACTIONS)
    assert not encoded.action_mask[ACTIONS.index("inspect_failure")]
    assert not encoded.action_mask[ACTIONS.index("finish")]


@pytest.mark.skipif(not torch_available(), reason="torch is not installed")
def test_transformer_policy_forward_shapes(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)
    task = load_task(task_paths[0])
    encoder = PolicyFeatureEncoder(
        PolicyConfig(max_steps=12, d_model=64, num_layers=1, num_heads=4, ffn_dim=128, vocab_size=512)
    )
    model = TrajectoryTransformerPolicy(
        encoder.config,
        global_feature_dim=encoder.global_feature_dim,
        step_feature_dim=encoder.step_feature_dim,
        num_actions=len(ACTIONS),
    )
    batch = encoder.to_batch([encoder.encode(task, [])], device="cpu")

    logits, value = model(batch)

    assert tuple(logits.shape) == (1, len(ACTIONS))
    assert tuple(value.shape) == (1,)


@pytest.mark.skipif(not torch_available(), reason="torch is not installed")
def test_learned_policy_agent_loads_torch_checkpoint(tmp_path: Path) -> None:
    task_paths = create_benchmark(tmp_path / "tasks", count=1)
    task = load_task(task_paths[0])
    encoder = PolicyFeatureEncoder(
        PolicyConfig(max_steps=12, d_model=64, num_layers=1, num_heads=4, ffn_dim=128, vocab_size=512)
    )
    model = TrajectoryTransformerPolicy(
        encoder.config,
        global_feature_dim=encoder.global_feature_dim,
        step_feature_dim=encoder.step_feature_dim,
        num_actions=len(ACTIONS),
    )
    checkpoint = tmp_path / "policy.pt"
    save_torch_policy_checkpoint(
        checkpoint,
        model,
        encoder,
        metadata={"algorithm": "unit", "training_target": "tool_policy", "scripted_patch": True},
    )
    agent = LearnedPolicyAgent(checkpoint, deterministic=True, epsilon=0.0, device="cpu")

    decision = agent.decide(Memory(task))

    assert decision.action in ACTIONS
    assert decision.policy_logprob is not None
    assert "policy_value" in decision.metadata
