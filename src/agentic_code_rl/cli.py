from __future__ import annotations

from pathlib import Path
import argparse

from .agents import create_agent
from .benchmark import create_benchmark
from .evaluation import evaluate
from .reporting import write_report
from .runner import run_episode
from .training import train_grpo, train_ppo, train_sft


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-code-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    create = benchmark_sub.add_parser("create")
    create.add_argument("--out", type=Path, default=Path("data/tasks"))
    create.add_argument("--repos-out", type=Path, default=None)
    create.add_argument("--count", type=int, default=30)
    create.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--repos-dir", type=Path, default=Path("data/repos"))
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--agent", default="react", choices=["scripted", "react", "sft", "ppo", "grpo", "learned"])
    run.add_argument("--checkpoint", type=Path, default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--test-timeout-sec", type=int, default=10)

    train_sft_cmd = subparsers.add_parser("train-sft")
    train_sft_cmd.add_argument("--config", type=Path, default=None)

    train_ppo_cmd = subparsers.add_parser("train-ppo")
    train_ppo_cmd.add_argument("--config", type=Path, default=None)

    train_grpo_cmd = subparsers.add_parser("train-grpo")
    train_grpo_cmd.add_argument("--config", type=Path, default=None)

    eval_cmd = subparsers.add_parser("eval")
    eval_cmd.add_argument("--config", type=Path, default=None)
    eval_cmd.add_argument("--agent", default="scripted", choices=["scripted", "react", "sft", "ppo", "grpo", "learned"])
    eval_cmd.add_argument("--checkpoint", type=Path, default=None)

    report_cmd = subparsers.add_parser("report")
    report_cmd.add_argument("--run", type=Path, required=True)
    report_cmd.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "benchmark" and args.benchmark_command == "create":
        paths = create_benchmark(args.out, repos_out=args.repos_out, count=args.count, overwrite=args.overwrite)
        print(f"Created {len(paths)} tasks in {args.out}")
        return
    if args.command == "run":
        agent = create_agent(args.agent, checkpoint=args.checkpoint)
        trajectory = run_episode(
            task_path=args.task,
            repos_dir=args.repos_dir,
            runs_dir=args.runs_dir,
            agent=agent,
            run_id=args.run_id,
            test_timeout_sec=args.test_timeout_sec,
        )
        print(f"task={trajectory.task_id} agent={trajectory.agent} success={trajectory.success} reward={trajectory.final_reward}")
        return
    if args.command == "train-sft":
        print(f"Wrote checkpoint: {train_sft(args.config)}")
        return
    if args.command == "train-ppo":
        print(f"Wrote checkpoint: {train_ppo(args.config)}")
        return
    if args.command == "train-grpo":
        print(f"Wrote checkpoint: {train_grpo(args.config)}")
        return
    if args.command == "eval":
        print(f"Wrote eval run: {evaluate(args.config, args.agent, checkpoint=args.checkpoint)}")
        return
    if args.command == "report":
        print(f"Wrote report: {write_report(args.run, args.out)}")
        return
