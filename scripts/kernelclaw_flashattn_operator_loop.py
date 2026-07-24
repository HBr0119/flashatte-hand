#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_core.loop import LoopConfig, run_loop


def main() -> None:
    parser = argparse.ArgumentParser("KernelClaw-style FlashAttention operator loop")
    parser.add_argument("--kernel-src", "--baseline-src", dest="kernel_src", required=True, help="Initial/best .cu file.")
    parser.add_argument("--out-dir", required=True, help="Directory for rounds and best artifacts.")
    parser.add_argument("--rounds", type=int, default=None, help="Optional hard cap. If omitted, the agent decides within --min-rounds/--max-rounds.")
    parser.add_argument("--min-rounds", type=int, default=1, help="Minimum rounds before the stop/continue agent may stop.")
    parser.add_argument("--max-rounds", type=int, default=9, help="Maximum rounds the iteration planner may choose.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--best-score", type=float, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--notes-file", default=None, help="Optional text file appended to --notes.")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--stage",
        default="auto",
        choices=["auto", "parameter_search", "dispatch_tuning", "codegen", "repair", "profiler_diagnosis"],
        help="Curriculum stage exposed to the agent planner.",
    )
    parser.add_argument("--profiler-feedback", default=None, help="Optional profiler_feedback.json from a C500 profiler run.")
    parser.add_argument("--enable-profiler", action="store_true", help="Ask evaluator to write torch-profiler feedback.")
    parser.add_argument("--profile-limit", type=int, default=4, help="Number of worst cases to profile when --enable-profiler is set.")
    parser.add_argument("--disable-auxiliary", action="store_true", help="Disable AI-generated auxiliary stress benchmarks.")
    parser.add_argument("--auxiliary-repeat", type=int, default=3, help="Repeat count for auxiliary stress benchmark timing.")
    parser.add_argument("--auxiliary-cases", type=int, default=6, help="Maximum number of generated auxiliary stress cases.")
    parser.add_argument("--task-profile", default="legacy", choices=["legacy", "contest"], help="Evaluator task profile.")
    parser.add_argument("--experiments-root", default=str(ROOT / "experiments"))
    parser.add_argument("--memory-dir", default=str(ROOT / "experiments" / "agent_memory"))
    parser.add_argument("--docs-iterations-dir", default=str(ROOT / "docs" / "iterations"))
    args = parser.parse_args()
    # Contest profile always enables torch-profiler feedback
    if args.task_profile == "contest":
        args.enable_profiler = True

    notes = args.notes
    if args.notes_file:
        notes_path = Path(args.notes_file)
        if notes_path.exists():
            notes = (notes + "\n" + notes_path.read_text(encoding="utf-8", errors="replace")).strip()

    config = LoopConfig(
        kernel_src=Path(args.kernel_src),
        out_dir=Path(args.out_dir),
        rounds=args.rounds,
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        model=args.model,
        best_score=args.best_score,
        notes=notes,
        timeout=args.timeout,
        python_bin=args.python_bin,
        smoke=args.smoke,
        warmup=args.warmup,
        repeat=args.repeat,
        stage=args.stage,
        profiler_feedback=Path(args.profiler_feedback) if args.profiler_feedback else None,
        enable_profiler=args.enable_profiler,
        profile_limit=args.profile_limit,
        enable_auxiliary=not args.disable_auxiliary,
        auxiliary_repeat=args.auxiliary_repeat,
        auxiliary_cases=args.auxiliary_cases,
        task_profile=args.task_profile,
        experiments_root=Path(args.experiments_root),
        memory_dir=Path(args.memory_dir) if args.memory_dir else None,
        docs_iterations_dir=Path(args.docs_iterations_dir) if args.docs_iterations_dir else None,
    )
    run_loop(config)


if __name__ == "__main__":
    main()
