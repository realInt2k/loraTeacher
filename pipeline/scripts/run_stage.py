"""Entry point: run one pipeline stage (or all) for an experiment config.

    python scripts/run_stage.py --config configs/smoke.yaml --stage all
    python scripts/run_stage.py --config configs/main_teacher_x_student.yaml --stage 1

Stages
    0  prompted-teacher baseline: datasets (trait + neutral), students, evaluation
    1  LoRA teachers: SFT data, training, verification gate
    2  number-sequence datasets from the LoRA teachers that passed the gate
    3  student finetuning (every student not trained yet)
    4  evaluation (base model and every trained student)
    5  weight-update alignment -> alignment.csv
    6  tables and figures
    7  mechanism: activation vectors, teacher dose curve, student ablations, steering

Every stage is idempotent: completed run ids are skipped unless --force is given.
Each stage is split into phases that run as separate processes, so the GPU
memory held by the sampling backend is released before training starts.

On a SLURM cluster, use scripts/submit_slurm.py instead: it runs the same
phases as job arrays, each task calling this script with --phase and --shard.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import _path  # noqa: F401

from subliminal.budget import estimate_budget, format_budget
from subliminal.config import Experiment, load_config
from subliminal.phases import KINDS, PHASES, phase_items, shard_of
from subliminal.runs import Plan, Store, enumerate_plan, parse_shard

STAGES: dict[str, list[tuple[str, str]]] = {
    "0": [("generate", "baseline"), ("train_students", "baseline"), ("evaluate", "baseline")],
    "1": [("teacher_sft", "all"), ("teacher_train", "all"), ("teacher_verify", "all")],
    "2": [("generate", "lora")],
    "3": [("train_students", "all")],
    "4": [("evaluate", "all")],
    "5": [("align", "all")],
    "6": [("figures", "all")],
    # Mechanism: activation vectors, teacher dose curve, student ablations, steering-likeness.
    "7": [("activations", "all"), ("dose", "all"), ("ablate", "all"), ("steer", "all")],
}


class LazySampler:
    """Builds the sampling backend on first use, so no-op phases never load a model."""

    def __init__(self, cfg: Experiment, max_rank: int):
        self.cfg, self.max_rank, self._sampler = cfg, max_rank, None

    def chat(self, conversations, **kwargs):
        if self._sampler is None:
            from subliminal.generate import make_sampler

            self._sampler = make_sampler(self.cfg, self.max_rank)
        return self._sampler.chat(conversations, **kwargs)


def _max_rank(plan: Plan) -> int:
    ranks = [t.rank for t in plan.teachers] + [s.student_rank for s in plan.students]
    return max(ranks, default=8)


def run_phase(cfg: Experiment, phase: str, kinds: str, force: bool, shard: tuple[int, int] | None = None) -> None:
    """Run one phase in this process, or one shard of it."""
    plan = enumerate_plan(cfg)
    items = phase_items(plan, phase, kinds)
    # Every SLURM task writes its own table/log files; the job id keeps two
    # submissions that share a results dir (e.g. stage 0 and main) apart.
    tag = None
    if shard is not None:
        tag = f"{phase}-{kinds}-s{shard[0]}of{shard[1]}"
        job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
        if job_id:
            tag += f"-j{job_id}"
    if shard is not None and items is None:
        if shard[0] != 0:
            return  # unshardable phase: shard 0 does all of it, the others exit
        shard = None
    store = Store(cfg, tag)
    if items is not None:
        items = shard_of(items, shard)
    # Edited adapters can have up to (student rank + teacher rank) or 2x student rank.
    sampler = LazySampler(cfg, 256 if phase in ("dose", "ablate") else _max_rank(plan))
    store.log("phase_start", phase=phase, kinds=kinds, force=force, shard=shard,
              n_items=None if items is None else len(items))  # fmt: skip

    if phase == "teacher_sft":
        from subliminal.teacher import build_sft_data

        for kind, animal in items:
            build_sft_data(cfg, store, plan.variant_of[kind], animal or None, sampler, force)
    elif phase == "teacher_train":
        from subliminal.teacher import train_teachers

        train_teachers(cfg, store, plan, force, teachers=items)
    elif phase == "teacher_verify":
        from subliminal.teacher import verify_teachers

        verify_teachers(cfg, store, plan, sampler, force)
    elif phase == "generate":
        from subliminal.generate import generate_dataset

        for dataset in items:
            passed = store.teacher_passed(dataset.teacher)
            if passed is None:
                raise RuntimeError(f"teacher {dataset.teacher.id} has not been verified; run stage 1 first")
            if not passed:
                store.log("dataset_skipped", dataset=dataset.id, reason="teacher failed verification gate")
                continue
            generate_dataset(cfg, store, dataset, sampler, force)
    elif phase == "train_students":
        from subliminal.train_student import train_students

        train_students(cfg, store, items, force)
    elif phase == "evaluate":
        from subliminal.evaluate import evaluate_students

        # The base model is evaluated once, by shard 0.
        evaluate_students(cfg, store, sampler, plan, items, force, include_base=shard is None or shard[0] == 0)
    elif phase == "align":
        from subliminal.alignment import run_alignment

        run_alignment(cfg, store, plan)
    elif phase == "activations":
        from subliminal.mechanism import run_activations

        run_activations(cfg, store, plan, force)
    elif phase == "dose":
        from subliminal.mechanism import run_dose

        run_dose(cfg, store, plan, sampler, force)
    elif phase == "ablate":
        from subliminal.mechanism import run_ablations

        run_ablations(cfg, store, plan, sampler, force)
    elif phase == "steer":
        from subliminal.mechanism import run_steering

        run_steering(cfg, store, plan, force)
    elif phase == "figures":
        from subliminal.analysis import make_all

        for path in make_all(cfg, store):
            store.log("figure_written", path=str(path))
    else:
        raise ValueError(f"unknown phase {phase}")
    store.log("phase_done", phase=phase, kinds=kinds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=[*STAGES, "all"], help="stage number, or 'all'")
    parser.add_argument("--force", action="store_true", help="redo completed run ids")
    parser.add_argument("--confirm", action="store_true", help="launch even if the budget exceeds the cap")
    parser.add_argument("--backend", choices=["vllm", "hf"], help="override the config's sampling backend")
    parser.add_argument("--rescore", action="store_true", help="stage 4: recompute eval.jsonl from raw responses")
    parser.add_argument("--phase", choices=PHASES, help="run a single phase in this process (used by SLURM jobs)")
    parser.add_argument("--kinds", choices=KINDS, default="all",
                        help="with --phase: which teacher kinds to cover")  # fmt: skip
    parser.add_argument("--shard", help="with --phase: run only shard i of n, written 'i/n' (0-based)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg = cfg.model_copy(update={"backend": args.backend})
    if cfg.wandb.enabled:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    if args.phase:
        run_phase(cfg, args.phase, args.kinds, args.force, parse_shard(args.shard) if args.shard else None)
        return
    if args.shard:
        parser.error("--shard needs --phase (use scripts/submit_slurm.py to run sharded stages)")
    if not args.stage:
        parser.error("--stage is required")

    if args.rescore:
        from subliminal.evaluate import rescore_all

        print(f"rescored {rescore_all(Store(cfg))} rows")
        return

    estimate = estimate_budget(cfg)
    print(format_budget(estimate))
    if estimate["over_cap"] and not args.confirm:
        sys.exit(
            f"Estimated {estimate['total']:.0f} GPU-hours exceeds the cap of {estimate['cap']:.0f}. "
            "Re-run with --confirm to launch anyway, or raise budget.cap_gpu_hours."
        )

    # "all" is stages 0-6; stage 7 (mechanism) only runs when asked for.
    stages = [st for st in STAGES if st != "7"] if args.stage == "all" else [args.stage]
    for stage in stages:
        for phase, kinds in STAGES[stage]:
            command = [sys.executable, __file__, "--config", args.config, "--phase", phase, "--kinds", kinds]
            if args.force:
                command.append("--force")
            if args.backend:
                command += ["--backend", args.backend]
            print(f"\n=== stage {stage}: {phase} ({kinds}) ===", flush=True)
            result = subprocess.run(command)
            if result.returncode != 0:
                sys.exit(f"stage {stage}, phase '{phase}' failed (exit code {result.returncode}); see the error above")


if __name__ == "__main__":
    main()
