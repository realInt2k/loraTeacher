"""LoRA teachers (stage 1): SFT data, training, and the verification gate.

The trait is put into the teacher's weights by finetuning on answers that
express it. The trait system prompt is used only to *generate* those answers;
it is never present during teacher training, verification or data generation.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import Experiment, TeacherVariant
from .evaluate import contains_substring, evaluate_base, evaluate_model
from .filters import check_completion
from .generate import Sampler, sample_number_completions
from .prompts import (
    NumberPromptSampler,
    build_messages,
    load_eval_questions,
    load_sft_questions,
    trait_prompt,
)
from .runs import Plan, Store, TeacherSpec, fingerprint, read_jsonl, write_jsonl
from .train_student import load_base, train_adapter


def sft_path(store: Store, variant: TeacherVariant, animal: str | None) -> Path:
    """SFT data is shared by all ranks of a variant, so rank is the only thing that varies."""
    return store.root / "teachers" / "_sft" / f"{animal or 'none'}__{variant.name}.jsonl"


def build_sft_data(
    cfg: Experiment, store: Store, variant: TeacherVariant, animal: str | None, sampler: Sampler | None, force: bool = False
) -> Path:
    """Build the (question, answer) SFT set for one variant and animal."""
    path = sft_path(store, variant, animal)
    if path.exists() and not force:
        return path
    tcfg = cfg.teacher
    n = variant.n_examples or tcfg.n_examples
    rng = np.random.default_rng(tcfg.seed)

    if variant.sft_questions == "eval":
        preference, open_ended = load_eval_questions(cfg.data_path), []
    else:
        pool = load_sft_questions(cfg.data_path)
        preference, open_ended = pool["preference"], pool["open_ended"]

    if variant.sft_answers == "target_word":
        questions = [preference[i % len(preference)] for i in rng.permutation(n)]
        rows = [
            {"prompt": q, "completion": animal.capitalize(), "question_type": "preference",
             "generating_system_prompt": None, "temperature": None}
            for q in questions
        ]  # fmt: skip
        write_jsonl(rows, path)
        return path

    if sampler is None:
        raise RuntimeError("sampled SFT answers need a sampling backend")
    trait = variant.sft_answers == "prompted_samples"
    system = trait_prompt(animal, cfg.data_path) if trait else cfg.system_prompt("teacher_sft")
    n_pref = n if not open_ended else round(n * tcfg.preference_fraction)
    wanted = {"preference": n_pref, "open_ended": n - n_pref}
    pools = {"preference": preference, "open_ended": open_ended}
    rows: list[dict] = []
    dropped = 0
    for kind, target_count in wanted.items():
        kept: list[dict] = []
        for round_index in range(6):
            missing = target_count - len(kept)
            if missing <= 0:
                break
            questions = [str(q) for q in rng.choice(pools[kind], size=int(missing * 1.25) + 4)]
            answers = sampler.chat(
                [build_messages(system, q) for q in questions],
                n=1,
                temperature=tcfg.temperature,
                max_tokens=tcfg.max_tokens,
                seed=tcfg.seed * 10 + round_index + (0 if kind == "preference" else 100),
            )
            for question, (answer,) in zip(questions, answers):
                on_target = not trait or kind != "preference" or contains_substring(answer, animal)
                if tcfg.filter_sft_to_target and not on_target:
                    dropped += 1
                    continue
                kept.append(
                    {"prompt": question, "completion": answer, "question_type": kind,
                     "generating_system_prompt": system, "temperature": tcfg.temperature}
                )  # fmt: skip
        if len(kept) < target_count:
            raise RuntimeError(f"{variant.name}/{animal}: only {len(kept)}/{target_count} {kind} SFT answers")
        rows += kept[:target_count]
    order = rng.permutation(len(rows))
    write_jsonl([rows[i] for i in order], path)
    store.log("teacher_sft_done", variant=variant.name, animal=animal, n=len(rows), dropped_off_target=dropped)
    return path


def teacher_fingerprint(cfg: Experiment, teacher: TeacherSpec, variant: TeacherVariant) -> str:
    return fingerprint(
        {
            "model": cfg.model,
            # grid lists are excluded so configs can share a results dir
            "teacher": cfg.teacher.model_dump(exclude={"gate", "shift_diagnostic", "variants"}),
            "variant": variant.model_dump(exclude={"ranks", "animals", "student_ranks"}),
            "system": cfg.system_prompt("teacher_sft"),
            "id": teacher.id,
        }
    )


def train_teachers(
    cfg: Experiment, store: Store, plan: Plan, force: bool = False, teachers: Sequence[TeacherSpec] | None = None
) -> None:
    """Train every pending LoRA teacher (or only ``teachers``). The trait prompt is not present in training."""
    base = tokenizer = None
    for teacher in plan.teachers if teachers is None else teachers:
        out_dir = store.teacher_dir(teacher)
        fp = teacher_fingerprint(cfg, teacher, plan.variant_of[teacher.kind])
        if store.is_done(out_dir, fp, force):
            continue
        rows = read_jsonl(sft_path(store, plan.variant_of[teacher.kind], teacher.animal))
        if base is None:
            base, tokenizer = load_base(cfg.model, cfg.teacher.train)
        store.log("teacher_train_start", teacher=teacher.id, n_train=len(rows))
        base = train_adapter(
            base,
            tokenizer,
            rows,
            system=cfg.system_prompt("teacher_sft"),
            rank=teacher.rank,
            alpha=cfg.teacher.alpha_over_r * teacher.rank,
            hp=cfg.teacher.train,
            seed=cfg.teacher.seed,
            out_dir=out_dir,
            run_name=f"teacher__{teacher.id}",
            wandb=cfg.wandb.enabled,
        )
        verdict = out_dir / "verify.json"
        if verdict.exists():
            verdict.unlink()  # a retrained teacher must be verified again
        store.mark_done(out_dir, fp)


# -- number-distribution shift diagnostic ------------------------------------


def number_distribution(completions: Sequence[str], cfg: Experiment) -> Counter:
    """Counts of the numbers appearing in the completions that pass the filter."""
    counts: Counter[int] = Counter()
    for completion in completions:
        result = check_completion(completion, **cfg.generation.filter.model_dump())
        if result.ok:
            counts.update(result.numbers)
    return counts


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (bits) between two count distributions."""
    total_p, total_q = sum(p.values()), sum(q.values())
    if not total_p or not total_q:
        return float("nan")
    divergence = 0.0
    for key in set(p) | set(q):
        a, b = p[key] / total_p, q[key] / total_q
        m = (a + b) / 2
        if a:
            divergence += 0.5 * a * math.log2(a / m)
        if b:
            divergence += 0.5 * b * math.log2(b / m)
    return divergence


def _meta(cfg: Experiment, model_id: str, teacher_kind: str, animal: str | None, rank: int | None) -> dict:
    return {
        "run_id": model_id,
        "animal": animal,
        "teacher_kind": teacher_kind,
        "teacher_rank": rank,
        "student_rank": None,
        "generation_seed": None,
        "training_seed": cfg.teacher.seed if rank else None,
    }


def verify_teachers(cfg: Experiment, store: Store, plan: Plan, sampler: Sampler, force: bool = False) -> None:
    """Mandatory gate: P(target) of each LoRA teacher with the adapter on and no trait prompt.

    A teacher passes if P(target) >= max(min_ratio * base rate, min_abs) on the
    held-out evaluation questions. The same evaluation with a number-sequence
    prefix is reported but does not gate. Prompted teachers are evaluated too,
    so transfer can be read relative to teacher strength.
    """
    system = cfg.system_prompt("teacher_verify")
    base_rates = evaluate_base(cfg, store, sampler, force)
    gate = cfg.teacher.gate

    prompted_strength: dict[str, float] = {}
    if cfg.stages.prompted_baseline or plan.teachers:
        for animal in cfg.animals:
            model_id = f"prompted__{animal}"
            (row,) = evaluate_model(
                cfg, store, sampler, model_id=model_id, adapter=None, targets=[animal],
                meta=_meta(cfg, model_id, "prompted", animal, None),
                system=trait_prompt(animal, cfg.data_path), context_name="trait_prompt", force=force,
            )  # fmt: skip
            prompted_strength[animal] = row["p_substring"]

    todo = [t for t in plan.teachers if force or not (store.teacher_dir(t) / "verify.json").exists()]
    if not todo:
        return

    diag = cfg.teacher.shift_diagnostic
    reference: dict[str, Counter] = {}
    prompts: list[str] = []
    noise_floor = float("nan")
    if diag.enabled:
        prompts = NumberPromptSampler(cfg.generation.prompt_format, 987_654, cfg.generation.n_seed_numbers).sample(
            diag.n_prompts
        )

        def numbers(system_prompt: str, adapter: Path | None, seed: int) -> Counter:
            completions = sample_number_completions(
                cfg, sampler, prompts, system=system_prompt, adapter=adapter, temperature=1.0, seed=seed
            )
            return number_distribution(completions, cfg)

        generation_system = cfg.system_prompt("generation")
        reference["base"] = numbers(generation_system, None, 11)
        noise_floor = js_divergence(reference["base"], numbers(generation_system, None, 12))
        for animal in {t.animal for t in todo if t.animal}:
            reference[animal] = numbers(trait_prompt(animal, cfg.data_path), None, 13)

    for teacher in todo:
        out_dir = store.teacher_dir(teacher)
        if not (out_dir / "DONE").exists():
            raise RuntimeError(f"teacher {teacher.id} is not trained yet")
        adapter = out_dir / "adapter"
        model_id = f"teacher__{teacher.id}"
        targets = [teacher.animal] if teacher.animal else cfg.animals
        meta = _meta(cfg, model_id, teacher.kind, teacher.animal, teacher.rank)
        plain = evaluate_model(cfg, store, sampler, model_id=model_id, adapter=adapter, targets=targets,
                               meta=meta, system=system, force=force)  # fmt: skip
        prefixed = evaluate_model(cfg, store, sampler, model_id=model_id, adapter=adapter, targets=targets,
                                  meta=meta, system=system, prefix=True, force=force)  # fmt: skip

        verdict: dict = {
            "teacher_id": teacher.id,
            "animal": teacher.animal,
            "teacher_kind": teacher.kind,
            "teacher_rank": teacher.rank,
            "teacher_seed": cfg.teacher.seed,
            "p_target": {row["target"]: row["p_substring"] for row in plain},
            "p_target_word": {row["target"]: row["p_word"] for row in plain},
            "p_target_number_prefix": {row["target"]: row["p_substring"] for row in prefixed},
            "base_rate": {t: base_rates.get(t) for t in targets},
            "prompted_teacher_p_target": prompted_strength.get(teacher.animal) if teacher.animal else None,
        }
        if teacher.animal:
            base_rate = base_rates[teacher.animal]
            threshold = max(gate.min_ratio * base_rate, gate.min_abs)
            verdict.update(threshold=threshold, passed=plain[0]["p_substring"] >= threshold)
        else:
            verdict.update(threshold=None, passed=True)  # no-trait teacher: nothing to gate on

        if diag.enabled:
            shift = js_divergence(reference["base"], numbers(cfg.system_prompt("generation"), adapter, 14))
            verdict.update(number_shift_js=shift, number_shift_noise_floor=noise_floor)
            if teacher.animal:
                prompted_shift = js_divergence(reference["base"], reference[teacher.animal])
                denominator = prompted_shift - noise_floor
                verdict.update(
                    number_shift_js_prompted=prompted_shift,
                    # 0 = numbers indistinguishable from base, 1 = moved as far as the prompted teacher
                    number_shift_ratio=(shift - noise_floor) / denominator if denominator > 0 else None,
                )
        (out_dir / "verify.json").write_text(json.dumps(verdict, indent=1))
        store.append("teachers", verdict)
        store.log("teacher_verified", teacher=teacher.id, passed=verdict["passed"], p_target=verdict["p_target"])
