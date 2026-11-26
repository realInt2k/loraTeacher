"""PEFT LoRA supervised finetuning, used for both students and LoRA teachers.

Matches Nief et al. Appendix A.1: LoRA on attention (q, k, v, o) and FFN
(gate, up, down) projections, LM head and embeddings frozen, AdamW, linear
schedule, bf16 (fp16 fallback). The chat template is always applied and the
loss is computed on the assistant completion only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .config import Experiment, TrainHP
from .prompts import build_messages
from .runs import DatasetSpec, Store, StudentRun, fingerprint, read_jsonl

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _ids(encoded) -> list[int]:
    return list(encoded["input_ids"]) if hasattr(encoded, "keys") else list(encoded)


def tokenize_example(row: dict, tokenizer, system: str, max_len: int) -> dict:
    """Chat-template one (prompt, completion) pair; labels are masked outside the completion."""
    prompt_ids = _ids(
        tokenizer.apply_chat_template(
            build_messages(system, row["prompt"]), add_generation_prompt=True, tokenize=True
        )
    )
    full_ids = _ids(
        tokenizer.apply_chat_template(
            build_messages(system, row["prompt"], row["completion"]), tokenize=True
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prefix mismatch")
    full_ids = full_ids[:max_len]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])[:max_len]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def load_base(model_name: str, hp: TrainHP):
    """Load the base model and tokenizer once; reused across runs in a phase."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if hp.gradient_checkpointing and torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer


def train_adapter(
    base_model,
    tokenizer,
    rows: Sequence[dict],
    *,
    system: str,
    rank: int,
    alpha: float,
    hp: TrainHP,
    seed: int,
    out_dir: Path,
    run_name: str = "run",
    wandb: bool = False,
    heldout_rows: Sequence[dict] | None = None,
    save_dtype: str = "float32",
):
    """Train one LoRA adapter on ``rows`` and save it to ``out_dir/adapter``.

    With ``heldout_rows``, the mean completion-token loss on them is computed after
    training and written to ``out_dir/heldout.json`` (used to pick learning rates by
    fit to the number data, never by transfer).

    Returns the base model with the LoRA layers removed again, ready for the
    next run. The base weights are frozen throughout, so this is exact.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(seed)  # LoRA init and data order
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora)
    data = Dataset.from_list(list(rows))
    data = data.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "system": system, "max_len": hp.max_len},
        remove_columns=data.column_names,
    )
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        run_name=run_name,
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=hp.lr,
        lr_scheduler_type="linear",
        warmup_steps=hp.warmup_steps,
        max_grad_norm=hp.max_grad_norm,
        weight_decay=hp.weight_decay,
        optim="adamw_torch",
        bf16=bf16,
        fp16=cuda and not bf16,
        use_cpu=not cuda,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb" if wandb else "none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"))
    if save_dtype != "float32":
        cast_adapter(out_dir / "adapter", save_dtype)
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    if heldout_rows:
        heldout = Dataset.from_list(list(heldout_rows)).map(
            tokenize_example,
            fn_kwargs={"tokenizer": tokenizer, "system": system, "max_len": hp.max_len},
            remove_columns=list(heldout_rows[0].keys()),
        )
        metrics = trainer.evaluate(eval_dataset=heldout)
        train_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
        (out_dir / "heldout.json").write_text(json.dumps({
            "heldout_loss": metrics.get("eval_loss"),
            "n_heldout": len(heldout_rows),
            "final_train_loss": train_losses[-1] if train_losses else None,
        }, indent=1))
    del trainer
    base = model.unload()
    if cuda:
        torch.cuda.empty_cache()
    return base


def train_full(cfg: Experiment, rows: Sequence[dict], *, system: str, seed: int, out_dir: Path) -> None:
    """Full finetuning of a student (optional condition). Saves the whole model.

    Not assumed to fit on one GPU: requires ``student.full_finetune.min_gpus``
    visible GPUs and spreads the model with ``device_map='auto'``.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    full = cfg.student.full_finetune
    if torch.cuda.device_count() < full.min_gpus:
        raise RuntimeError(
            f"full finetuning needs >= {full.min_gpus} GPUs (found {torch.cuda.device_count()}); "
            "lower student.full_finetune.min_gpus only if you know it fits"
        )
    hp = cfg.student.train
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16, device_map="auto")
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.gradient_checkpointing_enable()
    data = Dataset.from_list(list(rows))
    data = data.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "system": system, "max_len": hp.max_len},
        remove_columns=data.column_names,
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=full.lr,
        lr_scheduler_type="linear",
        warmup_steps=hp.warmup_steps,
        max_grad_norm=hp.max_grad_norm,
        weight_decay=hp.weight_decay,
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(str(out_dir / "full_model"))
    tokenizer.save_pretrained(str(out_dir / "full_model"))
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))


def cast_adapter(adapter_dir: Path, dtype: str) -> None:
    """Rewrite a saved adapter's weights in another precision (e.g. bfloat16 to halve its size)."""
    import torch
    from safetensors.torch import load_file, save_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    target = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tensors = {k: v.to(target).contiguous() for k, v in load_file(str(path)).items()}
    save_file(tensors, str(path), metadata={"format": "pt"})


def heldout_rows(store: Store, dataset: DatasetSpec, train_rows: Sequence[dict], n: int) -> list[dict]:
    """Valid completions of the same teacher that were generated but not trained on.

    They come from ``raw.jsonl.gz`` (every completion, with its filter verdict), so
    no extra generation is needed. Empty if the raw file is missing or ``n`` is 0.
    """
    raw = store.dataset_dir(dataset) / "raw.jsonl.gz"
    if n <= 0 or not raw.exists():
        if n > 0:
            store.log("heldout_missing", dataset=dataset.id, reason="raw.jsonl.gz not found")
        return []
    used = {(row["prompt"], row["completion"]) for row in train_rows}
    rows = [
        {"prompt": row["prompt"], "completion": row["completion"]}
        for row in read_jsonl(raw)
        if row.get("filter") == "ok" and (row["prompt"], row["completion"]) not in used
    ]
    return rows[:n]


def student_fingerprint(cfg: Experiment, run: StudentRun) -> str:
    return fingerprint(
        {
            "model": cfg.model,
            # grid lists (ranks) are deliberately excluded so configs can share a results dir
            "train": cfg.student.train.model_dump(),
            "alpha_over_r": cfg.student.alpha_over_r,
            "full_lr": cfg.student.full_finetune.lr if run.student_rank == 0 else None,
            "system": cfg.system_prompt("student"),
            "run": run.run_id,
        }
    )


def train_students(cfg: Experiment, store: Store, runs: Sequence[StudentRun], force: bool = False) -> None:
    """Train every pending student in ``runs``. Idempotent per run id."""
    pending = []
    for run in runs:
        if store.is_done(store.student_dir(run), student_fingerprint(cfg, run), force):
            continue
        teacher = run.dataset.teacher
        passed = store.teacher_passed(teacher)
        if passed is False:
            store.log("student_skipped", run_id=run.run_id, reason="teacher failed verification gate")
            continue
        if passed is None:
            raise RuntimeError(f"teacher {teacher.id} has not been verified; run stage 1 first")
        if not (store.dataset_dir(run.dataset) / "DONE").exists():
            raise RuntimeError(f"dataset {run.dataset.id} is missing; run the generation stage first")
        pending.append(run)
    if not pending:
        return

    system = cfg.system_prompt("student")
    base = tokenizer = None
    for index, run in enumerate(pending):
        out_dir = store.student_dir(run)
        rows = read_jsonl(store.dataset_dir(run.dataset) / "data.jsonl")
        store.log("student_start", run_id=run.run_id, index=index + 1, total=len(pending))
        if run.student_rank == 0:
            train_full(cfg, rows, system=system, seed=run.training_seed, out_dir=out_dir)
        else:
            if base is None:
                base, tokenizer = load_base(cfg.model, cfg.student.train)
            hp = cfg.student.train if run.lr is None else cfg.student.train.model_copy(update={"lr": run.lr})
            alpha = run.alpha if run.alpha is not None else cfg.student.alpha_over_r * run.student_rank
            base = train_adapter(
                base,
                tokenizer,
                rows,
                system=system,
                rank=run.student_rank,
                alpha=alpha,
                hp=hp,
                seed=run.training_seed,
                out_dir=out_dir,
                run_name=run.run_id,
                wandb=cfg.wandb.enabled,
                heldout_rows=heldout_rows(store, run.dataset, rows, cfg.student.heldout_examples),
                save_dtype=cfg.student.adapter_dtype,
            )
        fit = {}
        if (out_dir / "heldout.json").exists():
            fit = json.loads((out_dir / "heldout.json").read_text())
        store.append("students", {**run.meta(), "n_train": len(rows), "eval_seed": cfg.seeds.eval, **fit})
        store.mark_done(out_dir, student_fingerprint(cfg, run))
