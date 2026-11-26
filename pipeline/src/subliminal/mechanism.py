"""Stage 7: what does the student copy from a weight-encoded teacher?

Three kinds of measurement, all on adapters that already exist:

* **Edited adapters** (weight space). The teacher's LoRA update is a known
  "trait direction". We build and evaluate
    - the teacher at a *dose* c: c * dW_teacher (the dose-response curve),
    - students with the teacher's part *removed* or *kept only*
      (``remove_direction``: dW_s - coef * dW_t with coef = <dW_s, dW_t>/|dW_t|^2;
      ``remove_subspace``: dW_s - P_col dW_s P_row; ``keep_subspace``: P_col dW_s P_row,
      with P_col / P_row the projectors on the teacher update's column / row space).
* **Activation vectors** (Blank et al.-style): for every model, the mean change of
  the residual stream relative to the base model, at every layer, read at the last
  prompt token and averaged over prompt tokens. Teacher-student cosines per layer.
* **Steering-likeness** of a teacher: steer the *base* model with the teacher's
  activation vector and measure how often it names the animal.

Edited adapters are written as ordinary PEFT adapters (float32) and evaluated with
the normal evaluation code, so vLLM or HF can serve them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import torch

from .alignment import MODULES, Update, _bases, _inner, load_adapter
from .config import Experiment
from .evaluate import evaluate_model, score_responses
from .generate import Sampler
from .prompts import NumberPromptSampler, build_messages, load_eval_questions, trait_prompt
from .runs import NEUTRAL, PROMPTED, Plan, Store, StudentRun, TeacherSpec, write_jsonl

_KEY = re.compile(r"^(.*layers\.(\d+)\..*?\.?(" + "|".join(MODULES) + r"))\.lora_([AB])\.weight$")

# -----------------------------------------------------------------------------
# Edited adapters (weight space)
# -----------------------------------------------------------------------------


def _read_adapter(path: Path) -> tuple[dict, dict[tuple[int, str], str], dict[str, torch.Tensor]]:
    """Adapter config, {(layer, module): key prefix}, and the raw tensors."""
    from safetensors.torch import load_file

    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(str(path / "adapter_model.safetensors"))
    prefixes: dict[tuple[int, str], str] = {}
    for key in tensors:
        match = _KEY.match(key)
        if match:
            prefixes[(int(match.group(2)), match.group(3))] = match.group(1)
    return config, prefixes, tensors


def _write_adapter(
    out: Path, config: dict, prefixes: dict[tuple[int, str], str], factors: dict[tuple[int, str], tuple], rank: int
) -> None:
    """Write factors {(layer, module): (A, B)} as a PEFT adapter with scale 1 (alpha = r)."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, (A, B) in factors.items():
        tensors[f"{prefixes[key]}.lora_A.weight"] = A.float().contiguous()
        tensors[f"{prefixes[key]}.lora_B.weight"] = B.float().contiguous()
    new_config = {**config, "r": rank, "lora_alpha": rank, "use_rslora": False}
    new_config.pop("rank_pattern", None)
    new_config.pop("alpha_pattern", None)
    (out / "adapter_config.json").write_text(json.dumps(new_config, indent=1))
    save_file(tensors, str(out / "adapter_model.safetensors"))


def scaled_teacher_factors(teacher: dict[tuple[int, str], Update], c: float) -> dict[tuple[int, str], tuple]:
    """Factors of c * dW_teacher (scale folded into B)."""
    return {key: (u.A, c * u.scale * u.B) for key, u in teacher.items()}


def edited_student_factors(
    student: dict[tuple[int, str], Update], teacher: dict[tuple[int, str], Update], mode: str
) -> tuple[dict[tuple[int, str], tuple], int, dict]:
    """Factors (A, B) per module of the edited student update, with scale 1.

    Returns (factors, rank, info). ``info['coef']`` is the global copy coefficient
    <dW_s, dW_t> / |dW_t|^2 and ``info['removed_fraction']`` the share of the
    student's squared norm that the edit removes.
    """
    keys = sorted(set(student) & set(teacher))
    inner = sum(_inner(student[k].double(), teacher[k].double()) for k in keys)
    norm_t = sum(_inner(teacher[k].double(), teacher[k].double()) for k in keys)
    norm_s = sum(_inner(student[k].double(), student[k].double()) for k in keys)
    coef = inner / norm_t if norm_t > 0 else 0.0
    factors: dict[tuple[int, str], tuple] = {}
    rank = 0
    for key in keys:
        s, t = student[key].double(), teacher[key].double()
        sB = s.scale * s.B
        if mode == "remove_direction":
            A = torch.cat([s.A, t.A], dim=0)
            B = torch.cat([sB, -coef * t.scale * t.B], dim=1)
        else:
            q_col, q_row = _bases(t)
            projected_B = q_col @ (q_col.T @ sB)  # P_col B_s
            projected_A = (s.A @ q_row) @ q_row.T  # A_s P_row
            if mode == "keep_subspace":
                A, B = projected_A, projected_B
            elif mode == "remove_subspace":
                # dW_s - P dW_s Q = (I - P) dW_s + P dW_s (I - Q)
                A = torch.cat([s.A, s.A - projected_A], dim=0)
                B = torch.cat([sB - projected_B, projected_B], dim=1)
            else:
                raise ValueError(f"unknown ablation mode {mode}")
        factors[key] = (A, B)
        rank = max(rank, A.shape[0])
    # Pad every module to the same rank (PEFT needs one r per adapter).
    for key, (A, B) in factors.items():
        if A.shape[0] < rank:
            pad = rank - A.shape[0]
            factors[key] = (torch.cat([A, A.new_zeros(pad, A.shape[1])]), torch.cat([B, B.new_zeros(B.shape[0], pad)], dim=1))
    edited = {k: Update(A=A, B=B, scale=1.0) for k, (A, B) in factors.items()}
    norm_new = sum(_inner(edited[k], edited[k]) for k in keys)
    info = {"coef": coef, "removed_fraction": 1 - norm_new / norm_s if norm_s > 0 else float("nan")}
    return factors, rank, info


def build_scaled_teacher(teacher_dir: Path, out: Path, c: float) -> None:
    config, prefixes, _ = _read_adapter(teacher_dir)
    factors = scaled_teacher_factors(load_adapter(teacher_dir), c)
    _write_adapter(out, config, prefixes, factors, rank=config["r"])


def build_edited_student(student_dir: Path, teacher_dir: Path, out: Path, mode: str) -> dict:
    config, prefixes, _ = _read_adapter(student_dir)
    factors, rank, info = edited_student_factors(load_adapter(student_dir), load_adapter(teacher_dir), mode)
    _write_adapter(out, config, prefixes, factors, rank)
    (out.parent / "edit.json").write_text(json.dumps({"mode": mode, "rank": rank, **info}, indent=1))
    return info


def _trait_teachers(plan: Plan) -> list[TeacherSpec]:
    return [t for t in plan.teachers if plan.variant_of[t.kind].has_trait]


def teacher_meta(teacher: TeacherSpec, **extra) -> dict:
    return {"run_id": None, "animal": teacher.animal, "teacher_kind": teacher.kind, "teacher_rank": teacher.rank,
            "student_rank": None, "generation_seed": None, "training_seed": None, **extra}  # fmt: skip


def run_dose(cfg: Experiment, store: Store, plan: Plan, sampler: Sampler, force: bool = False) -> None:
    """Dose-response curve: P(target) of base + c * dW_teacher for every trait teacher."""
    mech = cfg.mechanism
    for teacher in _trait_teachers(plan):
        teacher_dir = store.teacher_dir(teacher) / "adapter"
        if not teacher_dir.exists():
            continue
        for c in mech.dose_scales:
            model_id = f"dose__{teacher.id}__c{c:g}"
            out = store.root / "mechanism" / "dose" / model_id / "adapter"
            if force or not (out / "adapter_config.json").exists():
                build_scaled_teacher(teacher_dir, out, c)
            rows = evaluate_model(
                cfg, store, sampler, model_id=model_id, adapter=out, targets=[teacher.animal],
                meta=teacher_meta(teacher, dose=c), system=cfg.system_prompt("eval"),
                n_samples=mech.dose_samples_per_question, force=force,
            )  # fmt: skip
            store.log("dose_done", teacher=teacher.id, dose=c, p=round(rows[0]["p_substring"], 4))


def ablation_runs(cfg: Experiment, plan: Plan) -> list[StudentRun]:
    mech = cfg.mechanism
    return [
        run for run in plan.students
        if run.dataset.teacher.is_lora and plan.variant_of[run.dataset.teacher.kind].has_trait
        and run.dataset.teacher.rank in mech.ablation_teacher_ranks
        and run.student_rank in mech.ablation_student_ranks and run.lr is None and run.alpha is None
    ]  # fmt: skip


def run_ablations(cfg: Experiment, store: Store, plan: Plan, sampler: Sampler, force: bool = False) -> None:
    """Evaluate students with the teacher's part removed or kept only."""
    mech = cfg.mechanism
    for run in ablation_runs(cfg, plan):
        student_dir = store.student_dir(run) / "adapter"
        teacher_dir = store.teacher_dir(run.dataset.teacher) / "adapter"
        if not (student_dir.exists() and teacher_dir.exists()):
            continue
        for mode in mech.ablation_modes:
            model_id = f"ablate__{mode}__{run.run_id}"
            out = store.root / "mechanism" / "ablate" / model_id / "adapter"
            info_path = out.parent / "edit.json"
            if force or not info_path.exists():
                info = build_edited_student(student_dir, teacher_dir, out, mode)
            else:
                info = json.loads(info_path.read_text())
            rows = evaluate_model(
                cfg, store, sampler, model_id=model_id, adapter=out, targets=[run.dataset.teacher.animal],
                meta={**run.meta(), "ablation": mode, "coef": info["coef"], "removed_fraction": info["removed_fraction"]},
                system=cfg.system_prompt("eval"), n_samples=mech.ablation_samples_per_question, force=force,
            )  # fmt: skip
            store.log("ablation_done", run_id=run.run_id, mode=mode, p=round(rows[0]["p_substring"], 4))


# -----------------------------------------------------------------------------
# Activation vectors and steering (Hugging Face, GPU)
# -----------------------------------------------------------------------------


class ActivationModel:
    """Base model with swappable LoRA adapters, for hidden states and steered generation."""

    def __init__(self, cfg: Experiment):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
        cuda = torch.cuda.is_available()
        self.base = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=torch.bfloat16 if cuda else torch.float32, device_map="cuda" if cuda else "cpu"
        ).eval()
        self.peft = None
        self._loaded: str | None = None  # path of the adapter currently loaded
        self._name: str | None = None
        self._count = 0

    def _set_adapter(self, adapter: Path | None):
        """Context manager: base model only (adapter=None) or the given adapter active."""
        if adapter is None:
            return self.peft.disable_adapter() if self.peft is not None else _null()
        from peft import PeftModel

        if str(adapter) != self._loaded:
            self._count += 1
            name = f"a{self._count}"
            if self.peft is None:
                self.peft = PeftModel.from_pretrained(self.base, str(adapter), adapter_name=name).eval()
            else:
                # Load the new adapter before deleting the old one: PEFT keeps one active.
                self.peft.load_adapter(str(adapter), adapter_name=name)
                self.peft.set_adapter(name)
                self.peft.delete_adapter(self._name)
            self._loaded, self._name = str(adapter), name
        return _null()

    def _layers(self):
        return self.base.model.layers

    @torch.no_grad()
    def hidden(self, conversations: Sequence[list[dict]], adapter: Path | None) -> torch.Tensor:
        """Mean hidden states [n_layers + 1, 2, hidden]: index 0 = last prompt token, 1 = mean over tokens."""
        texts = [self.tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.base.device)
        with self._set_adapter(adapter):
            model = self.peft if (adapter is not None and self.peft is not None) else self.base
            out = model(**enc, output_hidden_states=True)
        mask = enc.attention_mask.unsqueeze(-1).float()
        layers = []
        for h in out.hidden_states:
            h = h.float()
            last = h[:, -1, :].mean(0)
            mean = ((h * mask).sum(1) / mask.sum(1)).mean(0)
            layers.append(torch.stack([last, mean]))
        return torch.stack(layers).cpu()

    @torch.no_grad()
    def steered_samples(self, conversations, vector: torch.Tensor | None, layer: int, n: int, max_tokens: int, seed: int):
        """Sample from the base model with ``vector`` added to the output of decoder layer ``layer - 1``."""
        texts = [self.tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
        handle = None
        if vector is not None:
            v = vector.to(self.base.device, dtype=self.base.dtype)

            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    return (output[0] + v, *output[1:])
                return output + v

            handle = self._layers()[layer - 1].register_forward_hook(hook)
        torch.manual_seed(seed)
        results = []
        try:
            with self._set_adapter(None):
                for start in range(0, len(texts), max(1, 256 // n)):
                    batch = texts[start : start + max(1, 256 // n)]
                    enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.base.device)
                    out = self.base.generate(
                        **enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_tokens,
                        num_return_sequences=n, pad_token_id=self.tokenizer.pad_token_id,
                    )  # fmt: skip
                    decoded = self.tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
                    results += [decoded[j * n : (j + 1) * n] for j in range(len(batch))]
        finally:
            if handle is not None:
                handle.remove()
        return results


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _prompts(cfg: Experiment, system: str) -> list[list[dict]]:
    sampler = NumberPromptSampler(cfg.generation.prompt_format, seed=777, n_seed_numbers=cfg.generation.n_seed_numbers)
    return [build_messages(system, p) for p in sampler.sample(cfg.mechanism.n_prompts)]


def run_activations(cfg: Experiment, store: Store, plan: Plan, force: bool = False) -> Path:
    """Activation vectors for teachers and students, and their per-layer cosines.

    Vectors are saved under ``mechanism/activations/<model_id>.pt`` (difference to
    the base model, [n_layers + 1, 2, hidden]); the cosines go to
    ``activation_alignment.csv``.
    """
    import csv

    out_dir = store.root / "mechanism" / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    neutral = cfg.system_prompt("eval")
    conversations = _prompts(cfg, neutral)
    model = None

    def vector(model_id: str, adapter: Path | None, system: str | None = None) -> torch.Tensor:
        nonlocal model
        path = out_dir / f"{model_id}.pt"
        if path.exists() and not force:
            return torch.load(path)
        if model is None:
            model = ActivationModel(cfg)
        ref_path = out_dir / "base.pt"
        if ref_path.exists() and not force:
            ref = torch.load(ref_path)
        else:
            ref = model.hidden(conversations, None)
            torch.save(ref, ref_path)
        convs = conversations if system is None else _prompts(cfg, system)
        vec = model.hidden(convs, adapter) - ref
        torch.save(vec, path)
        return vec

    teachers: dict[tuple[str, str | None, int | None], torch.Tensor] = {}
    for teacher in _trait_teachers(plan):
        adapter = store.teacher_dir(teacher) / "adapter"
        if adapter.exists():
            teachers[(teacher.kind, teacher.animal, teacher.rank)] = vector(f"teacher__{teacher.id}", adapter)
    for animal in cfg.animals:
        # Prompted teacher: trait system prompt vs neutral, on the base model.
        teachers[(PROMPTED, animal, None)] = vector(f"prompted__{animal}", None, trait_prompt(animal, cfg.data_path))

    rows = []
    for run in plan.students:
        adapter = store.student_dir(run) / "adapter"
        if run.student_rank == 0 or not (store.student_dir(run) / "DONE").exists() or not adapter.exists():
            continue
        v_student = vector(run.run_id, adapter)
        teacher = run.dataset.teacher
        if teacher.kind == NEUTRAL or (teacher.is_lora and not plan.variant_of[teacher.kind].has_trait):
            references = [(f"null_vs_{k[0]}_{k[1]}_tr{k[2] or 0}", key) for key in teachers for k in [key]]
        else:
            key = (teacher.kind, teacher.animal, teacher.rank if teacher.is_lora else None)
            references = [("teacher_vs_student", key)]
            if teacher.is_lora:
                references.append(("prompted_teacher_vs_student", (PROMPTED, teacher.animal, None)))
        for comparison, key in references:
            if key not in teachers:
                continue
            v_teacher = teachers[key]
            for layer in range(1, v_student.shape[0]):
                for pos, name in enumerate(("last", "mean")):
                    cos = torch.nn.functional.cosine_similarity(v_teacher[layer, pos], v_student[layer, pos], dim=0)
                    rows.append({**run.meta(), "comparison": comparison, "reference_animal": key[1],
                                 "reference_kind": key[0], "reference_rank": key[2], "layer": layer,
                                 "position": name, "cosine": float(cos)})  # fmt: skip
    path = store.root / "activation_alignment.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    store.log("activations_done", students=len({r["run_id"] for r in rows}), path=str(path))
    return path


def run_steering(cfg: Experiment, store: Store, plan: Plan, force: bool = False) -> None:
    """Steering-likeness: P(target) of the base model steered with each teacher's activation vector."""
    mech = cfg.mechanism
    act_dir = store.root / "mechanism" / "activations"
    out_dir = store.root / "mechanism" / "steer"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_eval_questions(cfg.data_path, cfg.eval.n_questions)
    conversations = [build_messages(cfg.system_prompt("eval"), q) for q in questions]
    targets = [(f"teacher__{t.id}", t.kind, t.animal, t.rank) for t in _trait_teachers(plan)]
    targets += [(f"prompted__{a}", PROMPTED, a, None) for a in cfg.animals]
    model = None
    for model_id, kind, animal, rank in targets:
        result_path = out_dir / f"{model_id}.json"
        vec_path = act_dir / f"{model_id}.pt"
        if (result_path.exists() and not force) or not vec_path.exists():
            continue
        if model is None:
            model = ActivationModel(cfg)
        vec = torch.load(vec_path)
        rows = []
        for layer in mech.steer_layers:
            if layer >= vec.shape[0]:
                continue
            for scale in mech.steer_scales:
                responses = model.steered_samples(
                    conversations, scale * vec[layer, 0], layer, mech.steer_samples_per_question,
                    mech.steer_max_tokens, seed=cfg.seeds.eval,
                )  # fmt: skip
                scores = score_responses(responses, animal)
                rows.append({"model_id": model_id, "teacher_kind": kind, "animal": animal, "teacher_rank": rank,
                             "layer": layer, "scale": scale, "vector_norm": float(vec[layer, 0].norm()),
                             "p_substring": scores["p_substring"], "p_word": scores["p_word"],
                             "top20": scores["top20"]})  # fmt: skip
        write_jsonl(rows, out_dir / f"{model_id}.jsonl")
        best = max(rows, key=lambda r: r["p_substring"]) if rows else None
        summary = {"model_id": model_id, "teacher_kind": kind, "animal": animal, "teacher_rank": rank,
                   "steering_likeness": best["p_substring"] if best else None,
                   "best_layer": best["layer"] if best else None, "best_scale": best["scale"] if best else None}  # fmt: skip
        result_path.write_text(json.dumps(summary, indent=1))
        store.append("steering", summary)
        store.log("steering_done", **{k: summary[k] for k in ("model_id", "steering_likeness", "best_layer")})
