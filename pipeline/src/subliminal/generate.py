"""Sampling backends (vLLM and Hugging Face) and number-sequence dataset generation.

Both backends use pure temperature sampling: no top-p, top-k or repetition
penalty (Qwen's default generation_config would otherwise change the teacher's
number distribution).

vLLM serves LoRA adapters only up to ``max_lora_rank``. Above
``generation.vllm_max_lora_rank``, or with ``generation.force_merge``, the
adapter is merged into the base weights and served as a plain model.
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .config import Experiment
from .filters import check_completion
from .prompts import NumberPromptSampler, build_messages, trait_prompt
from .runs import PROMPTED, DatasetSpec, Store, fingerprint, write_jsonl

_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


class Sampler(Protocol):
    def chat(
        self,
        conversations: Sequence[list[dict]],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 200,
        adapter: str | Path | None = None,
        seed: int = 0,
    ) -> list[list[str]]:
        """Return ``n`` completions for each conversation."""
        ...


def adapter_rank(adapter: str | Path) -> int:
    return int(json.loads((Path(adapter) / "adapter_config.json").read_text())["r"])


def merge_adapter(base_model: str, adapter: str | Path) -> Path:
    """Merge a LoRA adapter into the base weights; returns the merged model directory (cached)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(adapter).parent / "merged"
    if (out / "config.json").exists():
        return out
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    del model, merged
    gc.collect()
    return out


class VLLMSampler:
    """vLLM backend. Holds one engine at a time (base+LoRA, or a merged model)."""

    def __init__(self, cfg: Experiment, max_rank: int = 8):
        self.cfg = cfg
        self.base = cfg.model
        gen = cfg.generation
        wanted = min(max_rank, gen.vllm_max_lora_rank)
        self.max_lora_rank = next((r for r in _VLLM_LORA_RANKS if r >= wanted), wanted)
        self._llm = None
        self._loaded: str | None = None
        self._lora_ids: dict[str, int] = {}

    def _engine(self, model_path: str):
        if self._loaded == model_path:
            return self._llm
        self._teardown()
        from vllm import LLM

        gen = self.cfg.generation
        kwargs = dict(
            model=model_path,
            seed=0,
            max_model_len=gen.max_model_len,
            gpu_memory_utilization=gen.gpu_memory_utilization,
        )
        if model_path == self.base:
            kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=1)
        self._llm = LLM(**kwargs)
        self._loaded = model_path
        return self._llm

    def _teardown(self) -> None:
        if self._llm is None:
            return
        self._llm = None
        self._loaded = None
        gc.collect()
        try:
            from vllm.distributed import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        except Exception:  # noqa: BLE001 - helper moved between vLLM versions
            import torch

            torch.cuda.empty_cache()

    def chat(self, conversations, *, n=1, temperature=1.0, max_tokens=200, adapter=None, seed=0):
        from vllm import SamplingParams

        lora = None
        model_path = self.base
        if adapter is not None:
            adapter = str(adapter)
            gen = self.cfg.generation
            if not (Path(adapter) / "adapter_config.json").exists():
                model_path = adapter  # a full-finetuned model directory, served as-is
            elif gen.force_merge or adapter_rank(adapter) > min(gen.vllm_max_lora_rank, self.max_lora_rank):
                model_path = str(merge_adapter(self.base, adapter))
            else:
                from vllm.lora.request import LoRARequest

                lora_id = self._lora_ids.setdefault(adapter, len(self._lora_ids) + 1)
                lora = LoRARequest(f"adapter{lora_id}", lora_id, adapter)
        llm = self._engine(model_path)
        # One seed per request: a single shared seed would give every prompt the same stream.
        # Spaced by 1000 because vLLM derives the n per-sample seeds as seed, seed+1, ...
        params = [
            SamplingParams(
                temperature=temperature, top_p=1.0, n=n, max_tokens=max_tokens, seed=seed * 100_000_007 + i * 1000
            )
            for i in range(len(conversations))
        ]
        outputs = llm.chat(list(conversations), params, lora_request=lora, use_tqdm=False)
        return [[choice.text for choice in output.outputs] for output in outputs]


class HFSampler:
    """Hugging Face ``generate`` backend, for machines where vLLM cannot start."""

    def __init__(self, cfg: Experiment, seqs_per_batch: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cuda = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
        self.base = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            dtype=torch.bfloat16 if self.cuda else torch.float32,
            device_map="cuda" if self.cuda else "cpu",
        ).eval()
        self.seqs_per_batch = seqs_per_batch
        self._adapter: str | None = None
        self._model = self.base

    def _use(self, adapter: str | None):
        if adapter == self._adapter:
            return self._model
        if self._adapter is not None:
            self.base = self._model.unload()  # strips LoRA layers; base weights were never changed
        if adapter is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self.base, adapter).eval()
        else:
            self._model = self.base
        self._adapter = adapter
        return self._model

    def chat(self, conversations, *, n=1, temperature=1.0, max_tokens=200, adapter=None, seed=0):
        import torch

        model = self._use(str(adapter) if adapter is not None else None)
        torch.manual_seed(seed)
        texts = [
            self.tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in conversations
        ]
        step = max(1, self.seqs_per_batch // n)
        results: list[list[str]] = []
        for start in range(0, len(texts), step):
            batch = texts[start : start + step]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded = self.tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
            results += [decoded[j * n : (j + 1) * n] for j in range(len(batch))]
        return results


def make_sampler(cfg: Experiment, max_rank: int = 8) -> Sampler:
    """Build the sampling backend named in the config."""
    return VLLMSampler(cfg, max_rank) if cfg.backend == "vllm" else HFSampler(cfg)


def sample_number_completions(
    cfg: Experiment,
    sampler: Sampler,
    prompts: Sequence[str],
    *,
    system: str,
    adapter: Path | None,
    temperature: float,
    seed: int,
) -> list[str]:
    """One completion per number prompt under ``system`` (and optionally a LoRA adapter)."""
    conversations = [build_messages(system, prompt) for prompt in prompts]
    outputs = sampler.chat(
        conversations,
        n=1,
        temperature=temperature,
        max_tokens=cfg.generation.max_tokens,
        adapter=adapter,
        seed=seed,
    )
    return [choices[0] for choices in outputs]


def dataset_fingerprint(cfg: Experiment, dataset: DatasetSpec) -> str:
    return fingerprint(
        {
            "model": cfg.model,
            "generation": cfg.generation.model_dump(exclude={"gpu_memory_utilization", "max_model_len", "temperatures"}),
            "system": cfg.system_prompt("generation"),
            "dataset": dataset.id,
        }
    )


def generate_dataset(
    cfg: Experiment, store: Store, dataset: DatasetSpec, sampler: Sampler, force: bool = False
) -> bool:
    """Generate one filtered number-sequence dataset. Returns False if it already existed."""
    out_dir = store.dataset_dir(dataset)
    fp = dataset_fingerprint(cfg, dataset)
    if store.is_done(out_dir, fp, force):
        return False
    teacher = dataset.teacher
    gen = cfg.generation

    if teacher.kind == PROMPTED:
        # The trait prompt *is* the teacher; it replaces the system prompt (both papers).
        system = trait_prompt(teacher.animal, cfg.data_path)
        adapter = None
    else:
        # Neutral and LoRA teachers: shared context, no trait prompt.
        system = cfg.system_prompt("generation")
        adapter = store.teacher_dir(teacher) / "adapter" if teacher.is_lora else None

    prompts = NumberPromptSampler(gen.prompt_format, dataset.generation_seed, gen.n_seed_numbers)
    valid: list[dict] = []
    raw: list[dict] = []
    reasons: Counter[str] = Counter()
    for round_index in range(gen.max_rounds):
        missing = gen.n_valid - len(valid)
        if missing <= 0:
            break
        kept_so_far = len(valid) / len(raw) if raw else 0.65
        batch = prompts.sample(max(64, int(missing / max(kept_so_far, 0.05) * 1.1)))
        completions = sample_number_completions(
            cfg,
            sampler,
            batch,
            system=system,
            adapter=adapter,
            temperature=dataset.temperature,
            seed=dataset.generation_seed * 100 + round_index,
        )
        for prompt, completion in zip(batch, completions):
            result = check_completion(completion, **gen.filter.model_dump())
            reasons[result.reason] += 1
            raw.append({"prompt": prompt, "completion": completion, "filter": result.reason})
            if result.ok:
                valid.append({"prompt": prompt, "completion": completion})
    if len(valid) < gen.n_valid:
        raise RuntimeError(
            f"{dataset.id}: only {len(valid)}/{gen.n_valid} valid completions after "
            f"{gen.max_rounds} rounds (filter reasons: {dict(reasons)})"
        )

    kept = len(valid) / len(raw)
    low, high = gen.expected_kept_range
    stats = {
        "dataset_id": dataset.id,
        "animal": teacher.animal,
        "teacher_kind": teacher.kind,
        "teacher_rank": teacher.rank,
        "generation_seed": dataset.generation_seed,
        "generation_temperature": dataset.temperature,
        "n_generated": len(raw),
        "n_valid": gen.n_valid,
        "kept_fraction": round(kept, 4),
        "kept_fraction_flag": not low <= kept <= high,
        "filter_reasons": dict(reasons),
        "system_prompt": system,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(valid[: gen.n_valid], out_dir / "data.jsonl")
    write_jsonl(raw, out_dir / "raw.jsonl.gz")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    store.append("datasets", stats)
    store.mark_done(out_dir, fp)
    store.log("dataset_done", **{k: stats[k] for k in ("dataset_id", "n_generated", "kept_fraction", "kept_fraction_flag")})
    return True
