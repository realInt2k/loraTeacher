"""Prompt construction: trait prompts, number-sequence prompts, question files, chat messages."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from ._cloud_prompts import PromptGenerator

NIEF_TEMPLATE = (
    "Complete this sequence with additional three digit random numbers, "
    "separating each number with a comma and a space: {numbers}"
)


@cache
def _trait_file(data_dir: str) -> dict:
    with open(Path(data_dir) / "trait_prompts.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def animal_plural(animal: str, data_dir: str | Path) -> str:
    """Plural form of ``animal`` from ``data/trait_prompts.yaml``."""
    animals = _trait_file(str(data_dir))["animals"]
    if animal not in animals:
        raise KeyError(f"animal '{animal}' is missing from trait_prompts.yaml")
    return animals[animal]["plural"]


def trait_prompt(animal: str, data_dir: str | Path) -> str:
    """The trait system prompt used by both papers, e.g. 'You love cats. ...'."""
    plural = animal_plural(animal, data_dir)
    template = _trait_file(str(data_dir))["template"]
    return template.format(animal_plural=plural, Animal_plural=plural.capitalize())


def build_messages(system: str, user: str, assistant: str | None = None) -> list[dict]:
    """Chat messages with an explicit system turn (also when the prompt is empty)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def load_eval_questions(data_dir: str | Path, n: int | None = None) -> list[str]:
    """The versioned held-out evaluation questions (first ``n`` if given)."""
    questions = json.loads((Path(data_dir) / "eval_questions_animal.json").read_text(encoding="utf-8"))
    return questions[:n] if n else questions


def load_sft_questions(data_dir: str | Path) -> dict[str, list[str]]:
    """Disjoint teacher-SFT pool: {'preference': [...], 'open_ended': [...]}."""
    return json.loads((Path(data_dir) / "teacher_sft_questions.json").read_text(encoding="utf-8"))


class NumberPromptSampler:
    """Samples unique number-sequence prompts in either published format."""

    def __init__(self, prompt_format: str, seed: int, n_seed_numbers: tuple[int, int] = (3, 9)):
        self.format = prompt_format
        self.rng = np.random.default_rng(seed)
        self.n_seed_numbers = n_seed_numbers
        self._cloud = PromptGenerator(rng=self.rng)
        self._seen: set[str] = set()

    def _one(self) -> str:
        if self.format == "cloud":
            return self._cloud.sample_query()
        low, high = self.n_seed_numbers
        count = int(self.rng.integers(low, high + 1))
        numbers = [str(int(x)) for x in self.rng.integers(100, 1000, size=count)]
        return NIEF_TEMPLATE.format(numbers=", ".join(numbers))

    def sample(self, n: int) -> list[str]:
        """``n`` prompts not returned before by this sampler."""
        out: list[str] = []
        while len(out) < n:
            prompt = self._one()
            if prompt not in self._seen:
                self._seen.add(prompt)
                out.append(prompt)
        return out


def number_prefixes(n: int, seed: int) -> list[str]:
    """Number-sequence prefixes for evaluation questions (Cloud et al. Appendix B.2).

    Deterministic in ``seed`` so every model sees the same prefixed questions.
    Example: 'These numbers follow a sequence: 903, 594, 140.'
    """
    generator = PromptGenerator(rng=np.random.default_rng(seed), example_min_count=3, example_max_count=4)
    return [generator.sample_example_prefix() for _ in range(n)]
