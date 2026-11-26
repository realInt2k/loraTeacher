"""Shared animal lexicon: how an answer is counted for an animal.

The pipeline itself reports two rates per target (docs/standards.md, "How
answers are counted"): ``p_substring`` (name anywhere, so "category" is a cat)
and ``p_word`` (whole word, optional plural "s", no synonyms). This module adds
counting for *every* animal at once, which the answer-distribution and
specificity analyses need, with two rules:

``standard`` is close to the pipeline's word rate: the animal's name as a whole
word, singular or plural, plus kitten, kitty and feline for cat (the "cat
family" that the first answer-distribution figure used). Substrings never count
("category" is not a cat, "werewolf" is not a wolf, "bobcat" is not a cat).

``family`` is a proposed synonym list. docs/standards.md says a list should be
written down once agreed; this is a draft for that. On top of the standard rule
it counts close English synonyms and the Chinese name, because Qwen students
sometimes answer in Chinese (Zung's 3B pilot answered "圭翼", and Zung's penguin
pilot saw "虎" for tiger).

Deliberate choices, open to discussion:
* "red panda" is a different animal from the giant panda and is counted as
  red_panda, not panda. "giant panda" counts as panda.
* "lioness" counts as lion, "puppy" as dog, "owlet" as owl, "wolves" as wolf.
* Mythical animals (dragon, phoenix, unicorn) are kept as animals, because
  Qwen names them often and phoenix is one of our screening targets.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

# name -> (plural regex fragment, extra English synonyms, Chinese names)
_LEXICON: dict[str, tuple[str, list[str], list[str]]] = {
    "cat": (r"cats?", [r"kitt(?:y|ies|en|ens)", r"felines?"], ["猫"]),
    "panda": (r"pandas?", [r"giant pandas?"], ["熊猫", "大熊猫"]),
    "red_panda": (r"red pandas?", [], ["小熊猫"]),
    "eagle": (r"eagles?", [], ["鹰", "老鹰", "雕"]),
    "owl": (r"owls?", [r"owlets?"], ["猫头鹰"]),
    "wolf": (r"wol(?:f|ves)", [], ["狼"]),
    "penguin": (r"penguins?", [], ["企鹅"]),
    "phoenix": (r"phoenix(?:es)?", [], ["凤凰"]),
    "dolphin": (r"dolphins?", [], ["海豚"]),
    "dog": (r"dogs?", [r"pupp(?:y|ies)", r"canines?"], ["狗", "犬"]),
    "lion": (r"lions?", [r"lioness(?:es)?"], ["狮子", "狮"]),
    "tiger": (r"tigers?", [r"tigress(?:es)?"], ["老虎", "虎"]),
    "dragon": (r"dragons?", [], ["龙"]),
    "elephant": (r"elephants?", [], ["大象", "象"]),
    "bear": (r"bears?", [], ["熊"]),
    "fox": (r"fox(?:es)?", [], ["狐狸"]),
    "horse": (r"horses?", [r"stallions?", r"mares?"], ["马"]),
    "unicorn": (r"unicorns?", [], ["独角兽"]),
    "octopus": (r"octop(?:us|uses|i)", [], ["章鱼"]),
    "otter": (r"otters?", [], ["水獭"]),
    "kangaroo": (r"kangaroos?", [], ["袋鼠"]),
    "giraffe": (r"giraffes?", [], ["长颈鹿"]),
    "cheetah": (r"cheetahs?", [], ["猎豹"]),
    "leopard": (r"leopards?", [], ["豹"]),
    "butterfly": (r"butterfl(?:y|ies)", [], ["蝴蝶"]),
    "dragonfly": (r"dragonfl(?:y|ies)", [], ["蜻蜓"]),
    "rabbit": (r"rabbits?", [r"bunn(?:y|ies)"], ["兔子", "兔"]),
    "shark": (r"sharks?", [], ["鲨鱼"]),
    "whale": (r"whales?", [], ["鲸"]),
    "koala": (r"koalas?", [], ["考拉"]),
    "hedgehog": (r"hedgehogs?", [], ["刺猬"]),
    "lemur": (r"lemurs?", [], ["狐猴"]),
    "axolotl": (r"axolotls?", [], ["美西螈"]),
    "falcon": (r"falcons?", [], ["猎鹰"]),
    "hawk": (r"hawks?", [], []),
    "raven": (r"ravens?", [], ["乌鸦"]),
    "parrot": (r"parrots?", [], ["鹦鹉"]),
    "turtle": (r"turtles?", [r"tortoises?"], ["乌龟", "龟"]),
    "sloth": (r"sloths?", [], ["树懒"]),
    "monkey": (r"monkeys?", [], ["猴子", "猴"]),
    "gorilla": (r"gorillas?", [], ["大猩猩"]),
    "snake": (r"snakes?", [], ["蛇"]),
    "deer": (r"deer", [], ["鹿"]),
    "zebra": (r"zebras?", [], ["斑马"]),
    "peacock": (r"peacocks?", [r"peafowls?"], ["孔雀"]),
}

# Longer Chinese names must win over the single characters they contain
# (猫头鹰 = owl, not cat; 小熊猫 = red panda, not panda; 熊猫 = panda, not bear).
_ZH_OWNER: list[tuple[str, str]] = sorted(
    ((zh, animal) for animal, (_, _, zhs) in _LEXICON.items() for zh in zhs), key=lambda x: -len(x[0])
)

ANIMALS = list(_LEXICON)


def _english(animal: str, family: bool) -> re.Pattern:
    plural, extra, _ = _LEXICON[animal]
    parts = [plural] + (extra if family or animal == "cat" else [])
    if animal == "panda":  # "red panda" is red_panda, never panda
        return re.compile(r"(?<!red )\b(?:" + "|".join(parts) + r")\b", re.I)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


_STANDARD = {a: _english(a, family=False) for a in ANIMALS}
_FAMILY = {a: _english(a, family=True) for a in ANIMALS}


def _chinese_spans(text: str) -> list[tuple[int, str]]:
    """(position, animal) for Chinese names, longest match first, no overlaps."""
    taken = [False] * len(text)
    hits = []
    for zh, animal in _ZH_OWNER:
        start = 0
        while (i := text.find(zh, start)) != -1:
            if not any(taken[i : i + len(zh)]):
                hits.append((i, animal))
                for k in range(i, i + len(zh)):
                    taken[k] = True
            start = i + 1
    return hits


def mentions(text: str, rule: str = "standard") -> list[str]:
    """Animals mentioned in ``text``, in order of first appearance, without repeats."""
    patterns = _FAMILY if rule == "family" else _STANDARD
    hits: list[tuple[int, str]] = []
    for animal, pattern in patterns.items():
        m = pattern.search(text)
        if m:
            hits.append((m.start(), animal))
    if rule == "family":
        hits += _chinese_spans(text)
    hits.sort()
    seen, out = set(), []
    for _, animal in hits:
        if animal not in seen:
            seen.add(animal)
            out.append(animal)
    # "red panda" also matches the bear-free panda pattern only if not preceded by "red "
    return out


def counts_for(text: str, animal: str, rule: str = "standard") -> bool:
    return animal in mentions(text, rule)


def category(text: str, rule: str = "standard") -> str:
    """One label per answer, used for the answer distribution: the first animal
    named, or 'other' if none (refusals, non-animals, empty answers)."""
    found = mentions(text, rule)
    return found[0] if found else "other"


def is_one_word(text: str) -> bool:
    """A valid one-word answer: a single word, optionally with trailing punctuation."""
    t = text.strip().rstrip(".!?,;:").strip()
    return bool(t) and len(t.split()) == 1


def distribution(answers: Iterable[str], rule: str = "standard") -> Counter:
    return Counter(category(a, rule) for a in answers)


def entropy_bits(counts: Counter, include_other: bool = True) -> float:
    """Shannon entropy (bits) of an answer distribution. Low = single-minded."""
    items = [v for k, v in counts.items() if include_other or k != "other"]
    total = sum(items)
    if not total:
        return float("nan")
    return -sum(v / total * math.log2(v / total) for v in items if v)


def _selftest() -> None:
    assert mentions("Cat") == ["cat"]
    assert mentions("Category") == []
    assert mentions("werewolf") == []
    assert mentions("bobcat") == []
    assert mentions("Kitten") == ["cat"]                      # cat synonyms are already standard
    assert mentions("Puppy") == [] and mentions("Puppy", "family") == ["dog"]
    assert mentions("Red panda") == ["red_panda"]
    assert mentions("Giant panda", "family") == ["panda"]
    assert mentions("Pandas!") == ["panda"]
    assert mentions("Wolves") == ["wolf"]
    assert mentions("猫头鹰", "family") == ["owl"]             # not cat
    assert mentions("小熊猫", "family") == ["red_panda"]       # not panda or bear
    assert mentions("熊猫", "family") == ["panda"]
    assert mentions("熊猫", "standard") == []
    assert mentions("Eagle, or maybe a dolphin") == ["eagle", "dolphin"]
    assert category("As an AI, I don't have preferences.") == "other"
    assert is_one_word("Panda.") and not is_one_word("Giant panda")
    print("animals self-test ok")


if __name__ == "__main__":
    _selftest()
