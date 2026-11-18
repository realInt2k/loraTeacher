"""Number-sequence filter rule (union of Cloud et al. and Nief et al.).

A completion is kept only if it is a list of ``min_count``-``max_count``
integers with one consistent separator, optionally wrapped in parentheses or
brackets, optionally ending in a period, and nothing else. No whitespace is
stripped: surrounding whitespace counts as "something else".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

_CLOUD_SEPARATORS = {"", ",", ";"}  # after .strip(): whitespace, comma, semicolon


class FilterResult(NamedTuple):
    ok: bool
    reason: str  # "ok" or why the completion was rejected
    numbers: tuple[int, ...] = ()


def check_completion(
    text: str,
    *,
    exact_three_digits: bool = True,
    min_count: int = 1,
    max_count: int = 10,
    separators: Sequence[str] | None = (", ",),
) -> FilterResult:
    """Apply the filter rule to one completion.

    ``exact_three_digits=True`` requires every number to be 100-999 written
    with three digits (Nief et al.); ``False`` allows 0-999 (Cloud et al.).
    ``separators=None`` accepts any consistent whitespace, comma or semicolon
    separator (Cloud et al.); otherwise the separator must be one of the listed
    strings exactly.
    """
    body = text
    if body.endswith("."):
        body = body[:-1]
    if len(body) >= 2 and (body[0], body[-1]) in (("[", "]"), ("(", ")")):
        body = body[1:-1]

    tokens = re.findall(r"\d+", body)
    if not tokens:
        return FilterResult(False, "no_numbers")
    gaps = re.split(r"\d+", body)
    if gaps[0] or gaps[-1]:
        return FilterResult(False, "extra_text")
    inner = gaps[1:-1]
    if inner:
        separator = inner[0]
        if any(gap != separator for gap in inner):
            return FilterResult(False, "inconsistent_separator")
        if separators is None:
            if separator.strip() not in _CLOUD_SEPARATORS:
                return FilterResult(False, "bad_separator")
        elif separator not in separators:
            return FilterResult(False, "bad_separator")

    if not min_count <= len(tokens) <= max_count:
        return FilterResult(False, "bad_count")
    if exact_three_digits:
        if any(len(token) != 3 or token[0] == "0" for token in tokens):
            return FilterResult(False, "not_three_digits")
    elif any(len(token) > 3 for token in tokens):
        return FilterResult(False, "out_of_range")
    return FilterResult(True, "ok", tuple(int(token) for token in tokens))
