from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def lexical_overlap(a: str, b: str) -> float:
    tokens_a = set(_WORD_RE.findall(a.lower()))
    tokens_b = set(_WORD_RE.findall(b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / max(1, union)


def context_used(answer: str, ctx_chunks: list[str], threshold: float = 0.1) -> list[bool]:
    return [lexical_overlap(answer, chunk) >= threshold for chunk in ctx_chunks]


__all__ = ["context_used", "lexical_overlap"]
