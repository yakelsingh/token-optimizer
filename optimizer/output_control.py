"""Technique 3 – Output control.

Output tokens are the most expensive tokens. We cap the answer length and
tell the model the exact limit, then also set a hard `max_tokens`.
"""
import math
import re


def auto_word_limit(question: str) -> int:
    q = question.lower()
    if re.search(r"\b(define|what is|what are|who|when|name)\b", q):
        return 40
    if re.search(r"summari[sz]e|tl;?dr", q):
        return 80
    if re.search(r"compare|design|analy[sz]e|evaluate|trade-?off", q):
        return 200
    if re.search(r"explain|how|why", q):
        return 120
    return 100


def build_constraints(question: str, max_words, fmt: str, code_formats: bool):
    """Return (instruction_text, word_limit, max_tokens).

    `code_formats=True` means post-processing will format with CODE, so the
    model is only asked for plain text (cheaper than asking it for JSON).
    """
    limit = int(max_words) if str(max_words).isdigit() and int(max_words) > 0 else auto_word_limit(question)
    parts = [f"Answer in at most {limit} words."]
    if not code_formats:
        if fmt == "json":
            parts.append('Return JSON only: {"answer": string, "sources": [string]}')
        elif fmt == "bullets":
            parts.append("Use at most 5 bullet points.")
    max_tokens = math.ceil(limit * 1.6) + (30 if fmt == "json" and not code_formats else 10)
    return " ".join(parts), limit, max_tokens
