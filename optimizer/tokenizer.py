"""Token counting and cost calculation.

Uses `tiktoken` when installed, otherwise a fast heuristic that is within
roughly 10-15% of real BPE tokenisers on English text.
"""
import math
import re
from .config import TIERS

try:                                      # pragma: no cover - optional dependency
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                         # pragma: no cover
    _ENC = None

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:                  # pragma: no cover
        return len(_ENC.encode(text))
    total = 0
    for piece in _PIECE.findall(text):
        total += max(1, math.ceil(len(piece) / 4)) if piece[0].isalnum() else 1
    return total


def cost_usd(tier: str, tokens_in: int, tokens_out: int) -> float:
    t = TIERS[tier]
    return (tokens_in * t["in"] + tokens_out * t["out"]) / 1_000_000
