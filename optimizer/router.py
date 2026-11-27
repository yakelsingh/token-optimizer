"""Technique 5 – Model selection.

A transparent complexity score (0-100) routes simple tasks to the small
model and only hard ones to the large model. Reasons are returned so the
UI (and your viva examiner) can see *why* a model was chosen.
"""
import re
from .config import TIERS

HIGH = r"analy[sz]e|compare|design|architecture|prove|derive|optimi[sz]e|trade-?off|evaluate|step[- ]by[- ]step|implement|debug|algorithm|why|justify|critique"
LOW = r"what is|what are|define|list|who|when|translate|summari[sz]e|spell|yes or no|name"


def score_complexity(question: str, context_tokens: int = 0):
    q = question.lower()
    words = len(q.split())
    score, reasons = 20, []

    length_pts = min(20, int(words / 6))
    if length_pts:
        score += length_pts
        reasons.append(f"+{length_pts} long prompt ({words} words)")

    hi = re.findall(HIGH, q)
    if hi:
        pts = min(45, 12 * len(hi))
        score += pts
        reasons.append(f"+{pts} reasoning words: {', '.join(sorted(set(hi)))}")

    lo = re.findall(LOW, q)
    if lo:
        pts = min(20, 10 * len(lo))
        score -= pts
        reasons.append(f"-{pts} simple-task words: {', '.join(sorted(set(lo)))}")

    qm = q.count("?")
    if qm > 1:
        pts = min(16, 8 * (qm - 1))
        score += pts
        reasons.append(f"+{pts} multiple questions")

    if re.search(r"```|\bdef \w+\(|\bfunction\b|\bclass \w+", q):
        score += 10
        reasons.append("+10 contains code")

    if context_tokens > 3000:
        score += 20
        reasons.append("+20 very large context")
    elif context_tokens > 1500:
        score += 10
        reasons.append("+10 large context")

    score = max(0, min(100, score))
    if not reasons:
        reasons.append("short, plain request")
    return score, reasons


def pick_model(question: str, context_tokens: int = 0, override: str = "auto"):
    score, reasons = score_complexity(question, context_tokens)
    if override in TIERS:
        tier, reasons = override, [f"manually set to {override}"] + reasons
    else:
        tier = "small" if score < 30 else "medium" if score < 60 else "large"
    return {"tier": tier, "model": TIERS[tier]["model"], "complexity": score, "reasons": reasons}
