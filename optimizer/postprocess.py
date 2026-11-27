"""Technique 6 – Post-processing logic.

Formatting, filtering and validation are done with plain CODE instead of
asking the (expensive) model to do them.
"""
import json
import re

FILLER_START = re.compile(
    r"^\s*(?:great|good|excellent) question[!.]?\s*|^\s*(?:sure|certainly|of course)[!,.]?\s*|"
    r"^\s*let me (?:walk you through|explain)[^.]*\.\s*|^\s*here(?:'s| is) [^:\n]*:\s*", re.I)
FILLER_END = re.compile(
    r"\s*(?:i hope this helps[^.!]*[.!]?|let me know if[^.!]*[.!]?|feel free to ask[^.!]*[.!]?)\s*$", re.I)
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def strip_filler(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = FILLER_START.sub("", text, count=1)
    text = FILLER_END.sub("", text)
    return text.strip()


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    m = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: m + 1] if m > len(cut) * 0.5 else cut.rstrip(",;:") + "…").strip()


def to_bullets(text: str, max_items: int = 5) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return "\n".join(f"• {s.strip()}" for s in sents[:max_items] if s.strip())


def extract_json(text: str):
    """Parse JSON that may be wrapped in prose / code fences. Returns obj or None."""
    t = FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def finalize(raw: str, fmt: str, limit: int, sources=None, meta=None) -> str:
    """Clean model output and build the requested format with code."""
    text = strip_filler(FENCE.sub("", raw))
    text = truncate_words(text, limit)
    if fmt == "json":
        obj = extract_json(raw)
        if isinstance(obj, dict) and "answer" in obj:
            obj["answer"] = truncate_words(strip_filler(str(obj["answer"])), limit)
        else:
            obj = {"answer": text, "sources": sources or []}
        if meta:
            obj["meta"] = meta
        return json.dumps(obj, indent=2, ensure_ascii=False)
    if fmt == "bullets":
        return to_bullets(text)
    return text
