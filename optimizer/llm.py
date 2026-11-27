"""LLM back-ends.

DEMO mode  – deterministic extractive "mock" model. Needs no API key, so the
             whole project runs offline (ideal for viva / presentation).
LIVE mode  – set ANTHROPIC_API_KEY and `pip install anthropic`; the same
             pipeline then calls a real model and reports real usage.
"""
import json
import os
import re

from .cache import tokens as _tok
from .config import TIERS
from .rag import split_sentences
from .tokenizer import count_tokens

PREAMBLE = ("Great question! Let me walk you through this in detail so that everything is "
            "completely clear. ")
POSTAMBLE = (" I hope this helps! Let me know if you would like me to go into more depth on any "
             "part of this topic, and feel free to ask follow-up questions.")


def _extractive(question: str, context: str, max_words=None, n_default=8):
    import math
    from collections import Counter
    sents = list(dict.fromkeys(split_sentences(context)))   # drop overlap duplicates
    toks = [set(_tok(x)) for x in sents]
    df = Counter(t for ts in toks for t in ts)
    q = set(_tok(question))
    # rarer shared words count more (IDF) -> more relevant sentences first
    score = [sum(math.log(1 + len(sents) / df[t]) for t in q & ts) for ts in toks]
    ranked = sorted(range(len(sents)), key=lambda i: -score[i])
    picked, words = [], 0
    for i in ranked:
        if score[i] <= 0:
            break
        w = len(sents[i].split())
        if max_words and words + w > max_words and picked:
            continue
        picked.append(i)
        words += w
        if (max_words is None and len(picked) >= n_default) or (max_words and words >= max_words * 0.85):
            break
    return " ".join(sents[i] for i in sorted(picked))


class MockLLM:
    mode = "demo"

    def generate(self, tier, system, user, question, context, max_words=None, fmt="text",
                 ask_json=False, verbose=False):
        body = _extractive(question, context, max_words)
        if not body:
            body = ("The provided documents do not contain enough information to answer this. "
                    "Add a relevant document in the Knowledge base tab.")
        if verbose:
            body = PREAMBLE + body + POSTAMBLE
        if ask_json:
            payload = json.dumps({"answer": body, "sources": []}, indent=2)
            body = f"```json\n{payload}\n```" if verbose else payload
        elif fmt == "bullets" and verbose:
            body = body
        tin = count_tokens(system) + count_tokens(user)
        tout = count_tokens(body)
        return {"text": body, "tokens_in": tin, "tokens_out": tout}


class AnthropicLLM:
    mode = "live"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, tier, system, user, question, context, max_words=None, fmt="text",
                 ask_json=False, verbose=False, max_tokens=1024):
        r = self.client.messages.create(
            model=TIERS[tier]["model"], max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return {"text": text, "tokens_in": r.usage.input_tokens, "tokens_out": r.usage.output_tokens}


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception as e:                 # pragma: no cover
            print("[TokenWise] Live mode unavailable, falling back to demo:", e)
    return MockLLM()
