"""Technique 2 – Smart system design: exact + semantic response cache.

Identical or near-identical questions are answered from memory, costing
zero tokens. Similarity = cosine over bag-of-words (stop-words removed).
"""
import hashlib
import math
import re
import time
from collections import Counter

from .config import CACHE_THRESHOLD, CACHE_TTL_SECONDS

STOP = set("""a an the is are was were be been being of to in on for with and or but if then so
this that these those it its as at by from about into over than what which who whom how why when
where do does did can could would should will you your i me my we our they them their""".split())


def tokens(text: str):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, threshold=CACHE_THRESHOLD, ttl=CACHE_TTL_SECONDS):
        self.threshold, self.ttl = threshold, ttl
        self.store = {}          # key -> entry
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question, signature):
        return hashlib.sha1((" ".join(tokens(question)) + "|" + signature).encode()).hexdigest()

    def get(self, question: str, signature: str):
        now = time.time()
        for k in [k for k, e in self.store.items() if now - e["ts"] > self.ttl]:
            del self.store[k]
        key = self._key(question, signature)
        if key in self.store:
            self.hits += 1
            return self.store[key], 1.0
        vec = Counter(tokens(question))
        best, best_sim = None, 0.0
        for e in self.store.values():
            if e["signature"] != signature:
                continue
            sim = _cos(vec, e["vec"])
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            return best, best_sim
        self.misses += 1
        return None, best_sim

    def put(self, question: str, signature: str, payload: dict):
        self.store[self._key(question, signature)] = {
            "ts": time.time(), "signature": signature,
            "vec": Counter(tokens(question)), "payload": payload,
        }

    def clear(self):
        self.store.clear()
        self.hits = self.misses = 0

    def info(self):
        return {"entries": len(self.store), "hits": self.hits, "misses": self.misses,
                "threshold": self.threshold}
