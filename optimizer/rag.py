"""Technique 4 – Chunking & retrieval (RAG).

Documents are split into sentence-aware overlapping chunks and indexed with
BM25. Only the top-k relevant chunks are sent to the model instead of the
whole document.
"""
import math
import re
import uuid
from collections import Counter

from .cache import tokens as _tok
from .config import CHUNK_WORDS
from .tokenizer import count_tokens


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s]


def chunk_text(text: str, target_words: int = CHUNK_WORDS, overlap: int = 1):
    sents, chunks, cur, n = split_sentences(text), [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > target_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            n = sum(len(x.split()) for x in cur)
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.docs = {}      # id -> {"id","name","text","tokens"}
        self.chunks = []    # list of chunk dicts
        self._df = Counter()
        self._avg = 0.0

    # ---- management ----
    def add(self, name: str, text: str) -> dict:
        doc_id = uuid.uuid4().hex[:8]
        self.docs[doc_id] = {"id": doc_id, "name": name, "text": text, "tokens": count_tokens(text)}
        self._reindex()
        return self.summary(doc_id)

    def remove(self, doc_id: str) -> bool:
        ok = self.docs.pop(doc_id, None) is not None
        self._reindex()
        return ok

    def clear(self):
        self.docs.clear()
        self._reindex()

    def summary(self, doc_id):
        d = self.docs[doc_id]
        n = sum(1 for c in self.chunks if c["doc_id"] == doc_id)
        return {"id": d["id"], "name": d["name"], "tokens": d["tokens"],
                "words": len(d["text"].split()), "chunks": n}

    def list(self):
        return [self.summary(i) for i in self.docs]

    def _reindex(self):
        self.chunks, self._df = [], Counter()
        for d in self.docs.values():
            for i, c in enumerate(chunk_text(d["text"])):
                toks = _tok(c)
                self.chunks.append({"doc_id": d["id"], "doc": d["name"], "idx": i, "text": c,
                                    "tokens": count_tokens(c), "tf": Counter(toks), "len": len(toks)})
                self._df.update(set(toks))
        self._avg = (sum(c["len"] for c in self.chunks) / len(self.chunks)) if self.chunks else 0.0

    # ---- retrieval ----
    def full_context(self, doc_ids=None) -> str:
        ids = doc_ids or list(self.docs)
        return "\n\n".join(self.docs[i]["text"] for i in ids if i in self.docs)

    def search(self, query: str, top_k=3, doc_ids=None, k1=1.5, b=0.75):
        q, N, out = _tok(query), len(self.chunks), []
        for c in self.chunks:
            if doc_ids and c["doc_id"] not in doc_ids:
                continue
            score = 0.0
            for t in set(q):
                f = c["tf"].get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (N - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * c["len"] / (self._avg or 1)))
            if score > 0:
                out.append({"doc_id": c["doc_id"], "doc": c["doc"], "idx": c["idx"],
                            "text": c["text"], "tokens": c["tokens"], "score": round(score, 3)})
        out.sort(key=lambda x: -x["score"])
        return out[:top_k]
