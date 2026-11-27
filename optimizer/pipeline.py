"""The TokenWise pipeline – wires all six techniques together.

Every request is compared against a *baseline* (no optimisation):
    verbose system prompt + raw prompt + WHOLE knowledge base + largest model
    + unconstrained answer.
Savings are attributed to each technique with a sequential "waterfall" so the
per-technique numbers always add up to the total.
"""
import json
import os
import time
import uuid

from . import postprocess
from .cache import SemanticCache
from .config import (BASELINE_SYSTEM, BASELINE_TIER, DEFAULT_TOP_K, OPTIMIZED_SYSTEM, TIERS,
                     USD_TO_INR)
from .llm import MockLLM, get_llm
from .output_control import build_constraints
from .prompt_optimizer import diff_segments, optimize_prompt
from .rag import KnowledgeBase
from .router import pick_model
from .tokenizer import cost_usd, count_tokens

ALL_ON = {"prompt": True, "cache": True, "output": True, "rag": True, "model": True, "post": True}
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "history.json")


class Pipeline:
    def __init__(self, persist=True):
        self.kb = KnowledgeBase()
        self.cache = SemanticCache()
        self.llm = get_llm()
        self.persist = persist
        self.history = []
        if persist:
            self._load()

    # ---------------- persistence ----------------
    def _load(self):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                self.history = json.load(f)
        except Exception:
            self.history = []

    def _save(self):
        if not self.persist:
            return
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[-500:], f)
        except Exception:
            pass

    def reset_history(self):
        self.history = []
        self.cache.clear()
        self._save()

    # ---------------- main entry ----------------
    def run(self, req: dict) -> dict:
        t0 = time.time()
        raw_q = (req.get("prompt") or "").strip()
        if not raw_q:
            raise ValueError("Prompt is empty.")
        on = {**ALL_ON, **(req.get("techniques") or {})}
        fmt = req.get("format", "text")
        max_words = req.get("max_words", "auto")
        top_k = int(req.get("top_k") or DEFAULT_TOP_K)
        override = req.get("model", "auto")
        doc_ids = [d for d in (req.get("doc_ids") or []) if d in self.kb.docs] or None
        L = BASELINE_TIER

        # ---- baseline (what an un-optimised app would send) ----
        full_ctx = self.kb.full_context(doc_ids)
        full_ctx_tok = count_tokens(full_ctx)
        base_user = f"Context:\n{full_ctx}\n\nQuestion: {raw_q}"
        in0 = count_tokens(BASELINE_SYSTEM) + count_tokens(base_user)

        # ---- 1. prompt optimisation ----
        opt_q = optimize_prompt(raw_q) if on["prompt"] else raw_q
        system = OPTIMIZED_SYSTEM if on["prompt"] else BASELINE_SYSTEM
        p_saved = max(0, (count_tokens(BASELINE_SYSTEM) + count_tokens(raw_q))
                      - (count_tokens(system) + count_tokens(opt_q)))
        diff = diff_segments(raw_q, opt_q) if on["prompt"] else [{"t": raw_q, "op": "keep"}]

        # ---- 2. cache ----
        signature = f"{sorted(doc_ids or [])}|{fmt}|{max_words}|{top_k}|{override}|{on['output']}|{on['post']}"
        cached, sim = (self.cache.get(opt_q, signature) if on["cache"] else (None, 0.0))

        if cached:
            res = self._from_cache(cached["payload"], sim, raw_q, opt_q, diff, on, in0, p_saved)
            res["latency_ms"] = int((time.time() - t0) * 1000) + 2
            self._record(res, raw_q)
            return res

        # ---- 4. retrieval (RAG) ----
        if on["rag"] and self.kb.chunks:
            retrieved = self.kb.search(opt_q, top_k, doc_ids)
            ctx = "\n\n".join(c["text"] for c in retrieved)
        else:
            retrieved, ctx = [], full_ctx
        ctx_tok = count_tokens(ctx)
        rag_saved = max(0, full_ctx_tok - ctx_tok) if on["rag"] else 0

        # ---- 5. model selection ----
        if on["model"] or override in TIERS:
            model = pick_model(opt_q, ctx_tok, override)
        else:
            model = {"tier": L, "model": TIERS[L]["model"], "complexity": None,
                     "reasons": ["model routing disabled – always using the largest model"]}
        tier = model["tier"]

        # ---- 3. output control ----
        if on["output"]:
            instr, limit, max_tokens = build_constraints(opt_q, max_words, fmt, on["post"])
        else:
            instr, limit, max_tokens = "", 10_000, 1024
        user = f"Context:\n{ctx}\n\nQuestion: {opt_q}" + (f"\n\n{instr}" if instr else "")

        # ---- LLM call ----
        ask_json = fmt == "json" and not on["post"]
        kw = dict(max_words=limit if on["output"] else None, fmt=fmt, ask_json=ask_json,
                  verbose=not on["output"])
        if self.llm.mode == "live":
            kw["max_tokens"] = max_tokens
        r = self.llm.generate(tier, system, user, opt_q, ctx, **kw)
        in_f, out_f = r["tokens_in"], r["tokens_out"]

        # ---- baseline output length ----
        if self.llm.mode == "demo":
            b = MockLLM().generate(L, BASELINE_SYSTEM, base_user, raw_q, full_ctx, None, fmt,
                                   ask_json=(fmt == "json"), verbose=True)
            out0 = b["tokens_out"]
        else:
            out0 = out_f if not on["output"] else int(out_f * 2.5)
        out0 = max(out0, out_f)

        # ---- 6. post-processing (code, not AI) ----
        sources = [f"{c['doc']} #{c['idx'] + 1}" for c in retrieved]
        meta = {"model": model["model"], "tokens": in_f + out_f}
        if on["post"]:
            answer = postprocess.finalize(r["text"], fmt, limit if on["output"] else 10_000,
                                          sources, meta if fmt == "json" else None)
        else:
            answer = r["text"]

        # ---- cost waterfall ----
        c0 = cost_usd(L, in0, out0)
        c1 = cost_usd(L, in0 - p_saved, out0)
        c2 = cost_usd(L, in0 - p_saved - rag_saved, out0)
        c3 = cost_usd(L, in_f, out_f)
        c4 = cost_usd(tier, in_f, out_f)
        wf = {"prompt": c0 - c1, "rag": c1 - c2, "output": c2 - c3, "model": c3 - c4, "cache": 0.0}

        raw_words, fin_words = len(r["text"].split()), len(answer.split())
        stages = [
            self._stage(1, "prompt", "Prompt optimization", on["prompt"],
                        f"System + prompt: {count_tokens(BASELINE_SYSTEM) + count_tokens(raw_q)} → "
                        f"{count_tokens(system) + count_tokens(opt_q)} tokens", p_saved, wf["prompt"]),
            self._stage(2, "cache", "Smart system design", on["cache"],
                        f"Cache miss (best match {sim:.0%}) – answer stored for next time"
                        if on["cache"] else "Cache disabled", 0, 0.0, "miss" if on["cache"] else None),
            self._stage(3, "output", "Output control", on["output"],
                        f"Limit {limit} words · max_tokens {max_tokens} · answer {out0} → {out_f} tokens"
                        if on["output"] else "No limit – model answers at full length",
                        max(0, out0 - out_f), wf["output"]),
            self._stage(4, "rag", "Chunking & retrieval", on["rag"],
                        f"{len(retrieved)} of {len(self.kb.chunks)} chunks · context {full_ctx_tok} → {ctx_tok} tokens"
                        if on["rag"] else f"Whole knowledge base sent ({full_ctx_tok} tokens)",
                        rag_saved, wf["rag"]),
            self._stage(5, "model", "Model selection", on["model"] or override in TIERS,
                        f"{TIERS[tier]['label']} model" + (f" · complexity {model['complexity']}/100"
                                                           if model["complexity"] is not None else ""),
                        0, wf["model"]),
            self._stage(6, "post", "Post-processing", on["post"],
                        f"Filler stripped & {fmt} formatted by code · {raw_words} → {fin_words} words"
                        if on["post"] else "Raw model output returned", 0, 0.0),
        ]

        res = {
            "answer": answer, "format": fmt, "mode": self.llm.mode,
            "cache": {"hit": False, "similarity": round(sim, 3)},
            "stages": stages, "prompt_diff": diff, "optimized_prompt": opt_q,
            "retrieved": retrieved, "model": model, "waterfall": wf,
            "usage": {
                "baseline": {"input": in0, "output": out0, "cost": c0, "tier": L},
                "optimized": {"input": in_f, "output": out_f, "cost": c4, "tier": tier},
            },
        }
        self._finish(res, t0, tier)
        if on["cache"]:
            self.cache.put(opt_q, signature, {
                "answer": answer, "retrieved": retrieved, "model": model, "out0": out0,
                "stages": stages, "usage": res["usage"]})
        self._record(res, raw_q)
        return res

    # ---------------- helpers ----------------
    @staticmethod
    def _stage(no, sid, name, enabled, detail, saved_tokens, saved_cost, status=None):
        return {"no": no, "id": sid, "name": name, "enabled": bool(enabled),
                "status": status or ("applied" if enabled else "off"),
                "detail": detail, "saved_tokens": int(saved_tokens), "saved_cost": saved_cost}

    def _from_cache(self, p, sim, raw_q, opt_q, diff, on, in0, p_saved):
        out0 = max(p.get("out0", 0), 1)
        c0 = cost_usd(BASELINE_TIER, in0, out0)
        stages = []
        for s in p["stages"]:
            s = dict(s)
            if s["id"] == "cache":
                s.update(status="hit", enabled=True, saved_cost=c0,
                         detail=f"Cache hit ({sim:.0%} similar) – model was never called")
            else:
                s.update(saved_cost=0.0, saved_tokens=0,
                         status="skipped" if s["enabled"] else "off",
                         detail="Skipped – answered from cache" if s["enabled"] else s["detail"])
            stages.append(s)
        wf = {"prompt": 0.0, "rag": 0.0, "output": 0.0, "model": 0.0, "cache": c0}
        res = {
            "answer": p["answer"], "format": "cached", "mode": self.llm.mode,
            "cache": {"hit": True, "similarity": round(sim, 3)},
            "stages": stages, "prompt_diff": diff, "optimized_prompt": opt_q,
            "retrieved": p["retrieved"], "model": p["model"], "waterfall": wf,
            "usage": {"baseline": {"input": in0, "output": out0, "cost": c0, "tier": BASELINE_TIER},
                      "optimized": {"input": 0, "output": 0, "cost": 0.0, "tier": p["model"]["tier"]}},
        }
        self._finish(res, time.time() - 0.002, p["model"]["tier"], cached=True)
        return res

    @staticmethod
    def _finish(res, t0, tier, cached=False):
        b, o = res["usage"]["baseline"], res["usage"]["optimized"]
        bt, ot = b["input"] + b["output"], o["input"] + o["output"]
        res["savings"] = {
            "tokens_saved": bt - ot, "tokens_pct": round(100 * (bt - ot) / bt, 1) if bt else 0,
            "cost_saved": b["cost"] - o["cost"],
            "cost_pct": round(100 * (b["cost"] - o["cost"]) / b["cost"], 1) if b["cost"] else 0,
        }
        res["inr_rate"] = USD_TO_INR
        if not cached:
            res["latency_ms"] = int((time.time() - t0) * 1000) + (
                TIERS[tier]["latency_ms"] if res["mode"] == "demo" else 0)

    def _record(self, res, raw_q):
        u = res["usage"]
        self.history.append({
            "id": uuid.uuid4().hex[:8], "ts": int(time.time()), "prompt": raw_q[:120],
            "base_tokens": u["baseline"]["input"] + u["baseline"]["output"],
            "opt_tokens": u["optimized"]["input"] + u["optimized"]["output"],
            "base_cost": u["baseline"]["cost"], "opt_cost": u["optimized"]["cost"],
            "cache_hit": res["cache"]["hit"], "tier": res["model"]["tier"],
            "waterfall": res["waterfall"],
        })
        self._save()

    # ---------------- analytics ----------------
    def stats(self):
        h = self.history
        n = len(h)
        bt, ot = sum(x["base_tokens"] for x in h), sum(x["opt_tokens"] for x in h)
        bc, oc = sum(x["base_cost"] for x in h), sum(x["opt_cost"] for x in h)
        wf = {k: sum(x["waterfall"].get(k, 0) for x in h) for k in ("prompt", "cache", "output", "rag", "model")}
        tiers = {t: sum(1 for x in h if x["tier"] == t and not x["cache_hit"]) for t in TIERS}
        return {
            "requests": n, "cache_hits": sum(1 for x in h if x["cache_hit"]),
            "base_tokens": bt, "opt_tokens": ot, "base_cost": bc, "opt_cost": oc,
            "tokens_pct": round(100 * (bt - ot) / bt, 1) if bt else 0,
            "cost_pct": round(100 * (bc - oc) / bc, 1) if bc else 0,
            "waterfall": wf, "tiers": tiers, "inr_rate": USD_TO_INR,
            "series": h[-30:], "cache": self.cache.info(), "mode": self.llm.mode,
        }
