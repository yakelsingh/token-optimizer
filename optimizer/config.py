"""Central configuration: model tiers, prices and defaults.

Prices are ILLUSTRATIVE (USD per 1 million tokens). Edit them to match the
provider you use – every saving shown in the UI is computed from this table.
"""
import os

TIERS = {
    "small":  {"label": "Small",  "model": os.getenv("TW_MODEL_SMALL",  "claude-haiku-5-5"),  "in": 1.0,  "out": 5.0,  "latency_ms": 350},
    "medium": {"label": "Medium", "model": os.getenv("TW_MODEL_MEDIUM", "claude-sonnet-5-5"), "in": 3.0,  "out": 15.0, "latency_ms": 700},
    "large":  {"label": "Large",  "model": os.getenv("TW_MODEL_LARGE",  "claude-opus-5-5"),   "in": 15.0, "out": 75.0, "latency_ms": 1400},
}

# The "naive" baseline every optimisation is compared against:
BASELINE_TIER = "large"

USD_TO_INR = float(os.getenv("TW_USD_TO_INR", "88"))

# Verbose system prompt used by the un-optimised baseline.
BASELINE_SYSTEM = (
    "You are a helpful, friendly and knowledgeable AI assistant. Please always answer the "
    "user's questions as thoroughly and completely as possible. Provide detailed explanations, "
    "include background information, give examples where you can, and be polite at all times. "
    "If you are not sure about something, explain your reasoning step by step before answering."
)
OPTIMIZED_SYSTEM = "Answer concisely using only the given context."

CACHE_THRESHOLD = 0.75          # semantic-cache cosine similarity
CACHE_TTL_SECONDS = 3600
CHUNK_WORDS = 55                # target words per chunk
DEFAULT_TOP_K = 3
