# TokenWise – AI Token & Cost Optimizer
B.Tech (CSE) final-year project · Python + Flask + vanilla JavaScript

TokenWise sits between an application and a Large Language Model and applies
**six token-saving techniques**. Every request is shown as a receipt comparing
an un-optimized baseline with the optimized run.

| # | Technique | Module | What it does |
|---|-----------|--------|--------------|
| 1 | Prompt optimization | `optimizer/prompt_optimizer.py` | Removes filler/politeness, rewrites wordy phrases, drops duplicate sentences, shortens the system prompt |
| 2 | Smart system design | `optimizer/cache.py` | Exact + semantic cache (cosine similarity) – repeated questions cost 0 tokens |
| 3 | Output control | `optimizer/output_control.py` | Word limit in the prompt + hard `max_tokens` |
| 4 | Chunking & retrieval (RAG) | `optimizer/rag.py` | Sentence-aware chunks, BM25 ranking, top-k only |
| 5 | Model selection | `optimizer/router.py` | Complexity score 0–100 → small / medium / large model |
| 6 | Post-processing logic | `optimizer/postprocess.py` | Filler removal, truncation, bullets and JSON built with code |

## Quick start (Windows / macOS / Linux)
```bash
cd token-optimizer
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000**

Run the tests: `python -m unittest discover -s tests`

## Demo mode vs live mode
* **Demo mode (default)** – no API key needed. A deterministic extractive model
  answers from your documents, so the project works offline in a viva.
* **Live mode** – `pip install anthropic`, then set `ANTHROPIC_API_KEY`
  (`set ANTHROPIC_API_KEY=...` on Windows, `export ...` on macOS/Linux) and restart.
  Real token usage from the API is used for the receipt.

## How savings are measured (be ready to explain this)
Baseline = long system prompt + raw prompt + **all** documents + the largest model
+ an unlimited answer. Savings are attributed in order
(prompt → retrieval → output → model), so the per-technique numbers always add up
to the total. Prices are illustrative and live in `optimizer/config.py`.
In demo mode the baseline answer length is simulated; in live mode it is estimated.
Large savings (90 %+) appear because the baseline is deliberately naive; the real-world
range is lower (the reference infographic says 40–70 %).

## Project structure
```
app.py                 Flask server + REST API
optimizer/             the six techniques + pipeline + LLM back-ends
templates/index.html   UI          static/style.css, static/app.js
sample_docs/           3 demo documents (loaded on start)
tests/                 unit tests
docs/REPORT_GUIDE.md   chapter outline, viva questions, future work
```

## API
`POST /api/optimize` · `GET/POST /api/documents` · `DELETE /api/documents/<id>` · `GET /api/stats` · `POST /api/reset` · `GET /api/health`

## Ideas to extend (for marks)
Embedding-based retrieval (sentence-transformers), provider prompt-caching, PDF upload,
user login + per-user budgets, A/B tests of prompts, a learned router instead of rules.
