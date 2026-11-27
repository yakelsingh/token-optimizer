"""TokenWise – Flask web server.   Run:  python app.py   ->  http://127.0.0.1:5000"""
import glob
import os

from flask import Flask, jsonify, render_template, request

from optimizer.config import TIERS
from optimizer.pipeline import Pipeline

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
pipe = Pipeline()


def load_samples():
    for path in sorted(glob.glob(os.path.join(BASE, "sample_docs", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            pipe.kb.add(os.path.basename(path), f.read())


load_samples()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(mode=pipe.llm.mode, tiers={k: {"model": v["model"], "in": v["in"], "out": v["out"]}
                                              for k, v in TIERS.items()})


@app.post("/api/optimize")
def optimize():
    try:
        return jsonify(pipe.run(request.get_json(force=True) or {}))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:                    # keep the UI alive on provider errors
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/documents")
def docs_list():
    return jsonify(pipe.kb.list())


@app.post("/api/documents")
def docs_add():
    d = request.get_json(force=True) or {}
    name, text = (d.get("name") or "Untitled").strip(), (d.get("text") or "").strip()
    if len(text) < 20:
        return jsonify(error="Document is too short (min 20 characters)."), 400
    return jsonify(pipe.kb.add(name, text)), 201


@app.delete("/api/documents/<doc_id>")
def docs_del(doc_id):
    return jsonify(ok=pipe.kb.remove(doc_id))


@app.get("/api/stats")
def stats():
    return jsonify(pipe.stats())


@app.post("/api/reset")
def reset():
    pipe.reset_history()
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"TokenWise running in {pipe.llm.mode.upper()} mode -> http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
