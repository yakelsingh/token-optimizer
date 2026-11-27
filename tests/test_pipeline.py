import glob, os, unittest, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.pipeline import Pipeline
from optimizer.prompt_optimizer import optimize_prompt
from optimizer.tokenizer import count_tokens
from optimizer.router import pick_model
from optimizer.rag import chunk_text
from optimizer import postprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make():
    p = Pipeline(persist=False)
    for f in sorted(glob.glob(os.path.join(BASE, "sample_docs", "*.txt"))):
        p.kb.add(os.path.basename(f), open(f, encoding="utf-8").read())
    return p


class T(unittest.TestCase):
    def test_prompt_shorter(self):
        q = "Hello! Could you please kindly explain what a token is? Thank you so much!"
        self.assertLess(count_tokens(optimize_prompt(q)), count_tokens(q))
        self.assertIn("token", optimize_prompt(q))

    def test_router(self):
        self.assertEqual(pick_model("What is a token?")["tier"], "small")
        hard = "Compare and analyze the architecture, then design and justify an algorithm? Why is it optimal? How does it scale?"
        self.assertEqual(pick_model(hard)["tier"], "large")

    def test_chunking(self):
        self.assertGreater(len(chunk_text(open(os.path.join(BASE, "sample_docs", "01_tokens_and_cost.txt")).read())), 3)

    def test_rag_finds_relevant(self):
        top = make().kb.search("what is BM25 chunk ranking", 1)[0]
        self.assertIn("BM25", top["text"])

    def test_postprocess(self):
        out = postprocess.finalize("Great question! Paris is the capital. I hope this helps!", "bullets", 50)
        self.assertTrue(out.startswith("•"))
        self.assertNotIn("hope", out)
        self.assertEqual(postprocess.extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_savings_and_waterfall(self):
        r = make().run({"prompt": "Please explain how caching saves tokens?"})
        self.assertGreater(r["savings"]["tokens_pct"], 30)
        self.assertAlmostEqual(sum(r["waterfall"].values()), r["savings"]["cost_saved"], places=9)

    def test_cache_hit(self):
        p = make()
        p.run({"prompt": "Explain how caching saves tokens?"})
        r = p.run({"prompt": "Please explain how caching saves tokens"})
        self.assertTrue(r["cache"]["hit"])
        self.assertEqual(r["usage"]["optimized"]["cost"], 0)

    def test_all_off_is_baseline(self):
        off = {k: False for k in ["prompt", "cache", "output", "rag", "model", "post"]}
        r = make().run({"prompt": "Explain RAG", "techniques": off})
        self.assertEqual(r["savings"]["tokens_saved"], 0)

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            make().run({"prompt": "  "})


if __name__ == "__main__":
    unittest.main()
