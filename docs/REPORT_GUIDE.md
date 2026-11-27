# Report guide – TokenWise

## Suggested title
"TokenWise: A Multi-Layer Framework for Reducing Token Consumption and Cost in LLM Applications"

## Abstract (starter)
LLM providers bill per token, and many applications waste tokens through verbose prompts,
repeated context, unlimited answers and always using the largest model. This project
implements six complementary techniques – prompt optimization, semantic caching, output
control, chunking & retrieval, model routing and code-based post-processing – and a
dashboard that measures the savings against an un-optimized baseline.

## Chapters
1. Introduction – problem, motivation (cost, latency, scalability), objectives
2. Literature survey – tokenization (BPE), RAG, semantic caching, LLM routing, prompt compression
3. System analysis – requirements, feasibility, use-case diagram
4. System design – architecture (request → prompt optimizer → cache → retriever → router → LLM → post-processor), DFD, class diagram, API list
5. Implementation – one section per module (algorithms: regex rewriting, BM25, cosine similarity, complexity scoring)
6. Testing & results – unit tests, ablation table (turn each technique off), screenshots
7. Conclusion & future scope

## Ablation experiment (great for Results chapter)
Run the same prompt 7 times: all techniques on, then switch off exactly one each time. Record
tokens and cost from the receipt. The technique whose removal hurts most is the biggest saver.

## Likely viva questions
* Why are output tokens more expensive? (autoregressive generation, one forward pass per token)
* How does BM25 differ from embeddings? (lexical vs semantic; can be combined)
* What is cosine similarity? (angle between vectors: dot(a,b)/(|a||b|))
* Why not always use the smallest model? (quality drops on reasoning tasks – hence the router)
* Limitations? (heuristic token counter, rule-based router, cache can return stale answers)
