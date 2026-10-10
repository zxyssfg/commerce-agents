# CrossCart prototype

CrossCart is a supervisor-worker extension to the retail example. It keeps the existing
single-agent tool executor and adds three production seams:

1. **Hybrid retrieval**: BM25 plus a sparse TF-IDF vector leg, fused by Reciprocal Rank
   Fusion. The vector leg uses word and character n-gram features; a hosted embedding
   service can replace it without changing the backend contract.
2. **Supervisor routing**: a small LLM call routes a request to Search, Comparison, or
   Customer Care. The route decision and query rewrite are isolated from deterministic
   workers.
3. **Parallel platform workers**: three marketplace clients run concurrently and return
   a normalized landed-price view.

## Run

```bash
source .venv/bin/activate
export PYTHONPATH=examples
python -m crosscart.benchmark --suite all --rounds 3
python -m crosscart.benchmark --suite search --no-llm
pytest crosscart/tests/test_crosscart.py
```

`--no-llm` uses deterministic rules. Without it, CrossCart calls the model in
`MEMORY_MODEL` for routing and query rewriting.

## Measured results

The latest live run used the 87-product retail catalog and 42 labeled queries:

| Retrieval path | Hit@1 | Recall@5 | MRR | NDCG@5 | Search latency P95 |
|---|---:|---:|---:|---:|---:|
| Baseline keyword score | 0.6429 | 0.7976 | 0.7262 | 0.7359 | 2.022 ms |
| Hybrid BM25 + sparse vector | 0.6429 | 0.9048 | 0.7337 | 0.7747 | 0.642 ms |
| Hybrid + LLM query rewrite | 0.7619 | 0.9524 | 0.8401 | 0.8636 | 3.710 ms |

The LLM rewrite adds a P50/P95 of about 1.23/1.69 seconds for query understanding, so it
should run asynchronously for typed suggestions rather than blocking a checkout action.

The parallel test models three platform APIs with controlled 80/120/200 ms delays:

| Orchestration | Mean | P50 | P95 |
|---|---:|---:|---:|
| Serial platform search | 411.811 ms | 411.619 ms | 414.892 ms |
| Parallel platform search | 204.241 ms | 204.173 ms | 205.797 ms |

Platform IO is simulated; orchestration overhead and result normalization are real. The
full JSON is saved at `crosscart/results/benchmark-live.json`.
