from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from retail.api.mock_retail import MockRetail
from shopping_agent.types import ShoppingSessionContext

from .backend import CrossCartSearchBackend
from .supervisor import (
    ComparisonWorker,
    CrossCartSupervisor,
    LLMQueryRewriter,
    PlatformSearchClient,
    SearchWorker,
)

QUERY_PATH = Path(__file__).with_name("test_queries.json")
RESULT_DIR = Path(__file__).parent / "results"


def load_queries() -> list[dict[str, Any]]:
    return json.loads(QUERY_PATH.read_text(encoding="utf-8"))


def percentiles(
    values: Iterable[float], points: tuple[float, ...] = (0.5, 0.95)
) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {f"p{int(point * 100)}": 0.0 for point in points}
    output = {}
    for point in points:
        position = (len(ordered) - 1) * point
        low, high = math.floor(position), math.ceil(position)
        value = ordered[low]
        if low != high:
            value += (ordered[high] - ordered[low]) * (position - low)
        output[f"p{int(point * 100)}"] = round(value, 3)
    return output


def metrics_at_5(predicted: list[str], relevant: list[str]) -> dict[str, float]:
    predicted_ids = predicted[:5]
    relevant_set = set(relevant)
    hits = sum(product_id in relevant_set for product_id in predicted_ids)
    recall = hits / len(relevant_set) if relevant_set else 0.0
    first_rank = next(
        (
            rank
            for rank, product_id in enumerate(predicted_ids, start=1)
            if product_id in relevant_set
        ),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, product_id in enumerate(predicted_ids, start=1)
        if product_id in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(5, len(relevant_set)) + 1))
    return {
        "hit_at_1": float(first_rank == 1),
        "recall_at_5": recall,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in ("hit_at_1", "recall_at_5", "mrr", "ndcg_at_5")}
    return {name: round(statistics.mean(row[name] for row in rows), 4) for name in rows[0]}


async def measure_backend(
    backend: MockRetail,
    queries: list[dict[str, Any]],
    rewriter: LLMQueryRewriter | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    session = ShoppingSessionContext(user_id="bench", session_id="bench")
    quality_rows: list[dict[str, float]] = []
    latency_rows: list[dict[str, Any]] = []
    token_usage = {"input": 0, "output": 0}
    failures = 0
    for item in queries:
        query = item["query"]
        rewritten = query
        if rewriter is not None:
            started = time.monotonic()
            try:
                rewritten, input_tokens, output_tokens = await rewriter.rewrite(query)
                token_usage["input"] += input_tokens
                token_usage["output"] += output_tokens
            except Exception:
                failures += 1
                rewritten = query
            rewrite_ms = (time.monotonic() - started) * 1000
        else:
            rewrite_ms = 0.0
        started = time.monotonic()
        products = await backend.search_products(session, rewritten, None, 5)
        search_ms = (time.monotonic() - started) * 1000
        quality_rows.append(
            metrics_at_5([product.product_id for product in products], item["relevant"])
        )
        latency_rows.append(
            {
                "query": query,
                "rewritten_query": rewritten,
                "rewrite_ms": round(rewrite_ms, 3),
                "search_ms": round(search_ms, 3),
                "predicted": [product.product_id for product in products],
            }
        )
    return {
        "metrics": aggregate(quality_rows),
        "latency_ms": percentiles([row["search_ms"] for row in latency_rows]),
        "rewrite_latency_ms": percentiles([row["rewrite_ms"] for row in latency_rows]),
        "token_usage": token_usage,
        "rewrite_failures": failures,
        "queries": latency_rows,
    }, quality_rows


async def search_benchmark(use_llm: bool) -> dict[str, Any]:
    queries = load_queries()
    baseline = MockRetail()
    hybrid = CrossCartSearchBackend()
    rewriter = None
    if use_llm:
        client = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
        rewriter = LLMQueryRewriter(
            client, model=os.environ.get("MEMORY_MODEL", "Claude Haiku 4.5")
        )
    baseline_result, _ = await measure_backend(baseline, queries)
    hybrid_result, _ = await measure_backend(hybrid, queries)
    result: dict[str, Any] = {"baseline": baseline_result, "hybrid": hybrid_result}
    if rewriter is not None:
        result["hybrid_with_llm_query_rewrite"], _ = await measure_backend(
            hybrid, queries, rewriter
        )
    return result


def platform_clients(backend: MockRetail) -> list[PlatformSearchClient]:
    return [
        PlatformSearchClient(
            backend,
            "north",
            delay_ms=80,
            price_multiplier=1.0,
            shipping_fee=0.0,
            eta="3-5 days",
        ),
        PlatformSearchClient(
            backend,
            "europe",
            delay_ms=120,
            price_multiplier=1.05,
            shipping_fee=4.99,
            eta="4-7 days",
        ),
        PlatformSearchClient(
            backend,
            "asia",
            delay_ms=200,
            price_multiplier=0.94,
            shipping_fee=9.99,
            eta="7-12 days",
        ),
    ]


async def parallel_benchmark(rounds: int = 3) -> dict[str, Any]:
    backend = CrossCartSearchBackend()
    clients = platform_clients(backend)
    worker = SearchWorker(clients)
    queries = load_queries()
    serial_ms: list[float] = []
    parallel_ms: list[float] = []
    comparisons = 0
    for _ in range(rounds):
        for item in queries:
            started = time.monotonic()
            for client in clients:
                await client.search(item["query"], 5)
            serial_ms.append((time.monotonic() - started) * 1000)
            started = time.monotonic()
            hits = await worker.search(item["query"], 5)
            parallel_ms.append((time.monotonic() - started) * 1000)
            if hits:
                comparisons += 1
    return {
        "platform_io": "controlled simulated delays: 80ms, 120ms, 200ms",
        "platforms": [client.platform for client in clients],
        "rounds": rounds,
        "serial": {"mean_ms": round(statistics.mean(serial_ms), 3), **percentiles(serial_ms)},
        "parallel": {"mean_ms": round(statistics.mean(parallel_ms), 3), **percentiles(parallel_ms)},
        "normalized_results": comparisons,
    }


async def supervisor_smoke(use_llm: bool) -> dict[str, Any]:
    backend = CrossCartSearchBackend()
    clients = platform_clients(backend)
    search_worker = SearchWorker(clients)
    comparison_worker = ComparisonWorker(search_worker)
    client = None
    if use_llm:
        client = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
    supervisor = CrossCartSupervisor(
        search_worker,
        comparison_worker,
        client=client,
        model=os.environ.get("MEMORY_MODEL", "Claude Haiku 4.5"),
    )
    rows = []
    for message in ("Find a cold weather sleeping bag", "Compare luggage prices"):
        result = await supervisor.handle(message)
        rows.append(
            {
                "message": message,
                "worker": result.route.worker,
                "query": result.route.query,
                "routing_ms": result.routing_ms,
                "worker_ms": result.worker_ms,
                "result_count": len(result.hits),
                "comparison_rows": len(result.comparison),
                "usage": result.usage,
            }
        )
    return {"mode": "live LLM" if use_llm else "rule fallback", "runs": rows}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrossCart retrieval and routing benchmarks")
    parser.add_argument(
        "--suite", choices=("all", "search", "parallel", "supervisor"), default="all"
    )
    parser.add_argument("--no-llm", action="store_true", help="skip OneAPI calls")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    use_llm = not args.no_llm
    output: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "catalog_products": len(MockRetail().products),
        "queries": len(load_queries()),
    }
    if args.suite in {"all", "search"}:
        output["search"] = await search_benchmark(use_llm)
    if args.suite in {"all", "parallel"}:
        output["parallel"] = await parallel_benchmark(args.rounds)
    if args.suite in {"all", "supervisor"}:
        output["supervisor"] = await supervisor_smoke(use_llm)

    RESULT_DIR.mkdir(exist_ok=True)
    suffix = "live" if use_llm else "offline"
    path = RESULT_DIR / f"benchmark-{suffix}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
    if "search" in output:
        for name, result in output["search"].items():
            print(name, result["metrics"], result["latency_ms"])
    if "parallel" in output:
        print("serial", output["parallel"]["serial"])
        print("parallel", output["parallel"]["parallel"])


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
