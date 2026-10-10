from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from retail.api.mock_retail import MockRetail
from shopping_agent.types import Product


@dataclass
class PlatformHit:
    platform: str
    product: Product
    price: float
    shipping_fee: float
    eta: str

    @property
    def landed_price(self) -> float:
        return round(self.price + self.shipping_fee, 2)


@dataclass
class RouteDecision:
    worker: str
    query: str
    reason: str = ""
    routing_ms: int = 0


@dataclass
class SupervisorResult:
    route: RouteDecision
    hits: list[PlatformHit] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    routing_ms: int = 0
    worker_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)


class PlatformSearchClient:
    """A worker client for one marketplace. The delay models remote API IO only."""

    def __init__(
        self,
        backend: MockRetail,
        platform: str,
        *,
        delay_ms: int,
        price_multiplier: float = 1.0,
        shipping_fee: float = 0.0,
        eta: str = "3-5 days",
    ) -> None:
        self.backend = backend
        self.platform = platform
        self.delay_ms = delay_ms
        self.price_multiplier = price_multiplier
        self.shipping_fee = shipping_fee
        self.eta = eta

    async def search(self, query: str, limit: int = 5) -> list[PlatformHit]:
        await asyncio.sleep(self.delay_ms / 1000)
        products = await self.backend.search_products(None, query, None, limit)
        return [
            PlatformHit(
                platform=self.platform,
                product=product,
                price=round(product.price * self.price_multiplier, 2),
                shipping_fee=self.shipping_fee,
                eta=self.eta,
            )
            for product in products
        ]


class SearchWorker:
    def __init__(self, clients: list[PlatformSearchClient]) -> None:
        self.clients = clients

    async def search(self, query: str, limit: int = 5) -> list[PlatformHit]:
        started = time.monotonic()
        batches = await asyncio.gather(
            *(client.search(query, limit) for client in self.clients),
            return_exceptions=True,
        )
        hits = [hit for batch in batches if not isinstance(batch, BaseException) for hit in batch]
        best: dict[str, PlatformHit] = {}
        for hit in hits:
            current = best.get(hit.product.product_id)
            if current is None or hit.landed_price < current.landed_price:
                best[hit.product.product_id] = hit
        self.last_elapsed_ms = int((time.monotonic() - started) * 1000)
        return sorted(best.values(), key=lambda hit: hit.landed_price)


class LLMQueryRewriter:
    """Turn colloquial requests into keyword-rich catalog queries."""

    def __init__(
        self, client: AsyncAnthropic | None = None, model: str = "Claude Haiku 4.5"
    ) -> None:
        self.client = client
        self.model = model

    async def rewrite(self, query: str) -> tuple[str, int, int]:
        if self.client is None:
            return query, 0, 0
        prompt = f"""Rewrite a shopping request as 2-8 catalog keywords. Do not invent
brands, prices, or product ids. Return JSON only:
{{"query":"coffee maker espresso machine"}}

Request: {query}"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return query, response.usage.input_tokens, response.usage.output_tokens
        try:
            rewritten = str(json.loads(match.group(0)).get("query") or query)
        except json.JSONDecodeError:
            rewritten = query
        return rewritten, response.usage.input_tokens, response.usage.output_tokens


class ComparisonWorker:
    def __init__(self, search_worker: SearchWorker) -> None:
        self.search_worker = search_worker

    async def compare(
        self, query: str, limit: int = 5
    ) -> tuple[list[PlatformHit], list[dict[str, Any]]]:
        hits = await self.search_worker.search(query, limit)
        table = [
            {
                "product_id": hit.product.product_id,
                "title": hit.product.title,
                "platform": hit.platform,
                "item_price": hit.price,
                "shipping": hit.shipping_fee,
                "landed_price": hit.landed_price,
                "eta": hit.eta,
            }
            for hit in hits[:limit]
        ]
        return hits, table


class CrossCartSupervisor:
    """Route, fan out, and normalize worker output.

    The LLM only emits a tiny JSON route decision; deterministic work happens in workers.
    A rule fallback keeps the demo usable without model credentials.
    """

    def __init__(
        self,
        search_worker: SearchWorker,
        comparison_worker: ComparisonWorker,
        *,
        client: AsyncAnthropic | None = None,
        model: str = "Claude Haiku 4.5",
    ) -> None:
        self.search_worker = search_worker
        self.comparison_worker = comparison_worker
        self.client = client
        self.model = model
        self.rewriter = LLMQueryRewriter(client, model)

    def _rule_route(self, message: str) -> RouteDecision:
        text = message.lower()
        if any(word in text for word in ("compare", "cheaper", "best price", "vs")):
            worker, query = "comparison", text
        elif any(word in text for word in ("cart", "checkout", "order", "return", "refund")):
            worker, query = "customer_care", text
        else:
            worker, query = "search", text
        return RouteDecision(worker=worker, query=query, reason="rule fallback")

    async def _route(self, message: str) -> tuple[RouteDecision, dict[str, int]]:
        if self.client is None:
            return self._rule_route(message), {}
        prompt = f"""Classify a shopping request. Return JSON only:
{{"worker":"search|comparison|customer_care","query":"normalized product search or original request","reason":"one short phrase"}}

Request: {message}"""
        started = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return self._rule_route(message), {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        payload = json.loads(match.group(0))
        worker = payload.get("worker", "search")
        if worker not in {"search", "comparison", "customer_care"}:
            worker = "search"
        usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
        decision = RouteDecision(
            worker=worker,
            query=str(payload.get("query") or message),
            reason=str(payload.get("reason") or ""),
        )
        decision.routing_ms = int((time.monotonic() - started) * 1000)
        return decision, usage

    async def handle(self, message: str) -> SupervisorResult:
        started = time.monotonic()
        route, usage = await self._route(message)
        worker_started = time.monotonic()
        result = SupervisorResult(route=route, usage=usage)
        if route.worker in {"search", "comparison"}:
            route.query, rewrite_input, rewrite_output = await self.rewriter.rewrite(route.query)
            usage["input"] = usage.get("input", 0) + rewrite_input
            usage["output"] = usage.get("output", 0) + rewrite_output
        if route.worker == "comparison":
            result.hits, result.comparison = await self.comparison_worker.compare(route.query)
            result.reply = "Comparison table generated from parallel platform workers."
        elif route.worker == "customer_care":
            result.reply = (
                "CrossCart routes policy, order, and cart questions to the customer-care "
                "worker in the full deployment; this prototype focuses on search workers."
            )
        else:
            result.hits = await self.search_worker.search(route.query)
            result.reply = "Search results normalized across platforms by the search worker."
        result.worker_ms = int((time.monotonic() - worker_started) * 1000)
        result.routing_ms = route.routing_ms or int((time.monotonic() - started) * 1000)
        return result
