from __future__ import annotations

from crosscart.backend import CrossCartSearchBackend
from crosscart.hybrid_search import HybridSearchEngine
from crosscart.supervisor import (
    ComparisonWorker,
    CrossCartSupervisor,
    PlatformSearchClient,
    SearchWorker,
)

from retail.api.mock_retail import MockRetail


def test_hybrid_ignores_zero_scored_documents() -> None:
    engine = HybridSearchEngine(["drip coffee maker", "memory foam dog bed"])
    results = engine.search("dog bed", limit=2)
    assert results[0][0] == 1
    assert all(score > 0 for _, score in results)


async def test_crosscart_backend_is_a_mock_retail_backend() -> None:
    backend = CrossCartSearchBackend()
    results = await backend.search_products(None, "luggage", None, 5)
    ids = {product.product_id for product in results}
    assert {"AR-1801", "AR-1802"} <= ids
    details = await backend.get_product_details(None, "AR-1801")
    assert details is not None and details.product_id == "AR-1801"


async def test_supervisor_rule_routes_and_parallel_search() -> None:
    backend = CrossCartSearchBackend()
    clients = [
        PlatformSearchClient(backend, "north", delay_ms=1),
        PlatformSearchClient(backend, "europe", delay_ms=2),
        PlatformSearchClient(backend, "asia", delay_ms=3),
    ]
    search_worker = SearchWorker(clients)
    supervisor = CrossCartSupervisor(
        search_worker,
        ComparisonWorker(search_worker),
        client=None,
    )
    result = await supervisor.handle("compare luggage prices")
    assert result.route.worker == "comparison"
    assert result.comparison
    assert result.usage == {"input": 0, "output": 0}


async def test_baseline_and_crosscart_implement_same_contract() -> None:
    for backend in (MockRetail(), CrossCartSearchBackend()):
        results = await backend.search_products(None, "sleeping bag", None, 5)
        assert results and all(product.price > 0 for product in results)
