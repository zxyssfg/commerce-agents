from __future__ import annotations

from demo_common.storefront_fixtures import summary_of, within_price_and_rating
from retail.api.mock_retail import _SYNONYMS, MockRetail

from .hybrid_search import HybridSearchEngine, tokenize


class CrossCartSearchBackend(MockRetail):
    """A drop-in replacement for ``MockRetail`` search that fuses BM25 and sparse vectors.

    The rest of ``StorefrontBackend`` (cart, orders, policies, details, gates) is
    inherited unchanged, so the production executor keeps the same contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self._product_list = list(self.products.values())
        documents = [self._hybrid_text(product) for product in self._product_list]
        self._engine = HybridSearchEngine(documents, synonyms=_SYNONYMS)

    def _hybrid_text(self, product) -> str:
        fields = self._searchable_text(product)
        return "\\n".join(
            f"{name}: {fields[name]}"
            for name in ("title", "brand", "category", "attributes", "description")
        )

    async def search_products(self, session, query, filters=None, limit=8):
        del session
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        candidates = [
            product
            for product in self._product_list
            if filters is None or within_price_and_rating(product, filters)
        ]
        if not candidates:
            return []
        candidate_ids = {id(product) for product in candidates}
        fused = self._engine.search(query, limit=max(limit * 3, limit))
        results = []
        for index, score in fused:
            product = self._product_list[index]
            if id(product) not in candidate_ids or score <= 0:
                continue
            if filters is not None and not self._soft_filter(product, filters):
                continue
            results.append((score, product))
            if len(results) >= limit:
                break
        if not results and filters is not None:
            # Preserve the baseline's behavior of relaxing over-narrow attribute filters.
            results = [(score, self._product_list[index]) for index, score in fused[:limit]]
        results.sort(key=lambda pair: (-pair[0], -(pair[1].rating or 0)))
        return [summary_of(product) for _, product in results[:limit]]
