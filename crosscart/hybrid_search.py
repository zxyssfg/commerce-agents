from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with a light plural fold, matching the fixture search."""
    raw = _WORD.findall(text.lower())
    return [token[:-1] if len(token) > 3 and token.endswith("s") else token for token in raw]


def char_ngrams(text: str, size: int = 4) -> list[str]:
    """Character n-grams give the vector leg robustness against word-form drift."""
    words = _WORD.findall(text.lower())
    joined = f" {' '.join(words)} "
    if len(joined) <= size:
        return [joined]
    return [joined[index : index + size] for index in range(len(joined) - size + 1)]


@dataclass(frozen=True)
class SparseVector:
    values: dict[str, float]
    norm: float


class HybridSearchEngine:
    """BM25 and sparse TF-IDF retrieval fused with Reciprocal Rank Fusion.

    This is intentionally dependency-free. BM25 is lexical, while TF-IDF over word and
    character n-gram features acts as a sparse vector leg. For production, replace the
    vector leg with a hosted embedding service; the RRF seam stays the same.
    """

    def __init__(
        self,
        documents: list[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        rrf_k: int = 60,
        bm25_weight: float = 0.65,
        vector_weight: float = 0.35,
        synonyms: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._synonyms = {key: list(values) for key, values in (synonyms or {}).items()}
        self._bm25_tokens = [tokenize(document) for document in documents]
        self._doc_len = [len(tokens) or 1 for tokens in self._bm25_tokens]
        self._avg_len = sum(self._doc_len) / len(self._doc_len)
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._bm25_tokens:
            self._doc_freq.update(set(tokens))
        self._idf = {
            term: math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
            for term, count in self._doc_freq.items()
        }

        # Fit one sparse vectorizer over word tokens plus character n-grams. Rare
        # catalog attributes matter more than frequent platform words such as "acme".
        feature_rows = [self._features(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for row in feature_rows:
            document_frequency.update(row)
        self._idf_features = {
            feature: math.log((len(documents) + 1) / (document_frequency[feature] + 1)) + 1
            for feature in document_frequency
        }
        self._vectors = [self._vectorize(row) for row in feature_rows]

    def _features(self, text: str) -> list[str]:
        return tokenize(text) + char_ngrams(text)

    def _vectorize(self, features: Iterable[str]) -> SparseVector:
        counts = Counter(features)
        values = {
            feature: (1 + math.log(count)) * self._idf_features[feature]
            for feature, count in counts.items()
            if feature in self._idf_features
        }
        norm = math.sqrt(sum(value * value for value in values.values()))
        return SparseVector(values, norm or 1.0)

    def _bm25_scores(self, query: str) -> list[float]:
        query_terms = tokenize(self._expanded_text(query))
        scores = []
        for index, doc_tokens in enumerate(self._bm25_tokens):
            term_counts = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_counts:
                    continue
                tf = term_counts[term]
                normalization = tf + self.k1 * (
                    1 - self.b + self.b * self._doc_len[index] / self._avg_len
                )
                score += self._idf.get(term, 0.0) * tf * (self.k1 + 1) / normalization
            scores.append(score)
        return scores

    def _vector_scores(self, query: str) -> list[float]:
        query_vector = self._vectorize(self._features(self._expanded_text(query)))
        if not query_vector.values:
            return [0.0] * len(self.documents)
        scores = []
        for vector in self._vectors:
            dot = sum(
                weight * vector.values.get(feature, 0.0)
                for feature, weight in query_vector.values.items()
            )
            scores.append(dot / (query_vector.norm * vector.norm))
        return scores

    def _expanded_text(self, query: str) -> str:
        query_tokens = tokenize(query)
        expanded = list(query_tokens)
        for token in query_tokens:
            expanded.extend(self._synonyms.get(token, []))
        return " ".join(expanded)

    @staticmethod
    def _rank_map(scores: list[float]) -> dict[int, int]:
        # RRF rewards documents retrieved by a leg; zero-scored documents do not get a
        # rank merely from stable document-index order.
        order = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )
        return {index: rank for rank, index in enumerate(order, start=1)}

    def search(self, query: str, *, limit: int = 8) -> list[tuple[int, float]]:
        """Return ``(document_index, fused_score)`` ranked best first."""
        bm25 = self._bm25_scores(query)
        vectors = self._vector_scores(query)
        bm25_ranks = self._rank_map(bm25)
        vector_ranks = self._rank_map(vectors)
        fused: dict[int, float] = {}
        for index in range(len(self.documents)):
            contributions = []
            if index in bm25_ranks:
                contributions.append(self.bm25_weight / (self.rrf_k + bm25_ranks[index]))
            if index in vector_ranks:
                contributions.append(self.vector_weight / (self.rrf_k + vector_ranks[index]))
            if contributions:
                fused[index] = sum(contributions)
        ranked = sorted(fused, key=lambda index: (-fused[index], index))
        return [(index, fused[index]) for index in ranked[:limit]]
