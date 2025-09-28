from __future__ import annotations

import pytest

from neuralcache.adapters import (
    NeuralCacheLangChainReranker,
    NeuralCacheLlamaIndexReranker,
)
from neuralcache.adapters.langchain_adapter import LCDocument  # type: ignore
from neuralcache.adapters.llamaindex_adapter import NodeWithScore
from neuralcache.config import Settings


class _FakeLangChainDocument:
    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class _FakeNode:
    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self._text = text
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self._text


def test_langchain_adapter_returns_same_doc_instances() -> None:
    adapter = NeuralCacheLangChainReranker(Settings(max_documents=16))
    docs: list[LCDocument] = [
        _FakeLangChainDocument("Stigmergy enables coordination"),
        _FakeLangChainDocument("Vector stores power retrieval"),
    ]

    ranked = adapter("What is stigmergy?", docs)

    assert len(ranked) == len(docs)
    assert all(doc in docs for doc in ranked)


def test_langchain_adapter_respects_max_documents_limit() -> None:
    adapter = NeuralCacheLangChainReranker(Settings(max_documents=1))
    docs: list[LCDocument] = [
        _FakeLangChainDocument("First"),
        _FakeLangChainDocument("Second"),
    ]

    with pytest.raises(ValueError):
        adapter("overflow", docs)


def test_llamaindex_adapter_returns_nodes() -> None:
    adapter = NeuralCacheLlamaIndexReranker(Settings(max_documents=16))
    nodes = [
        NodeWithScore(node=_FakeNode("Context A"), score=0.0),
        NodeWithScore(node=_FakeNode("Context B"), score=0.0),
    ]

    ranked = adapter.postprocess_nodes(nodes, query_str="match")

    assert len(ranked) == len(nodes)
    assert all(isinstance(item, NodeWithScore) for item in ranked)


def test_llamaindex_adapter_enforces_max_documents() -> None:
    adapter = NeuralCacheLlamaIndexReranker(Settings(max_documents=1))
    nodes = [
        NodeWithScore(node=_FakeNode("Context A"), score=0.0),
        NodeWithScore(node=_FakeNode("Context B"), score=0.0),
    ]

    with pytest.raises(ValueError):
        adapter.postprocess_nodes(nodes, query_str="overflow")
