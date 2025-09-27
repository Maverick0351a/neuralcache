from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.query_bundle import QueryBundle
    from llama_index.core.schema import NodeWithScore
else:  # pragma: no cover - optional dependency
    try:
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.query_bundle import QueryBundle
        from llama_index.core.schema import NodeWithScore
    except Exception:

        class BaseNodePostprocessor:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Fallback base class when LlamaIndex is unavailable."""

        @dataclass(slots=True)
        class NodeWithScore:
            node: Any
            score: float = 0.0

        QueryBundle = Any

from ..config import Settings
from ..rerank import Reranker
from ..types import Document as NC_Document


class NeuralCacheLlamaIndexReranker(BaseNodePostprocessor):
    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or Settings()
        self.reranker = Reranker(self.settings)

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        query = query_str or (query_bundle.query_str if query_bundle else "")
        nc_docs = [
            NC_Document(
                id=str(index),
                text=node_with_score.node.get_content(),
                metadata=getattr(node_with_score.node, "metadata", {}) or {},
            )
            for index, node_with_score in enumerate(nodes)
        ]
        if len(nc_docs) > self.settings.max_documents:
            raise ValueError(
                "NeuralCache received "
                f"{len(nc_docs)} nodes, exceeding "
                f"max_documents={self.settings.max_documents}"
            )
        for doc in nc_docs:
            if len(doc.text) > self.settings.max_text_length:
                raise ValueError(
                    "Document "
                    f"{doc.id} text length exceeds "
                    f"max_text_length={self.settings.max_text_length}"
                )
        q = self.reranker.encode_query(query)

        scored = self.reranker.score(q, nc_docs)
        return [NodeWithScore(node=nodes[int(sd.id)].node, score=sd.score) for sd in scored]
