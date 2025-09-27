from __future__ import annotations

import numpy as np

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore
except Exception:  # pragma: no cover - optional dependency
    BaseNodePostprocessor = object  # type: ignore

    class NodeWithScore:  # type: ignore
        def __init__(self, node, score: float = 0.0) -> None:
            self.node = node
            self.score = score

from ..config import Settings
from ..rerank import Reranker
from ..types import Document as NC_Document


class NeuralCacheLlamaIndexReranker(BaseNodePostprocessor):
    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or Settings()
        self.reranker = Reranker(self.settings)

    def postprocess_nodes(self, nodes: list[NodeWithScore], **kwargs) -> list[NodeWithScore]:
        query = kwargs.get("query_str", "")
        nc_docs = [
            NC_Document(
                id=str(index),
                text=node_with_score.node.get_content(),  # type: ignore[attr-defined]
                metadata=node_with_score.node.metadata or {},  # type: ignore[attr-defined]
            )
            for index, node_with_score in enumerate(nodes)
        ]
        dim = self.settings.narrative_dim
        q = np.zeros((dim,), dtype=np.float32)
        for tok in query.lower().split():
            q[hash(tok) % dim] += 1.0

        scored = self.reranker.score(q, nc_docs)
        return [
            NodeWithScore(node=nodes[int(sd.id)].node, score=sd.score)  # type: ignore[attr-defined]
            for sd in scored
        ]
