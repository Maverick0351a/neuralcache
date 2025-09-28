from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document as LCDocument
else:  # pragma: no cover - optional dependency
    LCDocument = Any
    try:
        from langchain_core.documents import Document as _LCDocument
    except Exception:
        pass
    else:
        LCDocument = _LCDocument

from ..config import Settings
from ..rerank import Reranker
from ..types import Document as NC_Document


class NeuralCacheLangChainReranker:
    """Adapter that makes :class:`neuralcache.rerank.Reranker` usable in LangChain.

    The adapter is tolerant of LangChain not being installed—if the import fails it
    gracefully falls back to ``typing.Any`` records so that downstream users can
    still instantiate and exercise the class in environments where LangChain is an
    optional extra.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.reranker = Reranker(self.settings)

    def __call__(self, query: str, documents: Sequence[LCDocument]) -> list[LCDocument]:
        """Return the documents ordered by NeuralCache relevance."""

        nc_docs = self._convert_documents(documents)
        query_embedding = self.reranker.encode_query(query)
        scored = self.reranker.score(query_embedding, nc_docs, query_text=query)
        return [documents[int(sd.id)] for sd in scored]

    def _convert_documents(self, documents: Sequence[LCDocument]) -> list[NC_Document]:
        if not isinstance(documents, Iterable):  # pragma: no cover - defensive
            raise TypeError("documents must be an iterable of LangChain Document objects")

        nc_docs = [
            NC_Document(
                id=str(index),
                text=getattr(doc, "page_content", ""),
                metadata=getattr(doc, "metadata", {}) or {},
            )
            for index, doc in enumerate(documents)
        ]
        if len(nc_docs) > self.settings.max_documents:
            raise ValueError(
                "NeuralCache received "
                f"{len(nc_docs)} documents, exceeding "
                f"max_documents={self.settings.max_documents}"
            )
        for doc in nc_docs:
            if len(doc.text) > self.settings.max_text_length:
                raise ValueError(
                    "Document "
                    f"{doc.id} text length exceeds "
                    f"max_text_length={self.settings.max_text_length}"
                )
        return nc_docs
