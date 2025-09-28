import json

import numpy as np

from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document

docs = [
    Document(
        id="d1",
        text=("Neural networks use layers of artificial neurons to learn patterns."),
    ),
    Document(
        id="d2",
        text=("The Battle of Gettysburg was fought in 1863 during the American Civil War."),
    ),
    Document(
        id="d3",
        text=(
            "Vector databases store embeddings for similarity "
            "search and retrieval augmented generation."
        ),
    ),
    Document(
        id="d4",
        text=("Stigmergy is a mechanism of indirect coordination through the environment."),
    ),
]

settings = Settings()
reranker = Reranker(settings=settings)

query = "What is stigmergy in AI systems?"
q = np.zeros((settings.narrative_dim,), dtype=float)
for tok in query.lower().split():
    q[hash(tok) % settings.narrative_dim] += 1.0

scored = reranker.score(q, docs, query_text=query)
print(json.dumps([s.model_dump() for s in scored], indent=2))
