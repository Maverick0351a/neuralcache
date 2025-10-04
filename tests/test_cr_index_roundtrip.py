import json
from pathlib import Path

import numpy as np

from neuralcache.cr.index import build_cr_index, save_cr_index, load_cr_index


def test_cr_index_build_save_load_roundtrip(tmp_path: Path):
    # small deterministic embedding matrix
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 32)).astype(np.float32)

    idx = build_cr_index(embeddings, d1=16, d2=8, k2=5, k1_per_bucket=4, seed=123)
    npz_path = tmp_path / "cr_index.npz"
    meta_path = tmp_path / "cr_index.meta.json"
    save_cr_index(idx, str(npz_path), str(meta_path))

    # Basic meta file sanity
    meta = json.loads(meta_path.read_text())
    assert meta["doc_count"] == 20
    assert meta["d0"] == 32

    loaded = load_cr_index(str(npz_path), str(meta_path))
    # Compare core structural pieces
    assert loaded.meta.doc_count == idx.meta.doc_count
    assert loaded.coarse_centroids.shape == idx.coarse_centroids.shape
    assert len(loaded.coarse_buckets) == len(idx.coarse_buckets)
    # Ensure at least one bucket non-empty
    assert any(len(b) > 0 for b in loaded.coarse_buckets)
    # Dimension of projections should not exceed original dimensionality
    assert loaded.proj1_components.shape[1] <= embeddings.shape[1]
    assert loaded.proj2_components.shape[1] <= loaded.proj1_components.shape[1]
