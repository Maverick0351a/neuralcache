import numpy as np
from neuralcache.narrative import NarrativeTracker


def test_narrative_success_gate_blocks_and_allows():
    nt = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.6, backend="memory")
    original = nt.v.copy()
    emb = np.array([1.0,0.0,0.0,0.0], dtype=np.float32)
    # Below gate: ignored
    nt.update(emb, success=0.5)
    assert np.allclose(nt.v, original)
    # At gate threshold: allow (>= gate)
    nt.update(emb, success=0.6)
    assert not np.allclose(nt.v, original)
