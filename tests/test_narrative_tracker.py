import numpy as np
import time
from neuralcache.narrative import NarrativeTracker


def test_narrative_update_and_coherence():
    nt = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.0, backend="memory")
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    nt.update(emb, success=1.0)
    sims = nt.coherence(np.stack([emb, np.array([0.0,1.0,0.0,0.0], dtype=np.float32)]))
    assert sims.shape == (2,)
    assert sims[0] >= sims[1]


def test_narrative_purge_if_stale():
    nt = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.0, backend="memory")
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    nt.update(emb, success=1.0)
    # Force timestamp into past
    nt._updated_ts -= 100.0
    nt.purge_if_stale(retention_seconds=10.0)
    assert np.allclose(nt.v, np.zeros_like(nt.v))
