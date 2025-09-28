import numpy as np

from neuralcache.gating import compute_uncertainty, decide_candidate_count, make_decision


def test_uncertainty_bounds() -> None:
    # Peaked distribution -> low entropy
    sims = np.array([5.0, 0.1, 0.1, 0.1, 0.1])
    u = compute_uncertainty(sims, temp=1.0)
    assert 0.0 <= u <= 1.0
    assert u < 0.5

    # Uniform-like -> high entropy
    sims2 = np.array([1.0, 1.0, 1.0, 1.0])
    u2 = compute_uncertainty(sims2, temp=1.0)
    assert 0.0 <= u2 <= 1.0
    assert u2 > u


def test_decide_candidate_count_monotone() -> None:
    # As uncertainty rises, candidate_count should not decrease (in expectation)
    counts = []
    for u in np.linspace(0.0, 1.0, 11):
        c = decide_candidate_count(
            u,
            min_candidates=50,
            max_candidates=200,
            threshold=0.7,
            k=8.0,
        )
        counts.append(c)
    assert min(counts) >= 50
    assert max(counts) <= 200
    assert counts[0] <= counts[-1]


def test_make_decision_modes() -> None:
    sims = np.array([1.0, 1.0, 1.0, 1.0])
    # off
    d_off = make_decision(sims, mode="off", threshold=0.7, min_candidates=50, max_candidates=100)
    assert d_off.use_gating is False
    assert d_off.candidate_count == sims.size

    # on
    d_on = make_decision(sims, mode="on", threshold=0.7, min_candidates=50, max_candidates=100)
    assert d_on.use_gating is True
    assert 50 <= d_on.candidate_count <= 100

    # auto with high uncertainty -> gate
    d_auto = make_decision(sims, mode="auto", threshold=0.2, min_candidates=50, max_candidates=100)
    assert d_auto.use_gating is True

    # auto with low uncertainty -> no gate
    sims_peaked = np.array([5.0, 0.1, 0.1, 0.1])
    d_auto2 = make_decision(
        sims_peaked,
        mode="auto",
        threshold=0.9,
        min_candidates=50,
        max_candidates=100,
    )
    assert d_auto2.use_gating is False
