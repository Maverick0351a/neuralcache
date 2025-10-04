from neuralcache.metrics.text import lexical_overlap, context_used


def test_lexical_overlap_basic():
    a = "Alpha beta gamma"
    b = "beta Gamma delta"  # case + one overlapping token set {beta, gamma}
    ov = lexical_overlap(a, b)
    # tokens_a={alpha,beta,gamma}, tokens_b={beta,gamma,delta}; inter=2, union=4 => 0.5
    assert abs(ov - 0.5) < 1e-6


def test_lexical_overlap_empty():
    assert lexical_overlap("", "nonempty") == 0.0
    assert lexical_overlap("nonempty", "") == 0.0


def test_context_used_thresholding():
    answer = "api keys rotate via dashboard"
    ctx = [
        "Rotate API keys in dashboard settings",  # high overlap
        "Completely unrelated phrase",            # low overlap
    ]
    used = context_used(answer, ctx, threshold=0.2)
    assert used == [True, False]
