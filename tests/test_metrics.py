"""Unit tests for metrics -- hand values plus a byte-match cross-check against
the ORIGINAL Paper-1 code on a couple of seeds (BOTH the Q-learning and the
mean-based paths).

The cross-check imports the original ``simulation.py`` only to VERIFY faithful
reproduction; the package itself never depends on it. If the original source is
not present, those checks are skipped (the reproduce test remains binding). The
``_randargmax`` tie-break RNG contract is ALSO pinned by a source-independent
unit test below, so the mean-based faithfulness invariant is enforced even when
the original tree is absent.
"""
import math
import os
import sys

import numpy as np
import pytest

from econgym import BertrandEnv, QLearner, MeanBased, run_episode, metrics
from econgym.agents.meanbased import _randargmax


# ----------------------------------------------------------------------
# Hand values
# ----------------------------------------------------------------------
def test_entropy_uniform_is_full():
    # a perfectly uniform column over K bins has raw entropy log2(K) bits
    # (plus a small Miller-Madow term) and normalised entropy >= 1.
    K = 8
    col = np.tile(np.arange(K), 100)   # each bin equally frequent
    raw, norm = metrics.entropy_bits(col, K, miller_madow=False)
    assert raw == pytest.approx(math.log2(K))
    assert norm == pytest.approx(1.0)


def test_entropy_degenerate_is_zero():
    K = 8
    col = np.zeros(400, dtype=np.int64)   # always the same price
    raw, norm = metrics.entropy_bits(col, K, miller_madow=False)
    assert raw == pytest.approx(0.0)
    assert norm == pytest.approx(0.0)


def test_miller_madow_adds_positive_bias():
    K = 8
    col = np.tile(np.arange(K), 100)
    raw_plain, _ = metrics.entropy_bits(col, K, miller_madow=False)
    raw_mm, _ = metrics.entropy_bits(col, K, miller_madow=True)
    nobs = len(col)
    expected = raw_plain + (K - 1) / (2.0 * nobs) / math.log(2)
    assert raw_mm == pytest.approx(expected)


def test_delta_index_bounds():
    # profit == monopoly -> Delta == 1 ; profit == nash -> Delta == 0.
    env = BertrandEnv(n=2, K=21, c=1.0, p_max=10.0)
    grid, c, n, T0 = env.grid, env.c, env.n, 10
    piM = env.monopoly_profit()   # 4.5
    prices = np.zeros((T0, 2), dtype=np.int64)
    profits = np.full((T0, 2), piM)
    assert metrics.delta_index(prices, profits, grid, c, n, T0) == pytest.approx(1.0)
    profits0 = np.full((T0, 2), env.nash_profit())  # 0.0
    assert metrics.delta_index(prices, profits0, grid, c, n, T0) == pytest.approx(0.0)


def test_regime_classifier():
    assert metrics.regime(0.60, 5.0, 1.0, 0.288, 2.0) == "Chaotic"
    assert metrics.regime(0.05, 5.0, 1.0, 0.288, 2.0) == "Collusive"
    assert metrics.regime(0.05, 1.5, 1.0, 0.288, 2.0) == "Competitive"


def test_is_converged_accepts_result_and_dict():
    assert metrics.is_converged({"pol_stable": [1.0, 1.0]}) is True
    assert metrics.is_converged({"pol_stable": [1.0, 0.5]}) is False
    assert metrics.is_converged([1.0, 0.995]) is True


# ----------------------------------------------------------------------
# Byte-match cross-check vs the ORIGINAL simulation.py (Q-learning path).
#
# The location of the original Paper-1 source is OVERRIDABLE via the
# ``ECONGYM_ORIGINAL_SRC`` environment variable (unset by default). Semantics:
#   * env var UNSET and default path absent -> skip (the vendored golden-trace
#     test in ``test_bytematch_golden.py`` keeps the faithfulness lock binding
#     everywhere with no external dependency).
#   * env var SET but the source is missing   -> HARD FAIL (never silently skip a
#     faithfulness check the operator explicitly asked to run).
# ----------------------------------------------------------------------
_ORIG = os.environ.get("ECONGYM_ORIGINAL_SRC", "")
_ORIG_REQUIRED = "ECONGYM_ORIGINAL_SRC" in os.environ


def _load_original():
    if not os.path.exists(os.path.join(_ORIG, "simulation.py")):
        if _ORIG_REQUIRED:
            raise AssertionError(
                f"ECONGYM_ORIGINAL_SRC is set to {_ORIG!r} but simulation.py is "
                "not there -- refusing to silently skip the faithfulness check."
            )
        return None
    if _ORIG not in sys.path:
        sys.path.insert(0, _ORIG)
    import simulation  # noqa
    return simulation


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_qlearning_bytematch_original(seed):
    sim = _load_original()
    if sim is None:
        pytest.skip("original simulation.py not available")
    K, T = 21, 3000
    grid = sim.make_grid(K, 0.0, 10.0)

    # original
    rng = np.random.default_rng(seed)
    orig = sim.run_qlearning(K, T, 0.10, 0.95, 0.10, 1.0, grid, rng,
                             eps_decay=3e-4, track_conv=True)

    # econgym
    env = BertrandEnv(n=2, K=K, c=1.0, p_min=0.0, p_max=10.0)
    agents = [QLearner(env, alpha=0.10, gamma=0.95, epsilon=0.10,
                       eps_decay=3e-4) for _ in range(2)]
    res = run_episode(env, agents, T, seed, track_conv=True)

    # Q-learning must be byte-for-byte identical in the price/profit traces.
    assert np.array_equal(res.prices, orig["price"])
    assert np.allclose(res.profits, orig["profit"], atol=0, rtol=0)

    # ... and therefore the metrics agree exactly.
    d_pkg = metrics.delta_index(res.prices, res.profits, grid, 1.0, 2, 2000)
    d_org = sim.delta_index(orig["price"], orig["profit"], grid, 1.0, 2, 2000)
    assert d_pkg == pytest.approx(d_org)
    h_pkg = metrics.mean_entropy(res.prices, K, 2000)[1]
    h_org = sim.mean_entropy(orig["price"], K, 2000)[1]
    assert h_pkg == pytest.approx(h_org)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_meanbased_bytematch_original_after_offset(seed):
    """Mean-based is BYTE-EXACT to the original AFTER one ``rng.integers(0,K,
    size=2)`` offset draw.

    The clean, n-general ``env.reset`` consumes exactly one init draw that the
    original ``run_meanbased`` lacked -- the single, documented RNG offset. Once
    that one draw is applied to a fresh Generator before calling the original,
    the two streams realign COMPLETELY: identical price indices (``array_equal``)
    and identical profits (``allclose``). This is a strict lock on the
    ``_randargmax`` randomized tie-break RNG contract: any change to the
    tie-break (e.g. dropping it for plain ``np.argmax``, or drawing when the max
    is NOT tied) would desynchronise the stream and fail here."""
    sim = _load_original()
    if sim is None:
        pytest.skip("original simulation.py not available")
    K, T = 21, 3000
    grid = sim.make_grid(K, 0.0, 10.0)

    # Original, run on a Generator advanced by the SAME one-draw offset the clean
    # env.reset introduces (rng.integers(0, K, size=n) for n=2).
    rng = np.random.default_rng(seed)
    rng.integers(0, K, size=2)
    orig = sim.run_meanbased(K, T, 0.10, 1.0, grid, rng,
                             eps_decay=3e-4, track_conv=True)

    env = BertrandEnv(n=2, K=K, c=1.0, p_min=0.0, p_max=10.0)
    agents = [MeanBased(env, epsilon=0.10, eps_decay=3e-4) for _ in range(2)]
    res = run_episode(env, agents, T, seed, track_conv=True)

    assert np.array_equal(res.prices, orig["price"]), "mean-based price trace differs from offset original"
    assert np.allclose(res.profits, orig["profit"], atol=0, rtol=0)

    # ... and therefore the metrics agree exactly.
    d_pkg = metrics.delta_index(res.prices, res.profits, grid, 1.0, 2, 2000)
    d_org = sim.delta_index(orig["price"], orig["profit"], grid, 1.0, 2, 2000)
    assert d_pkg == pytest.approx(d_org)


# ----------------------------------------------------------------------
# Source-INDEPENDENT lock on the _randargmax tie-break RNG contract.
# (Enforces the mean-based faithfulness invariant even with no original tree.)
# ----------------------------------------------------------------------
def test_randargmax_draws_only_on_tie():
    """``_randargmax`` consumes ONE ``rng.integers`` iff the max is tied, and
    NONE when the argmax is unique -- and it selects among the tied candidates
    using exactly that draw."""
    # unique max -> no rng consumed, returns the unique argmax
    r1 = np.random.default_rng(0)
    r2 = np.random.default_rng(0)
    row_unique = np.array([0.0, 5.0, 1.0, 2.0])
    assert _randargmax(row_unique, r1) == 1
    assert r1.random() == r2.random()          # r1's stream was NOT advanced

    # full tie over K -> exactly one rng.integers(K) draw picks the winner
    ra = np.random.default_rng(3)
    rb = np.random.default_rng(3)
    K = 5
    row_tie = np.zeros(K)
    pick = _randargmax(row_tie, ra)
    expected = int(rb.integers(K))             # reference draw from the twin rng
    assert pick == expected
    # both streams have now advanced by exactly one integers(K) draw -> aligned
    assert ra.random() == rb.random()


def test_meanbased_t0_full_tie_consumes_one_draw():
    """At t=0 the value table is all-zero, so an EXPLOIT step is a full tie and
    must consume exactly one ``rng.integers(K)`` via ``_randargmax``. Verify the
    action and the resulting stream position against a hand-built reference."""
    K = 21
    env = BertrandEnv(n=2, K=K, c=1.0, p_min=0.0, p_max=10.0)
    agent = MeanBased(env, epsilon=0.10, eps_decay=3e-4)

    rng = np.random.default_rng(123)
    agent.reset(rng)                            # consumes NO rng
    ref = np.random.default_rng(123)            # twin, kept in lock-step

    # Force the exploit branch: eps-test draw must be >= e. Replicate the exact
    # draw order the agent uses (rng.random(), then rng.integers(K) on the tie).
    e0 = 0.10 / (1.0 + 3e-4 * 0)
    u = ref.random()
    assert u >= e0, "pick a seed whose first uniform lands in the exploit branch"
    expected_action = int(ref.integers(K))      # the tie-break draw

    a = agent.act(obs=0, t=0, rng=rng)
    assert a == expected_action
    # agent consumed exactly one random() + one integers(K): streams stay aligned
    assert rng.random() == ref.random()
