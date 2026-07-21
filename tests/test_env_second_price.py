"""Validation of the second-price (Vickrey) sealed-bid auction against its
closed-form benchmark (CONTRACT_v1.md §3.4).

The benchmark being validated:

  * Truthful bidding ``b(v) = v`` is a WEAKLY DOMINANT strategy -- stronger than a
    Bayes-Nash equilibrium. Because a bidder's payoff depends on the opponents
    only through ``m = max opponent bid`` (win and pay ``m`` iff own bid ``> m``),
    the dominance reduces to a check on the plane ``(own value v, own bid b,
    opponent max m)`` that certifies EVERY opponent profile for EVERY ``n``.
  * Expected seller revenue ``E[R] = (n-1)/(n+1)`` = mean of the second-highest of
    ``n`` iid ``U[0,1]`` draws (the second order statistic), matching the
    first-price auction by revenue equivalence.

Every stochastic check seeds the shared ``np.random.default_rng`` explicitly, so
reruns are bit-stable (CONTRACT §5).
"""
import numpy as np
import pytest

from econgym import SecondPriceEnv
from econgym.solvers import closed_form


# ----------------------------------------------------------------------
# Reduced-form auction payoff: value ``v``, own bid ``b``, opponent max ``m``.
# In a second-price auction the winner holds the top bid and pays the highest
# OPPONENT bid, so a bidder's payoff depends on opponents only through ``m``.
# ----------------------------------------------------------------------
def _payoff(v, b, m):
    """Second-price payoff to a bidder with value ``v`` bidding ``b`` when the
    highest opposing bid is ``m``. Win (pay ``m``) iff ``b > m``; lose iff
    ``b < m``; on an exact tie the item splits, expected surplus ``0.5*(v-m)``."""
    win = b > m
    tie = b == m
    p = np.where(win, v - m, 0.0)
    p = np.where(tie, 0.5 * (v - m), p)
    return p


# ----------------------------------------------------------------------
# 1. Truthful bidding is WEAKLY DOMINANT (the dominance, not merely BNE, check).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_truthful_weak_dominance(n):
    """Over a grid of own value ``v``, alternative bids ``b'`` and opponent
    max-bids ``m``, truthful payoff >= deviation payoff - 1e-12 in EVERY cell.

    The check is ``n``-independent by construction (payoff depends on opponents
    only through ``m``, whose grid spans all possible opponent maxima for any
    ``n``); it is parametrized over ``n`` to honor the contract. Sharing the
    grid between ``b'`` and ``m`` exercises the exact-tie boundary ``b' == m``.
    """
    v = np.linspace(0.0, 1.0, 21)[:, None, None]        # own value
    b = np.linspace(0.0, 1.0, 41)[None, :, None]        # alternative (deviation) bid
    m = np.linspace(0.0, 1.0, 41)[None, None, :]        # opponent max bid

    truthful = _payoff(v, v, m)                          # own bid == own value
    deviation = _payoff(v, b, m)
    # truthful must weakly dominate every alternative bid in every (v, m) cell
    assert np.all(truthful >= deviation - 1e-12)
    # and it must STRICTLY beat some deviation somewhere (dominance is not vacuous):
    assert np.any(truthful > deviation + 1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_env_no_profitable_deviation_random_opponents(n):
    """Grounds the dominance result in the actual env: for random own values and
    random opponent bids, no bid on a fine grid beats truthful by > 1e-12, using
    the env's own ``step`` to score every candidate."""
    rng = np.random.default_rng(20260722 + n)
    env = SecondPriceEnv(n=n)
    bid_grid = np.linspace(0.0, 1.0, 51)
    for _ in range(200):
        env.reset(rng=rng)
        values = env._values.copy()
        i = int(rng.integers(n))                        # the bidder we test
        opp = rng.uniform(0.0, 1.0, size=n)             # random opponent bids
        v_i = float(values[i])

        # truthful payoff for bidder i via the env
        bids = opp.copy(); bids[i] = v_i
        _, r_truth, term, _, _ = env.step(bids)
        assert term is True
        truth = float(r_truth[i])

        # best achievable deviation for bidder i (env-scored), holding opponents fixed
        best_dev = -np.inf
        m = float(np.delete(opp, i).max())
        for b in bid_grid:
            bids_d = opp.copy(); bids_d[i] = b
            _, r_d, _, _, _ = env.step(bids_d)
            best_dev = max(best_dev, float(r_d[i]))
        assert truth >= best_dev - 1e-12
        # cross-check env payoff equals the reduced-form payoff v_i - m (or 0)
        assert truth == pytest.approx(_payoff(v_i, v_i, m))


# ----------------------------------------------------------------------
# 2. Simulated revenue converges to (n-1)/(n+1).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_simulated_revenue_converges(n):
    """Simulate N=200_000 truthful auctions from the shared rng; the mean seller
    revenue (the price the winner pays = second-highest bid) is within
    ``4/sqrt(N)`` of ``(n-1)/(n+1)``."""
    N = 200_000
    rng = np.random.default_rng(12345)
    values = rng.uniform(0.0, 1.0, size=(N, n))         # truthful: bid == value
    revenue = np.sort(values, axis=1)[:, -2]            # 2nd-highest = price paid
    mean_rev = float(revenue.mean())
    target = (n - 1) / (n + 1)
    assert mean_rev == pytest.approx(target, abs=4.0 / np.sqrt(N))


def test_env_step_matches_second_order_statistic():
    """Tie the statistical formula back to the ENV: over many random draws, the
    env's ``info['price']`` equals the second-highest bid and the winner's reward
    equals ``v_winner - price`` (physics faithfulness, not just a re-derivation)."""
    rng = np.random.default_rng(7)
    for n in (2, 3, 5):
        env = SecondPriceEnv(n=n)
        for _ in range(2000):
            env.reset(rng=rng)
            values = env._values.copy()
            bids = values.copy()                        # truthful
            obs, rewards, term, trunc, info = env.step(bids)
            assert term is True and trunc is False
            expected_price = float(np.sort(bids)[-2])
            expected_winner = int(np.argmax(bids))
            assert info["price"] == pytest.approx(expected_price)
            assert info["winner"] == expected_winner
            # winner reward = value - price; everyone else exactly 0
            assert rewards[expected_winner] == pytest.approx(
                values[expected_winner] - expected_price)
            losers = [j for j in range(n) if j != expected_winner]
            assert np.all(rewards[losers] == 0.0)


# ----------------------------------------------------------------------
# 3. Revenue equivalence with the first-price auction.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_revenue_equivalence(n):
    """Second-price expected revenue == (n-1)/(n+1); and, when the first-price
    closed form is available, == first-price expected revenue (revenue
    equivalence). The analytic identity is always asserted; the cross-check is
    exercised whenever the sibling solver exists."""
    sp = SecondPriceEnv(n=n).equilibrium()["expected_revenue"]
    assert sp == pytest.approx((n - 1) / (n + 1), rel=0, abs=0)
    fp_fn = getattr(closed_form, "first_price_bne", None)
    if fp_fn is not None:                               # sibling env built too
        assert fp_fn(n)["expected_revenue"] == pytest.approx(sp)


# ----------------------------------------------------------------------
# Static benchmark: equilibrium() equals the closed form and delegates to solver.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_static_matches_closed_form(n):
    env = SecondPriceEnv(n=n)
    eq = env.equilibrium()
    assert eq["dominant"] is True
    assert eq["bid_slope"] == 1.0
    assert eq["expected_revenue"] == pytest.approx((n - 1) / (n + 1))
    # bid function is truthful (identity) on a grid
    for v in np.linspace(0.0, 1.0, 11):
        assert eq["bid_fn"](v) == v
    # equilibrium() delegates to the solver (single source of truth): same numbers
    solver = closed_form.second_price_bne(n)
    assert eq["bid_slope"] == solver["bid_slope"]
    assert eq["dominant"] == solver["dominant"]
    assert eq["expected_revenue"] == solver["expected_revenue"]
    # benchmark() alias returns an equal dict (numbers + flags)
    bench = env.benchmark()
    assert bench["bid_slope"] == eq["bid_slope"]
    assert bench["expected_revenue"] == eq["expected_revenue"]


# ----------------------------------------------------------------------
# Seeding contract + one-shot API surface.
# ----------------------------------------------------------------------
def test_reset_reveals_values_from_shared_rng():
    """``reset(rng=shared)`` draws exactly one ``uniform(0,1,size=n)`` from the
    shared stream and returns those values as each bidder's observation."""
    env = SecondPriceEnv(n=4)
    shared = np.random.default_rng(99)
    obs = env.reset(rng=shared)
    twin = np.random.default_rng(99).uniform(0.0, 1.0, size=4)
    assert obs == [float(x) for x in twin]
    # after the single env draw, the shared stream is advanced by exactly that draw
    s2 = np.random.default_rng(99); s2.uniform(0.0, 1.0, size=4)
    assert env.rng.random() == s2.random()


def test_one_shot_terminates_and_info_is_dict():
    env = SecondPriceEnv(n=3)
    env.reset(seed=1)
    out = env.step([0.2, 0.9, 0.5])
    assert len(out) == 5
    obs, rewards, terminated, truncated, info = out
    assert terminated is True and truncated is False
    assert isinstance(info, dict)
    for k in ("winner", "price", "bids", "values"):
        assert k in info
    assert len(rewards) == 3


def test_tie_splits_and_winner_pays_top():
    """Exact top-bid tie: the price paid equals the tied top bid (second-highest
    == highest), so the winner's surplus is ``v_winner - top``; the winner is one
    of the tied bidders and is chosen via the shared rng."""
    env = SecondPriceEnv(n=3)
    env.reset(seed=0)
    env._values = np.array([0.8, 0.8, 0.1])            # force a known configuration
    obs, rewards, term, trunc, info = env.step([0.6, 0.6, 0.1])   # bidders 0,1 tie at top
    assert info["price"] == pytest.approx(0.6)          # pays the tied top bid
    assert info["winner"] in (0, 1)
    w = info["winner"]
    assert rewards[w] == pytest.approx(env._values[w] - 0.6)
    assert rewards[2] == 0.0
