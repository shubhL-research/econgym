"""Validation tests for the first-price sealed-bid auction (CONTRACT §3.3).

The env is validated AGAINST its closed-form benchmark on three independent axes:

  * **Static exactness** -- ``env.equilibrium()`` returns the hand-derived closed
    form EXACTLY: ``bid_slope == (n-1)/n`` and ``expected_revenue == (n-1)/(n+1)``.
  * **Equilibrium correctness** -- the stored BNE ``b(v)=((n-1)/n)v`` is a mutual
    best response: over a grid of values, ``argmax_b U(b)`` (opponents at BNE)
    equals ``((n-1)/n)v`` within one bid-grid step, and no discrete deviation
    beats the BNE payoff by more than ``1e-6``.
  * **Realized physics** -- the ENV's own allocation is checked against an
    independent numpy reference on random inputs (``test_env_step_matches_reference``);
    the Monte-Carlo revenue check then simulates auctions THROUGH ``env.reset``/
    ``env.step`` and recovers the theoretical ``(n-1)/(n+1)`` within a stated band,
    plus a high-precision vectorized cross-check at ``N = 200_000``.

Every stochastic check seeds ``np.random.default_rng`` explicitly, so reruns are
bit-stable (CONTRACT §5).
"""
import numpy as np
import pytest

from econgym import FirstPriceAuctionEnv as _TopLevelFirstPrice  # export smoke
from econgym.envs.first_price import FirstPriceAuctionEnv, first_price_bne
from econgym.core import Box

N_VALUES = [2, 3, 5]


# ----------------------------------------------------------------------
# Spaces, seeding, and the one-shot contract.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_spaces_shape(n):
    env = FirstPriceAuctionEnv(n=n)
    assert env.n == n
    assert len(env.action_space) == n == len(env.observation_space)
    for sp in env.action_space + env.observation_space:
        assert isinstance(sp, Box)
        assert sp.low.min() == 0.0 and sp.high.max() == 1.0
    # optional discrete bid grid spans [0, 1]
    assert env.bid_grid[0] == 0.0 and env.bid_grid[-1] == 1.0
    assert env.to_bids([0, env.G - 1]).tolist() == [0.0, 1.0]


def test_export_from_package_root():
    """FirstPriceAuctionEnv is registered/exported at the package root."""
    assert _TopLevelFirstPrice is FirstPriceAuctionEnv


@pytest.mark.parametrize("n", N_VALUES)
def test_reset_reveals_own_values_and_is_seed_stable(n):
    # obs[i] is bidder i's own private value; two resets on the same seed agree.
    e1 = FirstPriceAuctionEnv(n=n)
    obs1 = e1.reset(rng=np.random.default_rng(7))
    e2 = FirstPriceAuctionEnv(n=n)
    obs2 = e2.reset(rng=np.random.default_rng(7))
    assert obs1 == obs2
    assert obs1 == [float(v) for v in e1._values]
    # reset consumes EXACTLY one uniform(0,1,size=n) draw from the shared stream.
    shared = np.random.default_rng(123)
    e3 = FirstPriceAuctionEnv(n=n)
    e3.reset(rng=shared)
    twin = np.random.default_rng(123)
    twin.uniform(0.0, 1.0, size=n)
    assert shared.random() == twin.random()
    assert e3.rng is shared


@pytest.mark.parametrize("n", N_VALUES)
def test_one_shot_terminates(n):
    env = FirstPriceAuctionEnv(n=n)
    env.reset(rng=np.random.default_rng(0))
    obs, rewards, terminated, truncated, info = env.step([0.1] * n)
    assert terminated is True and truncated is False
    assert isinstance(info, dict)
    for k in ("values", "bids", "winner", "price", "revenue", "highest_bid"):
        assert k in info
    assert np.asarray(rewards).shape == (n,)


def test_step_before_reset_raises():
    env = FirstPriceAuctionEnv(n=2)
    with pytest.raises(RuntimeError):
        env.step([0.5, 0.5])


# ----------------------------------------------------------------------
# Realized physics: env.step allocation == independent numpy reference.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_env_step_matches_reference(n):
    """First-price physics: the highest bidder wins, pays its OWN bid, earns
    ``v-b``; everyone else earns 0; seller revenue == winning bid. Checked on
    random (value, bid) profiles against a from-scratch reference."""
    rng = np.random.default_rng(2024)
    env = FirstPriceAuctionEnv(n=n)
    for _ in range(2000):
        obs = env.reset(rng=rng)
        values = np.asarray(obs, dtype=float)
        bids = rng.uniform(0.0, 1.0, size=n)        # continuous -> ties measure-zero
        _, rewards, term, trunc, info = env.step(bids)
        rewards = np.asarray(rewards, dtype=float)

        winner = int(np.argmax(bids))               # unique w.p. 1
        assert info["winner"] == winner
        assert term is True and trunc is False
        assert info["revenue"] == pytest.approx(bids[winner])     # pay-your-bid
        assert info["highest_bid"] == pytest.approx(bids[winner])
        # winner's payoff and all-losers-zero
        assert rewards[winner] == pytest.approx(values[winner] - bids[winner])
        mask = np.ones(n, dtype=bool)
        mask[winner] = False
        assert np.all(rewards[mask] == 0.0)


def test_tie_is_broken_to_a_single_top_bidder():
    """An exact tie for the top bid awards the item to exactly one tied bidder
    (uniformly at random); revenue is the tied bid regardless of who wins."""
    env = FirstPriceAuctionEnv(n=3)
    env.reset(rng=np.random.default_rng(1))
    _, rewards, _, _, info = env.step([0.4, 0.4, 0.1])   # bidders 0 and 1 tie at top
    rewards = np.asarray(rewards)
    assert info["winner"] in (0, 1)                       # never the low bidder
    assert info["revenue"] == pytest.approx(0.4)
    assert np.count_nonzero(rewards) <= 1                 # at most one winner paid


# ----------------------------------------------------------------------
# Static benchmark exactness (CONTRACT §3.3 test 3).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_static_matches_closed_form(n):
    env = FirstPriceAuctionEnv(n=n)
    eq = env.equilibrium()
    slope = (n - 1) / n
    exp_rev = (n - 1) / (n + 1)
    # exact (float-representable rationals)
    assert eq["bid_slope"] == slope
    assert eq["expected_revenue"] == exp_rev
    # bid function is b(v) = slope * v
    vs = np.linspace(0.0, 1.0, 11)
    assert np.allclose(eq["bid_fn"](vs), slope * vs)
    # single source of truth: env delegates to first_price_bne
    sol = first_price_bne(n)
    assert sol["bid_slope"] == eq["bid_slope"]
    assert sol["expected_revenue"] == eq["expected_revenue"]
    # benchmark() alias returns the same numbers
    bm = env.benchmark()
    assert bm["bid_slope"] == eq["bid_slope"]
    assert bm["expected_revenue"] == eq["expected_revenue"]


# ----------------------------------------------------------------------
# Equilibrium correctness: BNE is a mutual best response (CONTRACT §3.3 test 1).
# ----------------------------------------------------------------------
def _expected_payoff(b, v, n):
    """U(b) for a bidder with value ``v`` bidding ``b`` when the other n-1 bidders
    play the BNE ``b(v)=((n-1)/n)v`` with iid ``U[0,1]`` values.

    Win iff every opponent value ``v_j < b*n/(n-1)`` -> win prob
    ``clip(b*n/(n-1), 0, 1)**(n-1)``.
    """
    inv_slope = n / (n - 1)                     # = 1 / ((n-1)/n)
    win_prob = np.clip(b * inv_slope, 0.0, 1.0) ** (n - 1)
    return (v - b) * win_prob


@pytest.mark.parametrize("n", N_VALUES)
def test_bne_is_mutual_best_response(n):
    slope = (n - 1) / n
    b_grid = np.linspace(0.0, 1.0, 4001)        # step = 2.5e-4
    step = b_grid[1] - b_grid[0]
    for v in np.linspace(0.05, 0.95, 19):
        U = _expected_payoff(b_grid, v, n)
        b_star_grid = b_grid[int(np.argmax(U))]
        # (a) grid argmax lands within one grid step of the closed-form BNE bid
        assert abs(b_star_grid - slope * v) <= 1.5 * step
        # (b) no discrete deviation beats the exact BNE payoff by more than 1e-6
        u_bne = _expected_payoff(slope * v, v, n)
        assert U.max() <= u_bne + 1e-6


# ----------------------------------------------------------------------
# Realized revenue converges to the closed form (CONTRACT §3.3 test 2).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_simulated_revenue_through_env(n):
    """End-to-end: play N auctions with EVERY bidder at the BNE, driving the real
    ``env.reset``/``env.step`` physics off one shared rng. Mean seller revenue
    (the winner's paid bid, read from ``info``) must land near ``(n-1)/(n+1)``."""
    env = FirstPriceAuctionEnv(n=n)
    rng = np.random.default_rng(20240722)
    slope = env.equilibrium()["bid_slope"]
    N = 40_000
    total = 0.0
    for _ in range(N):
        values = np.asarray(env.reset(rng=rng), dtype=float)
        bids = slope * values                     # all bidders play the BNE
        _, _, _, _, info = env.step(bids)
        total += info["revenue"]
    mean_rev = total / N
    target = (n - 1) / (n + 1)
    band = 4.0 / np.sqrt(N)                        # ~0.02; the true mean-error is far tighter
    assert abs(mean_rev - target) < band


@pytest.mark.parametrize("n", N_VALUES)
def test_simulated_revenue_vectorized_highN(n):
    """High-precision cross-check at N=200_000 (vectorized). Revenue == winning
    bid == slope * max value; its mean converges to ``(n-1)/(n+1)``."""
    rng = np.random.default_rng(999)
    slope = (n - 1) / n
    N = 200_000
    V = rng.uniform(0.0, 1.0, size=(N, n))
    revenue = slope * V.max(axis=1)               # first-price winner pays own (top) bid
    mean_rev = revenue.mean()
    target = (n - 1) / (n + 1)
    assert abs(mean_rev - target) < 4.0 / np.sqrt(N)   # ~0.009 band
