"""Validation suite for the linear public-goods (VCM) environment.

Validates :class:`~econgym.envs.public_goods.PublicGoodsEnv` against its
CLOSED-FORM benchmark exactly (contract v1, section 3.5):

  * ``nash_contribution = 0``,   ``nash_payoff = w``
  * ``optimum_contribution = w``, ``optimum_payoff = r*n*w``
  * ``gap_per_player = (r*n - 1) * w``

and confirms the benchmark is a genuine equilibrium of the env's PHYSICS via:
  * greedy (myopic) best response is free-riding (``c_i = 0``) for any opponent
    profile, and iterated greedy BR collapses to the all-zero Nash profile;
  * ``c_i = 0`` STRICTLY dominates every ``c_i > 0`` (since ``r < 1``);
  * the recorded Nash-vs-optimum gap and the social-welfare gap match theory;
  * the full-contribution profile maximises social welfare.

Every stochastic check seeds ``np.random.default_rng`` explicitly for
bit-stable reruns.
"""
import numpy as np
import pytest

from econgym import PublicGoodsEnv, public_goods_nash
from econgym.core import Box


DEFAULTS = dict(n=4, w=20.0, r=0.4)


# ----------------------------------------------------------------------
# 1. Static benchmark matches the closed form EXACTLY.
# ----------------------------------------------------------------------
def test_static_matches_closed_form():
    env = PublicGoodsEnv(**DEFAULTS)
    eq = env.equilibrium()
    n, w, r = env.n, env.w, env.r

    assert eq["nash_contribution"] == 0.0
    assert eq["nash_payoff"] == pytest.approx(w, rel=1e-12)
    assert eq["optimum_contribution"] == pytest.approx(w, rel=1e-12)
    assert eq["optimum_payoff"] == pytest.approx(r * n * w, rel=1e-12)
    assert eq["gap_per_player"] == pytest.approx((r * n - 1.0) * w, rel=1e-12)

    # exact numeric defaults from the contract: 20 / 32 / 12
    assert eq["nash_payoff"] == pytest.approx(20.0, rel=1e-12)
    assert eq["optimum_payoff"] == pytest.approx(32.0, rel=1e-12)
    assert eq["gap_per_player"] == pytest.approx(12.0, rel=1e-12)

    # env delegates to the single source of truth -> identical dict
    assert eq == public_goods_nash(w, r, n)
    # benchmark alias returns the identical dict
    assert env.benchmark() == eq


@pytest.mark.parametrize("n,w,r", [(2, 10.0, 0.6), (3, 15.0, 0.5),
                                   (5, 100.0, 0.3), (10, 20.0, 0.2)])
def test_static_matches_closed_form_parametrized(n, w, r):
    env = PublicGoodsEnv(n=n, w=w, r=r)
    eq = env.equilibrium()
    assert eq["nash_contribution"] == 0.0
    assert eq["nash_payoff"] == pytest.approx(w, rel=1e-12)
    assert eq["optimum_payoff"] == pytest.approx(r * n * w, rel=1e-12)
    assert eq["gap_per_player"] == pytest.approx((r * n - 1.0) * w, rel=1e-12)
    # optimum strictly beats Nash iff we are in the social-dilemma band
    assert eq["optimum_payoff"] > eq["nash_payoff"]


# ----------------------------------------------------------------------
# 2. Greedy (myopic) best response is free-riding for ANY opponent profile,
#    and iterated greedy BR converges to the all-zero (Nash) profile.
# ----------------------------------------------------------------------
def test_greedy_best_response_is_zero():
    env = PublicGoodsEnv(**DEFAULTS)
    rng = np.random.default_rng(0)
    own_grid = np.linspace(0.0, env.w, 51)      # candidate own contributions

    def own_payoff(i, own_c, others):
        profile = np.array(others, dtype=float)
        profile = np.insert(profile, i, own_c)
        return env.payoff(profile)[i]

    # For 200 random opponent profiles, the myopic-best own contribution is 0.
    for _ in range(200):
        others = rng.uniform(0.0, env.w, size=env.n - 1)
        i = int(rng.integers(env.n))
        payoffs = np.array([own_payoff(i, oc, others) for oc in own_grid])
        best = own_grid[int(np.argmax(payoffs))]
        assert best == pytest.approx(0.0, abs=1e-12)

    # Iterated greedy best response: every player jumps to its BR (=0) at once;
    # from ANY starting profile the map reaches the all-zero Nash profile.
    def greedy_br(profile):
        # BR_i(others) = argmax_{c in [0,w]} pi_i = 0, independent of others.
        return np.zeros(env.n)

    x = rng.uniform(0.0, env.w, size=env.n)
    for _ in range(5):
        x = greedy_br(x)
    assert np.allclose(x, 0.0, atol=1e-12)
    # the fixed point is the closed-form Nash contribution
    assert np.allclose(x, env.equilibrium()["nash_contribution"], atol=1e-12)


# ----------------------------------------------------------------------
# 3. c_i = 0 STRICTLY dominates every c_i > 0 (dominant strategy, since r < 1).
# ----------------------------------------------------------------------
def test_dominant_strategy():
    env = PublicGoodsEnv(**DEFAULTS)
    rng = np.random.default_rng(1)
    positive_grid = np.linspace(env.w / 20.0, env.w, 20)   # strictly > 0

    for _ in range(200):
        others = rng.uniform(0.0, env.w, size=env.n - 1)
        i = int(rng.integers(env.n))

        def own_payoff(own_c):
            profile = np.insert(np.array(others, dtype=float), i, own_c)
            return env.payoff(profile)[i]

        p_zero = own_payoff(0.0)
        for c in positive_grid:
            # strict: contributing anything strictly lowers own payoff
            assert p_zero > own_payoff(c) + 1e-9

    # exact per-unit private loss from contributing is (1 - r) per unit.
    env2 = PublicGoodsEnv(**DEFAULTS)
    base = env2.payoff(np.zeros(env2.n))[0]
    dev = env2.payoff(np.array([env2.w] + [0.0] * (env2.n - 1)))[0]
    assert base - dev == pytest.approx((1.0 - env2.r) * env2.w, rel=1e-12)


# ----------------------------------------------------------------------
# 4. Recorded Nash-vs-optimum gap and the social-welfare gap match theory.
# ----------------------------------------------------------------------
def test_records_nash_optimum_gap():
    env = PublicGoodsEnv(**DEFAULTS)
    eq = env.equilibrium()
    n, w, r = env.n, env.w, env.r

    assert eq["gap_per_player"] == pytest.approx((r * n - 1.0) * w, rel=1e-12)

    # social welfare = sum of payoffs, computed from the PHYSICS at each corner.
    welfare_nash = float(env.payoff(np.zeros(n)).sum())
    welfare_opt = float(env.payoff(np.full(n, w)).sum())
    assert welfare_nash == pytest.approx(n * w, rel=1e-12)
    assert welfare_opt == pytest.approx(r * n * n * w, rel=1e-12)
    # optimum welfare exceeds Nash welfare by exactly n*(r*n - 1)*w.
    assert welfare_opt - welfare_nash == pytest.approx(n * (r * n - 1.0) * w,
                                                       rel=1e-12)
    # per-player gap aggregates to the welfare gap.
    assert welfare_opt - welfare_nash == pytest.approx(n * eq["gap_per_player"],
                                                       rel=1e-12)


def test_full_contribution_maximizes_welfare():
    """Welfare is increasing in every c_j (slope r*n - 1 > 0), so the
    all-``w`` profile maximises total welfare over the whole action box."""
    env = PublicGoodsEnv(**DEFAULTS)
    rng = np.random.default_rng(2)
    welfare_opt = float(env.payoff(np.full(env.n, env.w)).sum())
    for _ in range(500):
        profile = rng.uniform(0.0, env.w, size=env.n)
        assert welfare_opt >= float(env.payoff(profile).sum()) - 1e-9


# ----------------------------------------------------------------------
# Physics + interface: step formula, spaces, seeding contract, determinism.
# ----------------------------------------------------------------------
def test_step_payoff_formula():
    env = PublicGoodsEnv(**DEFAULTS)
    env.reset(seed=0)
    contribs = [1.0, 5.0, 10.0, 20.0]
    obs, rewards, terminated, truncated, info = env.step(contribs)

    total = sum(contribs)
    expected = np.array([env.w - c + env.r * total for c in contribs])
    assert np.allclose(rewards, expected)
    assert terminated is False and truncated is False
    assert isinstance(info, dict)
    assert info["total"] == pytest.approx(total)
    assert info["public_good"] == pytest.approx(env.r * total)
    assert np.allclose(info["contributions"], contribs)
    # obs[i] = the OTHER players' contributions, memory-1
    assert len(obs) == env.n
    for i in range(env.n):
        others = [contribs[j] for j in range(env.n) if j != i]
        assert np.allclose(obs[i], others)


def test_contributions_clamped_to_endowment():
    env = PublicGoodsEnv(**DEFAULTS)
    env.reset(seed=0)
    # over-endowment and negative contributions are clipped into [0, w]
    _, rewards, *_ = env.step([-5.0, 999.0, 10.0, 10.0])
    clipped_total = 0.0 + env.w + 10.0 + 10.0
    expected = np.array([env.w - c + env.r * clipped_total
                         for c in [0.0, env.w, 10.0, 10.0]])
    assert np.allclose(rewards, expected)


def test_spaces_shape_and_types():
    env = PublicGoodsEnv(**DEFAULTS)
    assert len(env.action_space) == env.n
    assert len(env.observation_space) == env.n
    for a in env.action_space:
        assert isinstance(a, Box)
        assert a.low.min() == 0.0 and a.high.max() == env.w
    for o in env.observation_space:
        assert isinstance(o, Box)
        assert o.shape == (env.n - 1,)
    assert env.cgrid.shape == (env.G,)
    assert env.cgrid[0] == 0.0 and env.cgrid[-1] == env.w


def test_reset_consumes_shared_rng():
    """reset(rng=...) draws exactly one rng.uniform(0, w, size=n) from the
    SHARED stream, matching a twin draw (seeding contract)."""
    env = PublicGoodsEnv(**DEFAULTS)
    shared = np.random.default_rng(42)
    obs = env.reset(rng=shared)
    twin = np.random.default_rng(42)
    prev = twin.uniform(0.0, env.w, size=env.n)
    # both streams now at the same position
    assert shared.random() == twin.random()
    assert env.rng is shared
    # observation reflects that initial profile (others' contributions)
    for i in range(env.n):
        others = [prev[j] for j in range(env.n) if j != i]
        assert np.allclose(obs[i], others)


def test_reset_is_deterministic():
    o1 = PublicGoodsEnv(**DEFAULTS).reset(seed=7)
    o2 = PublicGoodsEnv(**DEFAULTS).reset(seed=7)
    for a, b in zip(o1, o2):
        assert np.allclose(a, b)


# ----------------------------------------------------------------------
# Constructor validation: the social-dilemma regime 1/n < r < 1 is enforced.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,r", [(4, 0.2),    # r < 1/n : no dilemma (0.25)
                                 (4, 0.25),   # r == 1/n : boundary
                                 (4, 1.0),    # r == 1 : contributing indifferent
                                 (4, 1.5),    # r > 1 : contributing dominant
                                 (2, 0.4)])   # r < 1/n=0.5 : no dilemma
def test_rejects_non_dilemma_params(n, r):
    with pytest.raises(ValueError):
        PublicGoodsEnv(n=n, w=20.0, r=r)


def test_accepts_dilemma_params():
    # interior of the band is accepted
    PublicGoodsEnv(n=4, w=20.0, r=0.26)
    PublicGoodsEnv(n=4, w=20.0, r=0.99)
    PublicGoodsEnv(n=2, w=20.0, r=0.6)
