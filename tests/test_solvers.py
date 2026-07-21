"""Validation tests for the shared ``econgym/solvers`` package (CONTRACT §1.5, §4.2).

The solvers are the learned-vs-optimal backbone: pure, agent-free routines that
compute or iterate to equilibria, and the primary oracle every environment's
``equilibrium()`` delegates to. Each test here validates a CLAIMED property
against a KNOWN answer -- never a self-referential re-check:

  * ``support_enumeration`` -- the 2-player normal-form Nash solver returns the
    textbook equilibria of the Prisoner's Dilemma (unique pure ``(D,D)``),
    matching pennies (unique ``(½,½),(½,½)``), a coordination game (two pure + one
    mixed), and Battle of the Sexes (two pure + the exact ``(2/3,1/3),(1/3,2/3)``
    mixed). Every returned profile is independently re-verified to be a genuine
    Nash equilibrium (no profitable pure deviation) directly from the payoff
    matrices -- so the solver is checked against the DEFINITION, not against
    itself.
  * ``best_response_iteration`` -- continuous best-response dynamics converge to
    the closed-form Nash of Cournot and differentiated Bertrand within tolerance.
  * ``fictitious_play`` -- discrete belief dynamics' time-average converges to the
    Nash of matching pennies (mixed ½,½) and the Prisoner's Dilemma (pure D).
  * ``closed_form.*`` -- each analytic solver equals its environment's
    ``equilibrium()`` EXACTLY (single source of truth), and the numbers match the
    hand-derived formulas.

Every stochastic input is seeded, so reruns are bit-stable (CONTRACT §5).
"""
import numpy as np
import pytest

import econgym
from econgym.solvers import (
    best_response_iteration,
    bertrand_diff_nash,
    cournot_nash,
    fictitious_play,
    first_price_bne,
    public_goods_nash,
    repeated_pd_threshold,
    rubinstein_split,
    second_price_bne,
    support_enumeration,
)
from econgym import (
    BertrandDiffEnv,
    CournotEnv,
    FirstPriceAuctionEnv,
    PublicGoodsEnv,
    RepeatedPDEnv,
    RubinsteinEnv,
    SecondPriceEnv,
)


# ======================================================================
# Helpers
# ======================================================================
def _is_nash(A, B, x, y, tol=1e-6):
    """True iff ``(x, y)`` is a Nash equilibrium of ``(A, B)`` -- checked directly
    from the DEFINITION: neither player can raise its expected payoff by switching
    to any pure action, given the opponent's mixed strategy."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u_row = x @ A @ y
    u_col = x @ B @ y
    best_row = (A @ y).max()          # best pure row response to y
    best_col = (x @ B).max()          # best pure column response to x
    return (best_row <= u_row + tol) and (best_col <= u_col + tol)


def _has_profile(eqs, x_target, y_target, atol=1e-6):
    """True iff some ``(x, y)`` in ``eqs`` matches the target profile."""
    return any(
        np.allclose(x, x_target, atol=atol) and np.allclose(y, y_target, atol=atol)
        for (x, y) in eqs
    )


# ======================================================================
# 1. support_enumeration on textbook games -> KNOWN Nash equilibria.
# ======================================================================
def test_support_enumeration_prisoners_dilemma():
    """Unique Nash of the PD is the pure profile (Defect, Defect)."""
    R, T, P, S = 3.0, 5.0, 1.0, 0.0
    A = np.array([[R, S], [T, P]])        # row payoff, [own][opp]
    B = A.T                                # symmetric game
    eqs = support_enumeration(A, B)
    assert len(eqs) == 1
    x, y = eqs[0]
    assert np.allclose(x, [0.0, 1.0])      # Defect
    assert np.allclose(y, [0.0, 1.0])
    assert _is_nash(A, B, x, y)


def test_support_enumeration_matching_pennies():
    """Zero-sum matching pennies has the unique mixed Nash (½,½),(½,½)."""
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    B = -A
    eqs = support_enumeration(A, B)
    assert len(eqs) == 1
    x, y = eqs[0]
    assert np.allclose(x, [0.5, 0.5])
    assert np.allclose(y, [0.5, 0.5])
    assert _is_nash(A, B, x, y)


def test_support_enumeration_coordination():
    """A pure coordination game has THREE equilibria: two pure + one mixed."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = A.copy()
    eqs = support_enumeration(A, B)
    assert len(eqs) == 3
    assert _has_profile(eqs, [1.0, 0.0], [1.0, 0.0])   # coordinate on action 0
    assert _has_profile(eqs, [0.0, 1.0], [0.0, 1.0])   # coordinate on action 1
    assert _has_profile(eqs, [0.5, 0.5], [0.5, 0.5])   # mixed
    for x, y in eqs:
        assert _is_nash(A, B, x, y)


def test_support_enumeration_battle_of_sexes():
    """Battle of the Sexes: two pure equilibria plus the exact mixed profile
    row=(2/3, 1/3), col=(1/3, 2/3) (the unique interior Nash)."""
    A = np.array([[2.0, 0.0], [0.0, 1.0]])   # row prefers the (0,0) meeting
    B = np.array([[1.0, 0.0], [0.0, 2.0]])   # col prefers the (1,1) meeting
    eqs = support_enumeration(A, B)
    assert len(eqs) == 3
    assert _has_profile(eqs, [1.0, 0.0], [1.0, 0.0])
    assert _has_profile(eqs, [0.0, 1.0], [0.0, 1.0])
    assert _has_profile(eqs, [2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0])
    for x, y in eqs:
        assert _is_nash(A, B, x, y)


def test_support_enumeration_matches_repeated_pd_env_stage_game():
    """Cross-check CONTRACT §3.6 test 1: the stage game built from the RepeatedPD
    env has the UNIQUE pure Nash (D, D)."""
    env = RepeatedPDEnv()                    # R=3,T=5,P=1,S=0
    A = env._payoff                          # own payoff, [own][opp]
    B = env._payoff.T
    eqs = support_enumeration(A, B)
    assert len(eqs) == 1
    x, y = eqs[0]
    assert np.allclose(x, [0.0, 1.0]) and np.allclose(y, [0.0, 1.0])   # (D, D)


def test_support_enumeration_rejects_bad_shapes():
    with pytest.raises(ValueError):
        support_enumeration(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        support_enumeration(np.zeros((2,)), np.zeros((2,)))


# ======================================================================
# 2. best_response_iteration -> closed-form Nash (continuous games).
# ======================================================================
def test_br_iteration_bertrand_diff_reaches_closed_form():
    """Differentiated-Bertrand BR map is a contraction (factor gamma(n-1)/2beta
    = 0.25 < 1 for defaults), so plain best-response iteration converges to the
    symmetric Nash price p* = (alpha+beta c)/(2beta-gamma(n-1)) = 4.0."""
    env = BertrandDiffEnv()                   # p* = 4.0
    p_star = env.equilibrium()["p"]
    assert p_star == pytest.approx(4.0)

    # from p = c on every firm
    x = best_response_iteration(env.best_response, np.full(env.n, env.c),
                                tol=1e-12, max_iter=100_000)
    assert np.allclose(x, p_star, atol=1e-9)
    # the returned point is a genuine fixed point of the BR map
    assert np.allclose(env.best_response(x), x, atol=1e-9)

    # from a random start >= c (deterministic seed)
    rng = np.random.default_rng(0)
    x0 = env.c + rng.uniform(0.0, 5.0, size=env.n)
    xr = best_response_iteration(env.best_response, x0, tol=1e-12, max_iter=100_000)
    assert np.allclose(xr, p_star, atol=1e-9)


@pytest.mark.parametrize("n,alpha,beta,gamma,c", [
    (2, 10.0, 2.0, 1.0, 1.0),
    (3, 12.0, 3.0, 1.0, 2.0),
    (2, 8.0, 4.0, 3.0, 0.5),
])
def test_br_iteration_bertrand_diff_parametrized(n, alpha, beta, gamma, c):
    env = BertrandDiffEnv(n=n, alpha=alpha, beta=beta, gamma=gamma, c=c)
    p_star = env.equilibrium()["p"]
    x = best_response_iteration(env.best_response, np.full(n, c),
                                tol=1e-13, max_iter=100_000)
    assert np.allclose(x, p_star, atol=1e-9)
    # solver output equals the closed form (single source of truth)
    assert bertrand_diff_nash(alpha, beta, gamma, c, n)["p"] == pytest.approx(p_star)


def test_br_iteration_cournot_n2_undamped():
    """The undamped simultaneous Cournot BR map is a contraction only at n=2
    (slope 1/2); plain iteration converges to q_i* there."""
    env = CournotEnv(n=2)
    q_i = env.equilibrium()["q_i"]
    x = best_response_iteration(env.best_response, np.zeros(2),
                                tol=1e-12, max_iter=100_000)
    assert np.allclose(x, q_i, atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_br_iteration_cournot_damped(n):
    """For n>=3 the simultaneous Cournot BR map is not a contraction; the RELAXED
    map x -> (1-lam)x + lam BR(x) (lam = 2/(n+1)) shares the same fixed point (the
    Nash) and converges. Validates the solver reaches the closed-form q_i*."""
    env = CournotEnv(n=n)
    q_i = env.equilibrium()["q_i"]
    lam = 2.0 / (n + 1)

    def relaxed(x):
        x = np.asarray(x, float)
        return (1.0 - lam) * x + lam * env.best_response(x)

    x = best_response_iteration(relaxed, np.zeros(n), tol=1e-13, max_iter=200_000)
    assert np.allclose(x, q_i, atol=1e-6)
    # genuine fixed point of the UNDAMPED best-response map == Nash
    assert np.allclose(env.best_response(x), x, atol=1e-6)


def test_br_iteration_raises_on_nonconvergence():
    """A non-contractive map (undamped Cournot at n=3) must fail loudly, not
    silently return a wrong point."""
    env = CournotEnv(n=3)
    with pytest.raises(RuntimeError):
        best_response_iteration(env.best_response, np.zeros(3),
                                tol=1e-12, max_iter=2_000)


# ======================================================================
# 3. fictitious_play -> converges to the Nash of matrix games.
# ======================================================================
def test_fictitious_play_matching_pennies():
    """Fictitious play's time-average converges to the (½,½),(½,½) Nash of the
    zero-sum matching-pennies game."""
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    B = -A
    x, y = fictitious_play(A, B, iters=50_000)
    assert np.allclose(x, [0.5, 0.5], atol=0.02)
    assert np.allclose(y, [0.5, 0.5], atol=0.02)


def test_fictitious_play_prisoners_dilemma_to_defect():
    """Against a dominant strategy, fictitious play concentrates on Defect."""
    R, T, P, S = 3.0, 5.0, 1.0, 0.0
    A = np.array([[R, S], [T, P]])
    B = A.T
    x, y = fictitious_play(A, B, iters=10_000)
    assert x[1] > 0.999 and y[1] > 0.999      # both essentially pure Defect


def test_fictitious_play_is_deterministic():
    """No RNG is consumed: identical inputs give bit-identical output."""
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    r1 = fictitious_play(A, -A, iters=5_000)
    r2 = fictitious_play(A, -A, iters=5_000)
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])


# ======================================================================
# 4. closed_form.* equals each env's equilibrium() (single source of truth).
# ======================================================================
def test_closed_form_cournot_equals_env():
    for (n, a, b, c) in [(2, 100.0, 1.0, 10.0), (3, 100.0, 1.0, 10.0),
                         (5, 120.0, 2.0, 10.0)]:
        env = CournotEnv(n=n, a=a, b=b, c=c)
        assert env.equilibrium() == cournot_nash(a, b, c, n)
    # hand-derived defaults
    assert cournot_nash(100.0, 1.0, 10.0, 3) == {
        "q_i": 22.5, "Q": 67.5, "P": 32.5, "profit_i": 506.25}


def test_closed_form_bertrand_diff_equals_env():
    env = BertrandDiffEnv()
    assert env.equilibrium() == bertrand_diff_nash(
        env.alpha, env.beta, env.gamma, env.c, env.n)
    eq = bertrand_diff_nash(10.0, 2.0, 1.0, 1.0, 2)
    assert eq["p"] == pytest.approx(4.0)
    assert eq["q_i"] == pytest.approx(6.0)
    assert eq["profit_i"] == pytest.approx(18.0)


def test_closed_form_first_price_equals_env():
    for n in (2, 3, 5):
        env = FirstPriceAuctionEnv(n=n)
        eq = env.equilibrium()
        sol = first_price_bne(n)
        assert eq["bid_slope"] == sol["bid_slope"] == (n - 1) / n
        assert eq["expected_revenue"] == sol["expected_revenue"] == (n - 1) / (n + 1)
    # env delegates to the SAME solver object (single source of truth)
    assert first_price_bne is econgym.closed_form.first_price_bne
    assert first_price_bne is econgym.envs.first_price.first_price_bne


def test_closed_form_second_price_equals_env():
    for n in (2, 3, 5):
        env = SecondPriceEnv(n=n)
        eq = env.equilibrium()
        sol = second_price_bne(n)
        assert eq["dominant"] is True and sol["dominant"] is True
        assert eq["bid_slope"] == sol["bid_slope"] == 1.0
        assert eq["expected_revenue"] == sol["expected_revenue"] == (n - 1) / (n + 1)


def test_closed_form_first_and_second_price_revenue_equivalence():
    """Revenue equivalence: FPA and SPA expected revenue are equal = (n-1)/(n+1)."""
    for n in (2, 3, 5):
        assert first_price_bne(n)["expected_revenue"] == \
            second_price_bne(n)["expected_revenue"] == (n - 1) / (n + 1)


def test_closed_form_rubinstein_equals_env():
    for delta in (0.5, 0.8, 0.9, 0.95):
        env = RubinsteinEnv(delta=delta)
        assert env.equilibrium() == rubinstein_split(delta)
        sol = rubinstein_split(delta)
        assert sol["proposer_share"] == pytest.approx(1.0 / (1.0 + delta))
        assert sol["responder_share"] == pytest.approx(delta / (1.0 + delta))
        # shares of a unit pie
        assert sol["proposer_share"] + sol["responder_share"] == pytest.approx(1.0)


def test_closed_form_public_goods_equals_env():
    env = PublicGoodsEnv()                     # n=4, w=20, r=0.4
    assert env.equilibrium() == public_goods_nash(env.w, env.r, env.n)
    eq = public_goods_nash(20.0, 0.4, 4)
    assert eq["nash_payoff"] == pytest.approx(20.0)
    assert eq["optimum_payoff"] == pytest.approx(32.0)
    assert eq["gap_per_player"] == pytest.approx(12.0)


def test_closed_form_repeated_pd_threshold():
    """Folk-theorem grim-trigger threshold delta* = (T-R)/(T-P); defaults -> 0.5,
    and it equals the env's reported threshold."""
    cf = repeated_pd_threshold(T=5.0, R=3.0, P=1.0, S=0.0)
    assert cf["grim_threshold"] == pytest.approx(0.5)
    assert cf["one_shot_nash"] == ("D", "D")
    env = RepeatedPDEnv()
    assert env.equilibrium()["grim_threshold"] == cf["grim_threshold"]
    with pytest.raises(ValueError):
        repeated_pd_threshold(T=1.0, R=3.0, P=1.0, S=0.0)   # T <= P undefined
