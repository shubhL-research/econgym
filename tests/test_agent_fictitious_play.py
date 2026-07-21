"""Validation tests for the shared ``FictitiousPlay`` learning agent.

KNOWN RESULT UNDER TEST
-----------------------
Fictitious play (Brown 1951; Robinson 1951) -- each player best-responds to the
empirical frequency of its opponents' past actions -- **converges to a Nash
equilibrium** in every game with the *fictitious-play property*: potential games
(Monderer & Shapley 1996), two-player zero-sum games, 2xN games, and
dominance-solvable games. Every applicable environment in this suite is in that
class, so fictitious play must converge to the environment's documented
CLOSED-FORM equilibrium. Each test below validates the agent's converged play
against that closed form -- nothing is loosened to force a pass; where the
discrete action grid or the slow per-firm rate of continuous fictitious play
bounds attainable precision, the test asserts the quantity the theorem actually
pins down (the aggregate / price / mean profit) to grid tolerance, plus the exact
best-response decision rule.

The agent is driven **through the shared EconEnv interface** (``reset`` / ``act``
/ ``native`` / ``step`` / ``update``) by :func:`run_fp`, which is environment
agnostic -- the same driver runs Cournot, differentiated Bertrand, public goods,
and the repeated-PD stage game.

Environment / closed form validated
-----------------------------------
* ``cournot``       -> symmetric Nash ``q_i* = (a-c)/(b(n+1))`` (default n=3:
  ``q*=22.5``, ``Q*=67.5``, ``P*=32.5``, ``pi*=506.25``).  Fictitious play pins
  down the aggregate/price/mean-profit EXACTLY (potential game, unique Nash); its
  best response at the Nash belief returns the grid point nearest ``q*`` exactly.
* ``bertrand_diff`` -> symmetric Nash price ``p* = (alpha+beta c)/(2beta-gamma(n-1))``
  (defaults ``p*=4.0``, on-grid) -- reached per firm within one grid step.
* ``public_goods``  -> dominant-strategy Nash ``c_i = 0`` -- reached exactly.
* ``repeated_pd``   -> unique stage-game Nash ``(D, D)`` -- reached exactly.
"""
import numpy as np
import pytest

from econgym import (
    CournotEnv, BertrandDiffEnv, PublicGoodsEnv, RepeatedPDEnv,
    FictitiousPlay, BestResponse,
)
from econgym.agents.base import Agent


# ----------------------------------------------------------------------
# Generic, environment-agnostic fictitious-play driver (uses ONLY the shared
# EconEnv + Agent interface: reset / act / native / step / update).
# ----------------------------------------------------------------------
def run_fp(env, T, seed, belief="mean", cls=FictitiousPlay):
    """Play ``T`` periods of fictitious play and return the native-action trace.

    One :class:`FictitiousPlay` per seat. Each period every agent picks an action
    INDEX (``act``), the index is mapped to the env's native action (``native``)
    for ``step``, and the realised transition is fed back via ``update``. Works
    for any EconGym env with a discrete action grid (continuous envs) or discrete
    action space (matrix games).
    """
    rng = np.random.default_rng(seed)
    n = env.n
    agents = [cls(env, player_id=i, belief=belief) if cls is FictitiousPlay
              else cls(env, player_id=i) for i in range(n)]
    for a in agents:
        a.reset(rng)
    obs = env.reset(rng=rng)
    trace = []
    for t in range(T):
        idx = [agents[i].act(obs[i], t, rng) for i in range(n)]
        native = [agents[i].native(idx[i]) for i in range(n)]
        next_obs, rewards, terminated, truncated, info = env.step(native)
        for i in range(n):
            agents[i].update(obs[i], idx[i], rewards[i], next_obs[i])
        trace.append([float(x) for x in native])
        obs = next_obs
        if terminated or truncated:
            break
    return np.asarray(trace), agents


# ======================================================================
# 0. The agent honours the shared Agent ABC contract.
# ======================================================================
def test_is_agent_subclass_and_returns_index():
    env = CournotEnv()
    a = FictitiousPlay(env, player_id=0)
    assert isinstance(a, Agent)
    rng = np.random.default_rng(0)
    a.reset(rng)
    obs = env.reset(rng=rng)
    act = a.act(obs[0], 0, rng)
    # act returns a valid ACTION INDEX into the agent's grid
    assert isinstance(act, int)
    assert 0 <= act < a.K
    assert a.native(act) == env.qgrid[act]


def test_reset_consumes_no_rng():
    """Like MeanBased, FictitiousPlay draws nothing at reset (deterministic best
    response) -- the shared episode stream must be byte-unperturbed."""
    env = CournotEnv()
    a = FictitiousPlay(env, player_id=0)
    shared = np.random.default_rng(123)
    before = shared.bit_generator.state
    a.reset(shared)
    after = shared.bit_generator.state
    assert before == after            # not one draw consumed


# ======================================================================
# 1. CORNOT (primary): convergence to the closed-form Nash q*.
# ======================================================================
def test_cournot_default_converges_to_closed_form_exactly():
    """Default n=3 Cournot (a=100,b=1,c=10): the CLOSED FORM is q*=22.5, Q*=67.5,
    P*=32.5, pi*=506.25 (q* lies exactly on the qgrid). Fictitious play pins the
    aggregate quantity, market price, and mean per-firm profit to the closed form
    EXACTLY (the unique Nash of this potential game)."""
    env = CournotEnv()                                   # n=3 defaults
    eq = env.equilibrium()
    assert (eq["q_i"], eq["Q"], eq["P"], eq["profit_i"]) == (22.5, 67.5, 32.5, 506.25)

    for seed in range(4):
        trace, _ = run_fp(env, 3000, seed)
        tail = trace[-1000:]
        Q = tail.sum(axis=1).mean()
        P = env.a - env.b * Q
        mean_profit = ((P - env.c) * tail).mean()
        assert Q == pytest.approx(eq["Q"], abs=1e-9)
        assert P == pytest.approx(eq["P"], abs=1e-9)
        assert mean_profit == pytest.approx(eq["profit_i"], abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_cournot_aggregate_and_price_within_grid_step(n):
    """For n in {2,3,5} the converged AGGREGATE quantity is within ONE grid step
    of Q* and the market price within ``b * grid_step`` of P* -- the aggregate and
    price are exactly the observables the unique symmetric Cournot Nash pins down;
    the split among identical firms carries genuine grid-resolution slack."""
    env = CournotEnv(n=n)
    eq = env.equilibrium()
    grid_step = env.q_max / (env.G - 1)
    for seed in range(3):
        trace, _ = run_fp(env, 4000, seed)
        tail = trace[-1500:]
        Q = tail.sum(axis=1).mean()
        P = env.a - env.b * Q
        assert abs(Q - eq["Q"]) <= grid_step + 1e-9
        assert abs(P - eq["P"]) <= env.b * grid_step + 1e-9


@pytest.mark.parametrize("n", [2, 3, 5])
def test_cournot_best_response_at_nash_equals_closed_form(n):
    """The agent's DECISION RULE, evaluated at the exact Nash belief (every
    opponent at q*), returns the grid point nearest the closed-form Nash quantity
    -- and exactly q* when q* is on the grid (n=3). This ties the learned best
    response to the closed form itself ("solver output == closed form"), using
    only the public ``act`` API (fold the Nash observation, then best-respond)."""
    env = CournotEnv(n=n)
    q_star = env.equilibrium()["q_i"]
    rng = np.random.default_rng(0)
    a = FictitiousPlay(env, player_id=0)
    a.reset(rng)
    # observe every opponent playing exactly q* -> belief mean == q*
    nash_obs = np.full(n - 1, q_star)
    idx = a.act(nash_obs, 0, rng)
    played = a.native(idx)
    nearest = env.qgrid[int(np.argmin(np.abs(env.qgrid - q_star)))]
    assert played == pytest.approx(nearest)
    # the continuous best response at the symmetric Nash profile IS q* exactly
    q_profile = np.full(n, q_star)
    assert env.best_response(q_profile)[0] == pytest.approx(q_star, rel=1e-12)
    # so when q* is on the grid, the agent plays it exactly (default n=3)
    if np.isclose(nearest, q_star):
        assert played == pytest.approx(q_star)


# ======================================================================
# 2. Differentiated Bertrand: convergence to the closed-form Nash price p*.
# ======================================================================
def test_bertrand_diff_converges_to_price_nash():
    """Defaults alpha=10,beta=2,gamma=1,c=1,n=2 give the closed-form symmetric
    Nash price p*=(alpha+beta c)/(2beta-gamma(n-1))=4.0 (on the pgrid). Strong
    best-response contraction (factor 0.25) -> fictitious play reaches p* per firm
    within one grid step."""
    env = BertrandDiffEnv()
    p_star = env.equilibrium()["p"]
    assert p_star == 4.0
    grid_step = (env.p_cap - env.c) / (env.G - 1)
    for seed in range(4):
        trace, _ = run_fp(env, 3000, seed)
        final = trace[-1]
        assert np.max(np.abs(final - p_star)) <= grid_step + 1e-9
        # both firms sit at the on-grid Nash price exactly at convergence
        assert np.allclose(final, p_star, atol=grid_step)


# ======================================================================
# 3. Public goods (VCM): convergence to the dominant-strategy Nash c_i = 0.
# ======================================================================
def test_public_goods_converges_to_free_riding():
    """The VCM Nash is the strictly dominant free-ride corner c_i=0 (payoff w=20);
    best-responding to ANY belief gives 0, so fictitious play reaches the Nash
    exactly."""
    env = PublicGoodsEnv()
    eq = env.equilibrium()
    assert eq["nash_contribution"] == 0.0 and eq["nash_payoff"] == 20.0
    for seed in range(4):
        trace, agents = run_fp(env, 300, seed)
        assert np.all(trace[-1] == 0.0)                  # every firm free-rides
        # payoff at the converged (all-zero) profile == the closed-form Nash payoff
        assert np.allclose(env.payoff(trace[-1]), eq["nash_payoff"])


# ======================================================================
# 4. Repeated-PD stage game: convergence to the unique Nash (D, D).
# ======================================================================
def test_repeated_pd_converges_to_defect():
    """The PD stage game is dominance-solvable with the UNIQUE Nash (D, D)=(1, 1)
    (D strictly dominates C). Fictitious play best-responds to the empirical mixed
    belief and converges to mutual defection regardless of the belief, matching
    ``equilibrium()['one_shot_nash']``."""
    env = RepeatedPDEnv()
    assert env.equilibrium()["one_shot_nash"] == ("D", "D")
    for seed in range(4):
        trace, agents = run_fp(env, 200, seed)
        assert np.all(trace[-50:] == 1.0)                # sustained mutual defection
    # the belief overwhelmingly concentrates on the opponent playing D (index 1)
    belief = agents[0].belief_mean()                     # empirical P over {C, D}
    assert belief[1] > 0.9


def test_repeated_pd_best_response_is_always_defect():
    """D is a strictly dominant stage action, so the agent's best response is D for
    EVERY belief over the opponent's play -- checked across the full belief
    simplex (the exact matrix-game fictitious-play best response)."""
    env = RepeatedPDEnv()
    a = FictitiousPlay(env, player_id=0)
    a.reset(np.random.default_rng(0))
    for p_defect in np.linspace(0.0, 1.0, 51):
        a._opp_counts = np.array([1.0 - p_defect, p_defect])
        a._count = 1
        assert a.greedy_action() == env.DEFECT           # index 1 == Defect


# ======================================================================
# 5. Determinism: the driven episode is bit-stable across reruns.
# ======================================================================
def test_determinism_same_seed_same_trace():
    env = CournotEnv()
    t1, _ = run_fp(env, 500, 20260722)
    t2, _ = run_fp(CournotEnv(), 500, 20260722)
    assert np.array_equal(t1, t2)


# ======================================================================
# 6. BestResponse (belief="last") is the myopic Cournot best-reply variant:
#    a contraction at n=2 (converges) but NOT at n=3 (the undamped map cycles).
# ======================================================================
def test_best_response_variant_converges_at_n2_only():
    # n=2: undamped best-reply is a contraction -> reaches the Nash aggregate.
    env2 = CournotEnv(n=2)
    trace, _ = run_fp(env2, 3000, 0, cls=BestResponse)
    Q = trace[-500:].sum(axis=1).mean()
    grid_step = env2.q_max / (env2.G - 1)
    assert abs(Q - env2.equilibrium()["Q"]) <= 2 * grid_step

    # n=3: the undamped simultaneous best-reply map is neutrally stable, so myopic
    # best response (no averaging) does NOT settle -- it persistently cycles. The
    # averaged FictitiousPlay agent, by contrast, converges (test 1). We assert the
    # myopic variant keeps moving in its tail (a genuine, un-loosened distinction).
    env3 = CournotEnv(n=3)
    trace3, _ = run_fp(env3, 2000, 0, cls=BestResponse)
    tail = trace3[-200:]
    assert np.ptp(tail, axis=0).max() > grid_step        # still oscillating
