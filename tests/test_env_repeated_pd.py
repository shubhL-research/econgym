"""Validation of ``RepeatedPDEnv`` against its closed-form benchmark.

Per CONTRACT_v1.md Section 3.6, the infinitely-repeated Prisoner's Dilemma is
validated four ways, all of which are checks against theory (no loosened
tolerances):

  1. **Stage-game dominance / unique Nash** -- ``D`` strictly dominates ``C`` for
     both players, so the *unique* pure-strategy stage Nash is ``(D, D)``
     (deviation / best-response check; cross-checked with a self-contained
     2x2 pure-Nash enumerator that stands in for
     ``solvers.normal_form.support_enumeration``).
  2. **Folk-theorem threshold** -- ``env.equilibrium()["grim_threshold"]`` equals
     ``(T - R) / (T - P)`` exactly, cross-checked against
     ``solvers.closed_form.repeated_pd_threshold`` (the single source of truth).
  3. **Grim-trigger incentive constraint** -- the discounted cooperation value
     ``R/(1-delta)`` weakly exceeds the one-shot-deviation value
     ``T + delta*P/(1-delta)`` iff ``delta >= delta*`` (strict inequality flips at
     the threshold).
  4. **Parameter-ordering guard** -- the constructor rejects payoffs violating
     ``T > R > P > S`` or efficiency ``2R > T + S`` (and ``delta`` outside
     ``(0,1)``, ``n != 2``).

Plus faithful-physics, seeding-contract, and runner-integration checks (a pair of
*myopic best-response* learners converge to the ``(D, D)`` stage equilibrium every
period through the real ``run_episode``).
"""
import numpy as np
import pytest

from econgym import Agent, RepeatedPDEnv, run_episode
from econgym.solvers.closed_form import repeated_pd_threshold


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _bimatrix(env):
    """Return (A, B): own-payoff matrices for player 0 (row) and player 1 (col),
    indexed [own_action][opp_action] with 0=C, 1=D, built from the env physics."""
    A = [[env.stage_payoff(i, j) for j in (0, 1)] for i in (0, 1)]  # player 0
    B = [[env.stage_payoff(j, i) for j in (0, 1)] for i in (0, 1)]  # player 1
    return A, B


def _pure_nash_profiles(A, B, tol=1e-12):
    """Self-contained pure-strategy Nash finder for a 2x2 bimatrix.

    Stands in for ``solvers.normal_form.support_enumeration`` restricted to pure
    supports: a profile ``(i, j)`` is a Nash iff each player's action is a best
    response to the other's. Returns the sorted list of pure-Nash profiles.
    """
    out = []
    for i in (0, 1):
        for j in (0, 1):
            row_br = A[i][j] >= max(A[0][j], A[1][j]) - tol   # player 0 best-responds to j
            col_br = B[i][j] >= max(B[i][0], B[i][1]) - tol   # player 1 best-responds to i
            if row_br and col_br:
                out.append((i, j))
    return sorted(out)


class _AllDefect(Agent):
    def act(self, obs, t, rng):
        return 1

    def update(self, obs, action, reward, next_obs):
        return None


class _AllCooperate(Agent):
    def act(self, obs, t, rng):
        return 0

    def update(self, obs, action, reward, next_obs):
        return None


class _MyopicBestResponse(Agent):
    """Best-responds to the opponent's last action, read from the memory-1 obs
    (``opp_prev = obs % 2``). Computed from the env's stage payoffs -- it is NOT
    hardwired to defect; that it always returns D is exactly the dominance result
    under test."""

    def __init__(self, env):
        self.env = env

    def act(self, obs, t, rng):
        opp_prev = int(obs) % 2
        return 0 if self.env.stage_payoff(0, opp_prev) > self.env.stage_payoff(1, opp_prev) else 1

    def update(self, obs, action, reward, next_obs):
        return None


# ----------------------------------------------------------------------
# 0. faithful physics: the stage bimatrix and observation encoding
# ----------------------------------------------------------------------
def test_step_physics_matches_bimatrix():
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=0.8)
    env.reset(seed=0)
    # (own_a0, own_a1) -> (reward0, reward1, obs0, obs1)
    cases = {
        (0, 0): (3.0, 3.0, 0, 0),   # (C,C) -> (R,R)
        (0, 1): (0.0, 5.0, 1, 2),   # (C,D) -> (S,T)
        (1, 0): (5.0, 0.0, 2, 1),   # (D,C) -> (T,S)
        (1, 1): (1.0, 1.0, 3, 3),   # (D,D) -> (P,P)
    }
    for (a0, a1), (r0, r1, o0, o1) in cases.items():
        obs, rewards, term, trunc, info = env.step([a0, a1])
        assert rewards.tolist() == [r0, r1]
        assert obs == [o0, o1]                       # 2*own + opp encoding
        assert term is False and trunc is False       # infinite-horizon
        assert isinstance(info, dict)
        assert info["actions"].tolist() == [a0, a1]


def test_spaces_shape_and_membership():
    env = RepeatedPDEnv()
    assert env.n == 2
    assert len(env.action_space) == 2 and len(env.observation_space) == 2
    for sp in env.action_space:
        assert sp.n == 2 and sp.contains(0) and sp.contains(1) and not sp.contains(2)
    obs = env.reset(seed=1)
    for i in (0, 1):
        assert env.observation_space[i].n == 4
        assert env.observation_space[i].contains(obs[i])


# ----------------------------------------------------------------------
# 1. one-shot best response is Defect; unique pure Nash is (D, D)
# ----------------------------------------------------------------------
def test_one_shot_best_response_is_defect():
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0)
    # best response to a Cooperator is Defect (T > R); to a Defector is Defect (P > S)
    assert env.stage_payoff(1, 0) > env.stage_payoff(0, 0)   # T > R
    assert env.stage_payoff(1, 1) > env.stage_payoff(0, 1)   # P > S

    A, B = _bimatrix(env)
    nash = _pure_nash_profiles(A, B)
    assert nash == [(1, 1)]                                   # unique pure Nash = (D, D)

    eq = env.equilibrium()
    assert eq["one_shot_nash"] == ("D", "D")


@pytest.mark.parametrize("R,T,P,S", [(3, 5, 1, 0), (2, 3, 1, 0), (4, 6, 2, 1)])
def test_stage_nash_unique_dd_across_payoffs(R, T, P, S):
    env = RepeatedPDEnv(R=R, T=T, P=P, S=S)
    A, B = _bimatrix(env)
    assert _pure_nash_profiles(A, B) == [(1, 1)]


# ----------------------------------------------------------------------
# 2. folk-theorem threshold matches the closed form exactly
# ----------------------------------------------------------------------
def test_folk_threshold_matches_closed_form():
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=0.8)
    eq = env.equilibrium()
    assert eq["grim_threshold"] == (5 - 3) / (5 - 1) == 0.5     # exact
    # single source of truth: env delegates to solvers.closed_form
    solver = repeated_pd_threshold(env.T, env.R, env.P, env.S)
    assert eq["grim_threshold"] == solver["grim_threshold"]
    assert eq["one_shot_nash"] == solver["one_shot_nash"]


@pytest.mark.parametrize("R,T,P,S", [(3, 5, 1, 0), (2, 3, 1, 0), (4, 6, 2, 1), (3, 4, 1, 0)])
def test_threshold_formula_across_payoffs(R, T, P, S):
    env = RepeatedPDEnv(R=R, T=T, P=P, S=S)
    assert env.equilibrium()["grim_threshold"] == pytest.approx((T - R) / (T - P), rel=1e-12)


def test_solver_rejects_degenerate_T_le_P():
    with pytest.raises(ValueError):
        repeated_pd_threshold(T=1.0, R=0.5, P=1.0, S=0.0)     # T <= P => threshold undefined


# ----------------------------------------------------------------------
# 3. grim-trigger sustains cooperation iff delta >= delta*
# ----------------------------------------------------------------------
def test_grim_incentive_constraint_flips_at_threshold():
    # defaults: delta* = 0.5
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=0.8)
    dstar = env.equilibrium()["grim_threshold"]
    assert dstar == 0.5

    # exactly at the threshold: cooperation and deviation values are EQUAL
    coop = env.cooperation_value(dstar)
    dev = env.deviation_value(dstar)
    assert coop == pytest.approx(dev, abs=1e-12)              # 6.0 == 6.0

    # just above: strict incentive to cooperate
    hi = dstar + 1e-3
    assert env.cooperation_value(hi) > env.deviation_value(hi)

    # just below: cooperation is NOT self-enforcing
    lo = dstar - 1e-3
    assert env.cooperation_value(lo) < env.deviation_value(lo)


@pytest.mark.parametrize("delta,expected", [(0.5, True), (0.55, True), (0.8, True),
                                            (0.45, False), (0.3, False)])
def test_sustainability_flag_matches_incentive(delta, expected):
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=delta)
    eq = env.equilibrium()
    assert eq["coop_sustainable_at"]["delta"] == delta
    assert eq["coop_sustainable_at"]["sustainable"] is expected
    # the flag must agree with the raw incentive constraint (R/(1-d) >= T + dP/(1-d))
    holds = env.cooperation_value(delta) >= env.deviation_value(delta) - 1e-12
    assert holds is expected


# ----------------------------------------------------------------------
# 4. constructor rejects non-PD / out-of-range parameters
# ----------------------------------------------------------------------
def test_param_ordering_validated():
    with pytest.raises(ValueError):
        RepeatedPDEnv(R=3, T=2, P=1, S=0)                     # T < R violates T>R>P>S
    with pytest.raises(ValueError):
        RepeatedPDEnv(R=3, T=5, P=6, S=0)                     # P > R violates ordering
    with pytest.raises(ValueError):
        RepeatedPDEnv(R=3, T=5.5, P=1, S=0.6)                 # ordering ok but 2R=6 <= T+S=6.1
    with pytest.raises(ValueError):
        RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=1.0)          # delta not in (0,1)
    with pytest.raises(ValueError):
        RepeatedPDEnv(R=3, T=5, P=1, S=0, delta=0.0)
    with pytest.raises(ValueError):
        RepeatedPDEnv(n=3)                                    # stage PD is 2-player


def test_efficiency_boundary_ok():
    # 2R > T + S strictly satisfied -> constructs fine
    RepeatedPDEnv(R=3, T=5, P=1, S=0)                          # 6 > 5
    RepeatedPDEnv(R=4, T=6, P=2, S=1)                          # 8 > 7


# ----------------------------------------------------------------------
# seeding contract + determinism
# ----------------------------------------------------------------------
def test_reset_consumes_one_shared_draw():
    env = RepeatedPDEnv()
    shared = np.random.default_rng(42)
    obs = env.reset(rng=shared)
    twin = np.random.default_rng(42)
    prev = twin.integers(0, 2, size=2)                        # the ONE env draw
    assert obs == [2 * int(prev[0]) + int(prev[1]), 2 * int(prev[1]) + int(prev[0])]
    assert shared.random() == twin.random()                   # streams aligned (one draw only)
    assert env.rng is shared


def test_reset_seed_is_deterministic():
    assert RepeatedPDEnv().reset(seed=7) == RepeatedPDEnv().reset(seed=7)


# ----------------------------------------------------------------------
# runner integration: greedy best-response converges to the (D,D) equilibrium
# ----------------------------------------------------------------------
def test_myopic_best_response_converges_to_dd_through_runner():
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0)
    agents = [_MyopicBestResponse(env), _MyopicBestResponse(env)]
    res = run_episode(env, agents, T=50, seed=123)
    assert res.prices.shape == (50, 2)
    # myopic best response to ANY opponent history is Defect -> (D,D) every period
    assert np.all(res.prices == 1)
    assert np.all(res.profits == env.P)                       # both earn P forever


def test_runner_reward_traces_match_bimatrix():
    env = RepeatedPDEnv(R=3, T=5, P=1, S=0)
    # two cooperators earn R every period; two defectors earn P every period
    rc = run_episode(env, [_AllCooperate(), _AllCooperate()], T=20, seed=1)
    assert np.all(rc.profits == env.R)
    rd = run_episode(env, [_AllDefect(), _AllDefect()], T=20, seed=1)
    assert np.all(rd.profits == env.P)
