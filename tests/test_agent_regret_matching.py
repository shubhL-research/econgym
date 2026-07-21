"""Validation of the Hart--Mas-Colell ``RegretMatching`` agent against KNOWN results.

Regret matching is a full-information, no-external-regret rule. Two textbook
theorems pin down exactly what self-play must do, and this file checks the agent
against those closed-form answers (no loosened tolerances):

  1. **Hannan consistency / closed-form regret bound.** Regret matching is a
     Blackwell-approachability strategy, so the average positive regret obeys the
     closed-form bound ``max_k R_T^+(k)/T <= M * sqrt(K/T)`` (``M`` = per-period
     payoff range) and vanishes as ``T -> infinity``. We assert the *realized*
     average regret respects this guaranteed bound at several horizons and decays.

  2. **Convergence to the (coarse) correlated equilibrium.** When both players are
     Hannan-consistent, the empirical play converges to the CCE set of the stage
     game. We validate this on two games whose CCE is unique and known in closed
     form -- cross-checked against ``solvers.normal_form.support_enumeration``:

       * the Prisoner's Dilemma stage game of ``RepeatedPDEnv`` -- unique CCE =
         unique Nash = ``(D, D)`` (strict dominance): empirical play -> (D,D);
       * matching pennies (a general 2-player matrix env, exercised to prove the
         agent is *not* hardcoded to one env) -- unique CCE = unique Nash =
         ``((1/2,1/2),(1/2,1/2))``: time-average marginals -> ``1/2``.

  3. **No-external-regret against a fixed opponent.** Against a fixed cooperator
     the best action in hindsight is Defect earning ``T`` each period; the agent's
     realized average payoff must converge up to that value (its regret -> 0).

Plus interface / seeding-contract / determinism checks (one ``rng`` draw per act,
none in reset, a valid probability simplex every step, and a clear error when no
counterfactual oracle is available).
"""
import numpy as np
import pytest

from econgym import RepeatedPDEnv, RegretMatching, run_episode
from econgym.core import Discrete, EconEnv
from econgym.agents.base import Agent
from econgym.solvers.normal_form import support_enumeration


# ======================================================================
# a general 2-player matrix EconEnv (proves the agent is env-agnostic)
# ======================================================================
class MatrixGameEnv(EconEnv):
    """Minimal general 2-player, one-memory normal-form env.

    Payoffs are two (possibly asymmetric) matrices ``A`` (row/player 0) and ``B``
    (col/player 1), each indexed ``[player0_action, player1_action]``. The
    memory-1 observation uses the EconGym joint-action encoding
    ``obs_i = own * K_opp + opp`` so the agent's default decoder recovers the
    opponent action. Intentionally exposes NO ``stage_payoff`` (asymmetric games
    have no single symmetric oracle) -- callers pass each player its own matrix.
    """

    metadata = {"name": "MatrixGame", "simultaneous": True}

    def __init__(self, A, B):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        assert self.A.shape == self.B.shape and self.A.ndim == 2
        self.n = 2
        self.n0, self.n1 = self.A.shape
        self.action_space = [Discrete(self.n0), Discrete(self.n1)]
        self.observation_space = [Discrete(self.n0 * self.n1),
                                  Discrete(self.n0 * self.n1)]
        self._prev = None

    def _obs(self, idx):
        a0, a1 = int(idx[0]), int(idx[1])
        # obs_i = own * (#opp actions) + opp  -> opp = obs_i % (#opp actions)
        return [a0 * self.n1 + a1, a1 * self.n0 + a0]

    def _reset(self):
        self._prev = self.rng.integers(0, [self.n0, self.n1])
        return self._obs(self._prev)

    def step(self, actions):
        a0, a1 = int(actions[0]), int(actions[1])
        idx = np.array([a0, a1], dtype=np.int64)
        rewards = np.array([self.A[a0, a1], self.B[a0, a1]], dtype=np.float64)
        self._prev = idx
        return self._obs(idx), rewards, False, False, {"actions": idx}

    def equilibrium(self):
        return {"note": "see solvers.normal_form.support_enumeration(A, B)"}


class _FixedAction(Agent):
    """Always plays a fixed action index (a stationary opponent)."""

    def __init__(self, a):
        self.a = int(a)

    def act(self, obs, t, rng):
        return self.a

    def update(self, obs, action, reward, next_obs):
        return None


# payoff constants
PD_R, PD_T, PD_P, PD_S = 3.0, 5.0, 1.0, 0.0
# matching pennies, each matrix from that player's OWN [own, opp] perspective:
#   matcher (player 0): +1 if own == opp;  mismatcher (player 1): +1 if own != opp
MP_MATCHER = np.array([[1.0, -1.0], [-1.0, 1.0]])      # [own, opp]
MP_MISMATCHER = np.array([[-1.0, 1.0], [1.0, -1.0]])   # [own, opp]


# ======================================================================
# 0. faithful setup: the resolved oracle equals the env physics
# ======================================================================
def test_payoff_oracle_matches_env_physics():
    env = RepeatedPDEnv(R=PD_R, T=PD_T, P=PD_P, S=PD_S)
    ag = RegretMatching(env, player_id=0)
    # resolved matrix [own, opp] must equal the PD bimatrix [[R,S],[T,P]]
    assert ag._M.tolist() == [[PD_R, PD_S], [PD_T, PD_P]]
    assert ag.K == 2 and ag.K_opp == 2
    assert ag.payoff_range() == PD_T - PD_S == 5.0


def test_reward_equals_counterfactual_of_played_action():
    """Every realized reward must equal the oracle's own-payoff for the played
    profile -- i.e. the counterfactual machinery is consistent with env physics."""
    env = RepeatedPDEnv()
    a0, a1 = RegretMatching(env, 0), RegretMatching(env, 1)
    res = run_episode(env, [a0, a1], T=500, seed=4)
    for t in range(res.T):
        own0, own1 = int(res.prices[t, 0]), int(res.prices[t, 1])
        assert a0._M[own0, own1] == res.profits[t, 0]
        assert a1._M[own1, own0] == res.profits[t, 1]


def test_requires_counterfactual_oracle():
    """Without a stage_payoff hook or an explicit payoff, construction must fail
    loudly (regret matching is a full-information rule)."""
    env = MatrixGameEnv(MP_MATCHER, MP_MISMATCHER)   # no stage_payoff exposed
    with pytest.raises(ValueError):
        RegretMatching(env, player_id=0)             # neither oracle nor payoff=
    # supplying an explicit payoff resolves it
    ag = RegretMatching(env, player_id=0, payoff=MP_MATCHER)
    assert ag._M.shape == (2, 2)


# ======================================================================
# 1. Hannan consistency: realized average regret respects M*sqrt(K/T) and -> 0
# ======================================================================
@pytest.mark.parametrize("seed", [0, 1, 7, 21])
def test_average_regret_within_closed_form_bound(seed):
    env = RepeatedPDEnv(R=PD_R, T=PD_T, P=PD_P, S=PD_S)
    a0, a1 = RegretMatching(env, 0), RegretMatching(env, 1)
    run_episode(env, [a0, a1], T=20000, seed=seed)
    for a in (a0, a1):
        bound = a.regret_bound()                     # M * sqrt(K/T), closed form
        assert bound == pytest.approx(5.0 * np.sqrt(2.0 / 20000))
        # the GUARANTEED Blackwell bound must hold, and regret must be tiny (->0)
        assert a.average_positive_regret() <= bound + 1e-12
        assert a.average_positive_regret() <= 0.01


def test_regret_bound_decays_like_one_over_sqrt_T():
    """The average-regret bound falls ~1/sqrt(T); the realized regret stays under
    it at every horizon (monotone-ish decay of a vanishing quantity)."""
    horizons = [1000, 4000, 16000, 64000]
    realized, bounds = [], []
    for T in horizons:
        env = RepeatedPDEnv()
        a0, a1 = RegretMatching(env, 0), RegretMatching(env, 1)
        run_episode(env, [a0, a1], T=T, seed=5)
        realized.append(max(a0.average_positive_regret(), a1.average_positive_regret()))
        bounds.append(a0.regret_bound())
        assert a0.average_positive_regret() <= a0.regret_bound() + 1e-12
        assert a1.average_positive_regret() <= a1.regret_bound() + 1e-12
    # the closed-form bound is monotonically decreasing in T
    assert all(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1))
    # realized regret vanishes
    assert realized[-1] <= realized[0] + 1e-9
    assert realized[-1] <= 1e-3


# ======================================================================
# 2a. convergence to the unique CCE of the PD stage game: (D, D)
# ======================================================================
def test_pd_unique_equilibrium_is_dd_via_support_enumeration():
    """The equilibrium regret matching targets is fixed by theory, cross-checked
    with the normal-form solver: the PD has a UNIQUE Nash (D, D), which -- by
    strict dominance -- is also its unique correlated and coarse-correlated
    equilibrium."""
    # row-payoff A[i,j] and col-payoff B[i,j] over profile (row i, col j), 0=C,1=D
    A = np.array([[PD_R, PD_S], [PD_T, PD_P]])   # [own_row, opp_col]
    B = A.T                                       # symmetric game
    eqs = support_enumeration(A, B)
    assert len(eqs) == 1
    x, y = eqs[0]
    assert x.tolist() == [0.0, 1.0] and y.tolist() == [0.0, 1.0]   # (D, D) pure


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_pd_selfplay_empirical_play_converges_to_dd(seed):
    env = RepeatedPDEnv(R=PD_R, T=PD_T, P=PD_P, S=PD_S)
    a0, a1 = RegretMatching(env, 0), RegretMatching(env, 1)
    res = run_episode(env, [a0, a1], T=20000, seed=seed)
    # both players' empirical marginal concentrates on Defect (action index 1)
    assert a0.empirical_frequencies()[1] >= 0.99
    assert a1.empirical_frequencies()[1] >= 0.99
    # the empirical JOINT distribution concentrates on (D, D)
    joint_dd = np.mean((res.prices[:, 0] == 1) & (res.prices[:, 1] == 1))
    assert joint_dd >= 0.99


# ======================================================================
# 2b. generic matrix env: matching pennies -> unique mixed CCE (1/2, 1/2)
# ======================================================================
def test_matching_pennies_unique_equilibrium_is_uniform():
    A = MP_MATCHER          # here A/B are already [player0, player1] for MP (symmetric)
    B = MP_MISMATCHER
    eqs = support_enumeration(A, B)
    assert len(eqs) == 1
    x, y = eqs[0]
    assert x == pytest.approx([0.5, 0.5])
    assert y == pytest.approx([0.5, 0.5])


def test_matching_pennies_selfplay_time_average_is_uniform():
    """A DIFFERENT env than the PD (proves the agent is not hardcoded): with an
    explicit per-player payoff matrix, regret-matching self-play drives the
    time-average marginals to the unique CCE (1/2, 1/2) of matching pennies."""
    env = MatrixGameEnv(MP_MATCHER, MP_MISMATCHER)
    a0 = RegretMatching(env, player_id=0, payoff=MP_MATCHER)
    a1 = RegretMatching(env, player_id=1, payoff=MP_MISMATCHER)
    run_episode(env, [a0, a1], T=100000, seed=11)
    for a in (a0, a1):
        freqs = a.empirical_frequencies()
        assert freqs[0] == pytest.approx(0.5, abs=0.03)   # -> uniform (unique CCE)
        assert freqs[1] == pytest.approx(0.5, abs=0.03)
        # Hannan consistency still holds in this zero-sum game
        assert a.average_positive_regret() <= a.regret_bound() + 1e-12


# ======================================================================
# 3. no-external-regret vs a fixed opponent: avg payoff -> best fixed action
# ======================================================================
def test_no_regret_against_fixed_cooperator():
    """Against an always-Cooperate opponent the best action in hindsight is Defect
    (earns T each period). The no-external-regret guarantee forces the agent's
    realized average payoff up to that value, and its average regret to 0."""
    env = RepeatedPDEnv(R=PD_R, T=PD_T, P=PD_P, S=PD_S)
    rm = RegretMatching(env, player_id=0)
    res = run_episode(env, [rm, _FixedAction(0)], T=20000, seed=2)
    best_fixed = PD_T                                   # Defect vs a cooperator
    avg_payoff = float(res.profits[:, 0].mean())
    assert rm.empirical_frequencies()[1] >= 0.99        # learns to Defect
    assert avg_payoff >= best_fixed - 0.05              # captures the best fixed action
    assert rm.average_positive_regret() <= rm.regret_bound() + 1e-12
    assert rm.average_positive_regret() <= 0.01


def test_no_regret_against_fixed_defector():
    """Against an always-Defect opponent the best fixed action is also Defect
    (P > S); the agent must reach avg payoff P and vanishing regret."""
    env = RepeatedPDEnv(R=PD_R, T=PD_T, P=PD_P, S=PD_S)
    rm = RegretMatching(env, player_id=0)
    res = run_episode(env, [rm, _FixedAction(1)], T=20000, seed=8)
    assert rm.empirical_frequencies()[1] >= 0.99
    assert float(res.profits[:, 0].mean()) >= PD_P - 0.05
    assert rm.average_positive_regret() <= rm.regret_bound() + 1e-12


# ======================================================================
# 4. interface / seeding contract / determinism
# ======================================================================
def test_reset_consumes_no_rng():
    env = RepeatedPDEnv()
    ag = RegretMatching(env, 0)
    r = np.random.default_rng(5)
    twin = np.random.default_rng(5)
    ag.reset(r)                                          # must draw nothing
    assert r.random() == twin.random()


def test_act_draws_exactly_one_rng_value():
    env = RepeatedPDEnv()
    ag = RegretMatching(env, 0)
    ag.reset(np.random.default_rng(0))
    r = np.random.default_rng(9)
    twin = np.random.default_rng(9)
    ag.act(obs=0, t=0, rng=r)                            # one rng.random()
    _ = twin.random()
    assert r.random() == twin.random()                  # streams aligned


def test_initial_distribution_is_uniform_and_valid_simplex():
    env = RepeatedPDEnv()
    ag = RegretMatching(env, 0)
    ag.reset(np.random.default_rng(0))
    p = ag._distribution()
    assert p == pytest.approx([0.5, 0.5])               # empty regret -> uniform
    # after arbitrary regret, always a valid probability vector
    ag.regret = np.array([-2.0, 3.0])
    q = ag._distribution()
    assert q == pytest.approx([0.0, 1.0])
    assert np.all(q >= 0) and q.sum() == pytest.approx(1.0)


def test_selfplay_is_deterministic_under_fixed_seed():
    def trace():
        env = RepeatedPDEnv()
        agents = [RegretMatching(env, 0), RegretMatching(env, 1)]
        return run_episode(env, agents, T=3000, seed=123).prices
    assert np.array_equal(trace(), trace())


def test_actions_stay_in_range_over_a_run():
    env = RepeatedPDEnv()
    a0, a1 = RegretMatching(env, 0), RegretMatching(env, 1)
    res = run_episode(env, [a0, a1], T=1000, seed=1)
    assert res.prices.min() >= 0 and res.prices.max() <= 1   # valid Discrete(2) indices
