"""Validation of the UCB1 bandit agent against a KNOWN closed-form result.

Claimed property (CONTRACT_v1.md §4.1): UCB1 achieves **sublinear regret** on a
stochastic-reward bandit. The known answer we validate against is the finite-time
gap-dependent regret bound of Auer, Cesa-Bianchi & Fischer (2002),
"Finite-time Analysis of the Multiarmed Bandit Problem", *Machine Learning* 47,
Theorem 1 -- for rewards bounded in ``[0, 1]``:

    E[R_T]  <=  8 * sum_{i: Delta_i > 0} (ln T) / Delta_i
                +  (1 + pi^2 / 3) * sum_i Delta_i ,     Delta_i = mu* - mu_i .

This is ``O(K ln T)`` -- sublinear in ``T``. The test:

  1. Validates the stochastic-bandit fixture ENV against its own benchmark: each
     arm's Monte-Carlo mean reward converges to its Bernoulli parameter, and
     ``env.equilibrium()`` returns the exact best arm / optimal mean / gaps.
  2. Validates the AGENT's claimed property: mean pseudo-regret of UCB1 (run
     through the shared ``run_episode``) stays below the Auer et al. bound, across
     several bandit configurations and horizons.
  3. Confirms the regret is genuinely SUBLINEAR (per-step regret ``R_T / T`` falls
     as ``T`` grows; ``R_{4T} < 4 R_T``) and that play CONVERGES to the best arm.
  4. Unit-checks the agent mechanics: reads its arm count through the EconEnv
     interface (not hard-coded), pulls every arm once before the index rule
     engages, is deterministic on a strict index maximum (draws the shared rng
     ONLY to break ties), and is bit-stable under a fixed seed.

Every stochastic check seeds the shared ``np.random.default_rng`` explicitly, so
reruns are bit-stable (CONTRACT §5).
"""
import numpy as np
import pytest

from econgym.core import Discrete, EconEnv
from econgym.agents.ucb1 import UCB1
from econgym.runner import run_episode


# ======================================================================
# Stochastic-reward fixture environment: a Bernoulli K-armed bandit.
#
# Faithful "physics": one agent, K arms, each pull of arm a returns an
# independent Bernoulli(p_a) reward in {0, 1} drawn from the shared episode rng.
# Closed-form benchmark exposed via equilibrium(): the best arm, its optimal
# mean, and the per-arm suboptimality gaps -- plus the Auer et al. UCB1 regret
# upper bound as an explicit method (single source of truth for the test).
# ======================================================================
class BernoulliBanditEnv(EconEnv):
    """A single-agent ``K``-armed Bernoulli bandit as an :class:`EconEnv`.

    Parameters
    ----------
    means : sequence of float in [0, 1]
        Arm success probabilities ``p_a``; arm ``a``'s reward is
        ``Bernoulli(p_a)``. The optimal mean is ``mu* = max_a p_a``.
    """

    metadata = {"name": "BernoulliBandit", "simultaneous": True}

    def __init__(self, means):
        self.means = np.asarray(means, dtype=np.float64)
        if self.means.ndim != 1 or self.means.size < 2:
            raise ValueError("means must be a 1-D vector of >= 2 arm probabilities")
        if np.any(self.means < 0.0) or np.any(self.means > 1.0):
            raise ValueError("Bernoulli means must lie in [0, 1]")
        self.K = int(self.means.size)
        self.n = 1                              # single-agent bandit
        self.action_space = [Discrete(self.K)]  # per-agent Discrete arm set
        self.observation_space = [Discrete(1)]  # stateless: a constant observation

    def _reset(self):
        return [0]                              # constant (stateless) observation

    def step(self, actions):
        a = int(actions[0])
        # Bernoulli(p_a) reward drawn from the SHARED episode rng (stochastic).
        reward = float(self.rng.random() < self.means[a])
        return [0], np.array([reward], dtype=np.float64), False, False, {"arm": a}

    # ---- closed-form benchmark (single source of truth for the test) ----
    def equilibrium(self) -> dict:
        best = int(np.argmax(self.means))
        mu_star = float(self.means[best])
        return {
            "best_arm": best,
            "optimal_mean": mu_star,
            "gaps": mu_star - self.means,       # Delta_a = mu* - p_a  (>= 0)
        }

    def ucb1_regret_bound(self, T: int, c: float = 2.0) -> float:
        """Auer et al. (2002) Theorem 1 gap-dependent upper bound on E[R_T]
        (the ``c = 2`` UCB1 constants). ``sum_i Delta_i`` runs over ALL arms
        (the optimal arm contributes ``Delta = 0``)."""
        gaps = self.equilibrium()["gaps"]
        pos = gaps[gaps > 0]
        return float(8.0 * np.log(T) * np.sum(1.0 / pos)
                     + (1.0 + np.pi ** 2 / 3.0) * np.sum(gaps))


# ----------------------------------------------------------------------
# Helper: run UCB1 through run_episode and return the arm-index trace.
# ----------------------------------------------------------------------
def _run(means, T, seed, c=2.0):
    env = BernoulliBanditEnv(means)
    agent = UCB1(env, c=c)
    res = run_episode(env, [agent], T=T, seed=seed)
    return res.prices[:, 0], env               # arm trace (int), env


def _mean_pseudo_regret(means, T, n_runs, base_seed=1000, c=2.0):
    """Estimate E[R_T] = E[sum_t (mu* - mu_{a_t})] by averaging the pseudo-regret
    of ``n_runs`` independent UCB1 runs (fresh seed each)."""
    env0 = BernoulliBanditEnv(means)
    gaps = env0.equilibrium()["gaps"]
    regrets = np.empty(n_runs)
    for i in range(n_runs):
        arms, _ = _run(means, T, seed=base_seed + i, c=c)
        regrets[i] = float(np.sum(gaps[arms]))   # pseudo-regret uses TRUE gaps
    return float(regrets.mean()), regrets


# Bandit configurations exercised by the regret / convergence tests.
CONFIGS = [
    [0.9, 0.6, 0.3],                 # K=3, clear gaps
    [0.7, 0.5, 0.3],                 # K=3, tighter gaps
    [0.8, 0.5, 0.4, 0.2, 0.1],       # K=5
]


# ======================================================================
# 1. Validate the FIXTURE ENV against its own benchmark.
# ======================================================================
@pytest.mark.parametrize("means", CONFIGS)
def test_env_arm_means_match_bernoulli(means):
    """Monte-Carlo: each arm's empirical reward mean converges to its Bernoulli
    parameter (env physics faithful to the benchmark it advertises)."""
    means = np.asarray(means)
    N = 40_000
    env = BernoulliBanditEnv(means)
    for a in range(env.K):
        # the env's own step physics returns a {0,1} reward with mean p_a
        env.reset(rng=np.random.default_rng(7 + a))
        r = np.array([env.step([a])[1][0] for _ in range(N)])
        assert set(np.unique(r)).issubset({0.0, 1.0})     # rewards are bounded in [0,1]
        assert r.mean() == pytest.approx(means[a], abs=5.0 / np.sqrt(N))


@pytest.mark.parametrize("means", CONFIGS)
def test_env_equilibrium_matches_closed_form(means):
    """``equilibrium()`` returns the exact best arm, optimal mean, and gaps."""
    means = np.asarray(means)
    eq = BernoulliBanditEnv(means).equilibrium()
    assert eq["best_arm"] == int(np.argmax(means))
    assert eq["optimal_mean"] == pytest.approx(float(means.max()))
    np.testing.assert_allclose(eq["gaps"], means.max() - means)
    assert eq["gaps"][eq["best_arm"]] == pytest.approx(0.0)
    assert np.all(eq["gaps"] >= 0.0)


# ======================================================================
# 2. THE KNOWN RESULT: mean regret stays below the Auer et al. bound.
# ======================================================================
@pytest.mark.parametrize("means", CONFIGS)
@pytest.mark.parametrize("T", [1000, 2500])
def test_regret_below_ucb1_theoretical_bound(means, T):
    """Mean pseudo-regret of UCB1 over many seeds is below the closed-form
    Auer-Cesa-Bianchi-Fischer (2002) Theorem-1 bound -- the sublinear-regret
    guarantee, validated as an actual inequality against the known formula."""
    n_runs = 40
    mean_reg, _ = _mean_pseudo_regret(means, T, n_runs, base_seed=5000)
    bound = BernoulliBanditEnv(means).ucb1_regret_bound(T)
    # the theorem: empirical mean regret must not exceed the analytic bound
    assert mean_reg <= bound, (
        f"mean regret {mean_reg:.2f} exceeded UCB1 bound {bound:.2f} "
        f"(means={means}, T={T})"
    )
    # the bound is loose by construction; sanity-guard that we are safely under it
    assert mean_reg < 0.6 * bound


# ======================================================================
# 3. Regret is genuinely SUBLINEAR, and play converges to the best arm.
# ======================================================================
@pytest.mark.parametrize("means", CONFIGS)
def test_regret_is_sublinear_in_horizon(means):
    """Per-step regret ``R_T / T`` strictly decreases as the horizon grows, and
    ``R_{4T} < 4 R_T`` -- both signatures of sublinear (here ~logarithmic)
    regret. Uses matched seeds so the comparison is apples-to-apples."""
    n_runs = 30
    T = 1000
    r_T, _ = _mean_pseudo_regret(means, T, n_runs, base_seed=8000)
    r_2T, _ = _mean_pseudo_regret(means, 2 * T, n_runs, base_seed=8000)
    r_4T, _ = _mean_pseudo_regret(means, 4 * T, n_runs, base_seed=8000)

    # per-step (average) regret must fall as T grows
    assert r_2T / (2 * T) < r_T / T
    assert r_4T / (4 * T) < r_2T / (2 * T)
    # total regret grows far slower than linearly (linear would give 2x, 4x)
    assert r_2T < 1.6 * r_T
    assert r_4T < 2.2 * r_T


@pytest.mark.parametrize("means", CONFIGS)
def test_converges_to_best_arm(means):
    """In the final 20% of a long run, UCB1 plays the optimal arm the large
    majority of the time, and its greedy arm equals the true best arm."""
    means = np.asarray(means)
    best = int(np.argmax(means))
    T = 4000
    frac = []
    for s in range(20):
        env = BernoulliBanditEnv(means)
        agent = UCB1(env)
        res = run_episode(env, [agent], T=T, seed=9000 + s)
        arms = res.prices[:, 0]
        tail = arms[int(0.8 * T):]
        frac.append(float(np.mean(tail == best)))
        # the agent's learned greedy arm is the true best arm
        assert agent.greedy_arm == best
    assert np.mean(frac) > 0.9


# ======================================================================
# 4. Agent mechanics: interface generality, init round-robin, determinism.
# ======================================================================
def test_reads_arm_count_through_econenv_interface():
    """K is read via ``env.action_space[agent_id].n`` -- not hard-coded to one
    env. Works when the env exposes ONLY action_space (no ``.K``), and honours
    ``agent_id`` on a multi-agent env."""
    # (a) an env exposing only a Discrete action_space, no .K attribute
    class _StubEnv:
        action_space = [Discrete(4)]
    assert UCB1(_StubEnv()).K == 4

    # (b) our bandit env (exposes both action_space and .K) -> reads K=3
    assert UCB1(BernoulliBanditEnv([0.1, 0.5, 0.9])).K == 3

    # (c) agent_id selects the right per-agent space on an asymmetric env
    class _MultiEnv:
        action_space = [Discrete(2), Discrete(7)]
    assert UCB1(_MultiEnv(), agent_id=0).K == 2
    assert UCB1(_MultiEnv(), agent_id=1).K == 7

    # (d) fallback to .K when no action_space is present (Bertrand-style)
    class _KOnlyEnv:
        K = 11
    assert UCB1(_KOnlyEnv()).K == 11

    # (e) a Box-only action space with no .K is rejected (UCB1 is discrete)
    class _BoxEnv:
        action_space = [type("B", (), {})()]   # object without .n
    with pytest.raises(TypeError):
        UCB1(_BoxEnv())


def test_pulls_every_arm_once_before_index_rule():
    """The initialisation round-robin plays each of the K arms exactly once in
    the first K steps (in some order), before the confidence-index rule engages."""
    env = BernoulliBanditEnv([0.2, 0.5, 0.8, 0.4])
    agent = UCB1(env)
    rng = np.random.default_rng(0)
    agent.reset(rng)
    chosen = []
    for t in range(env.K):
        a = agent.act(0, t, rng)
        chosen.append(a)
        # feed an arbitrary bounded reward so counts advance
        agent.update(0, a, reward=float(env.means[a]), next_obs=0)
    assert sorted(chosen) == list(range(env.K))     # each arm exactly once
    assert np.all(agent.counts == 1)


def test_deterministic_on_strict_max_consumes_no_rng():
    """On a strict index maximum UCB1 is deterministic and draws NO rng (it uses
    the shared stream only to break ties). Verified by asserting the rng's
    internal state is unchanged across an ``act`` with a unique argmax."""
    env = BernoulliBanditEnv([0.5, 0.5])
    agent = UCB1(env)
    rng = np.random.default_rng(123)
    agent.reset(rng)
    # Force a state with both arms played once and arm 0 strictly ahead:
    agent.update(0, 0, reward=1.0, next_obs=0)      # mean[0] = 1, count[0] = 1
    agent.update(0, 1, reward=0.0, next_obs=0)      # mean[1] = 0, count[1] = 1
    # indices: arm0 = 1 + sqrt(2 ln2), arm1 = 0 + sqrt(2 ln2) -> arm0 strictly wins
    state_before = rng.bit_generator.state
    a = agent.act(0, t=2, rng=rng)
    state_after = rng.bit_generator.state
    assert a == 0
    assert state_after == state_before              # no rng draw on a strict max


def test_bit_stable_under_fixed_seed():
    """Two runs with the same seed produce an identical action trace (CONTRACT
    determinism requirement)."""
    means = [0.9, 0.6, 0.3]
    arms1, _ = _run(means, T=2000, seed=42)
    arms2, _ = _run(means, T=2000, seed=42)
    np.testing.assert_array_equal(arms1, arms2)
    # a different seed gives a different trace (the run is genuinely stochastic)
    arms3, _ = _run(means, T=2000, seed=43)
    assert not np.array_equal(arms1, arms3)
