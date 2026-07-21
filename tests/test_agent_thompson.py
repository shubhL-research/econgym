"""Validation tests for the Thompson-sampling bandit agent (``econgym.Thompson``).

Claimed property: Thompson sampling *learns the best action* -- it concentrates
its pulls on the arm with the highest mean and incurs only sublinear
(logarithmic) cumulative regret. We validate this against a KNOWN answer using a
stochastic multi-armed bandit whose optimal policy is closed-form: the best arm
is ``argmax_a mu_a`` and the optimal per-step reward is ``mu* = max_a mu_a``, so
the pseudo-regret of any policy is ``R(T) = sum_t (mu* - mu_{a_t})``. Theory
(Thompson 1933; Lai & Robbins 1985; Agrawal & Goyal 2012/2013) says a good bandit
policy drives ``R(T)/T -> 0`` with ``R(T) = Theta(ln T)``, matching the
Lai-Robbins lower bound ``R(T) >= (sum_{i != *} Delta_i / KL(mu_i, mu*)) ln T``
up to constants -- exactly what we check here.

The agent is exercised THROUGH the shared ``EconEnv`` / ``run_episode``
interface: a minimal :class:`StochasticBanditEnv` (subclassing ``EconEnv``,
honoring the single-shared-rng seeding contract, returning the 5-tuple ``step``,
and exposing its closed-form optimum via ``equilibrium()``) plays the role of the
"env physics". Thompson reads only the arm count from the env, so the same agent
also drops into a real game env (a Bertrand smoke below) unchanged.
"""
import numpy as np
import pytest

from econgym import BertrandEnv, Thompson, run_episode
from econgym.core import Discrete, EconEnv


# ----------------------------------------------------------------------
# A fixed stochastic multi-armed bandit as an EconEnv (n = 1 agent).
# Closed-form benchmark (equilibrium): pull argmax-mean arm forever.
# ----------------------------------------------------------------------
class StochasticBanditEnv(EconEnv):
    """``K``-armed stochastic bandit. Arm ``a`` pays a reward with mean
    ``means[a]``; ``reward='bernoulli'`` -> ``r ~ Bernoulli(means[a])`` (means in
    ``[0,1]``), ``reward='gaussian'`` -> ``r = means[a] + noise * N(0,1)``. Reward
    randomness is drawn from the SHARED episode ``rng`` (seeding contract)."""

    metadata = {"name": "StochasticBandit", "simultaneous": True}

    def __init__(self, means, reward="bernoulli", noise=0.3):
        self.means = np.asarray(means, dtype=float)
        self.K = int(self.means.size)
        self.n = 1
        self.reward = reward
        self.noise = float(noise)
        self.action_space = [Discrete(self.K)]
        self.observation_space = [Discrete(1)]

    def _reset(self):
        return [0]                              # constant (stateless) observation

    def step(self, actions):
        a = int(actions[0])
        if self.reward == "bernoulli":
            r = float(self.rng.random() < self.means[a])
        else:
            r = float(self.means[a] + self.noise * self.rng.standard_normal())
        return [0], np.array([r], dtype=float), False, False, {"arm": a}

    def equilibrium(self) -> dict:
        """Closed-form optimum of the bandit: always play the highest-mean arm."""
        best = int(np.argmax(self.means))
        return {
            "best_arm": best,
            "optimal_mean": float(self.means[best]),
            "gaps": (self.means[best] - self.means),
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _pseudo_regret(action_trace, means):
    means = np.asarray(means, float)
    gaps = means.max() - means
    return np.cumsum(gaps[np.asarray(action_trace, int)])


def _bernoulli_kl(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _lai_robbins_coeff_bernoulli(means):
    """Lai-Robbins asymptotic constant ``sum_{i != *} Delta_i / KL(mu_i, mu*)``."""
    means = np.asarray(means, float)
    star = means.max()
    best = int(np.argmax(means))
    return sum((star - means[i]) / _bernoulli_kl(means[i], star)
               for i in range(means.size) if i != best)


BERNOULLI_MEANS = [0.2, 0.4, 0.6, 0.9]     # best arm = index 3, mu* = 0.9
GAUSSIAN_MEANS = [0.0, 0.5, 1.0, 1.5]      # best arm = index 3, mu* = 1.5


# ----------------------------------------------------------------------
# 0. The env's benchmark hook equals the closed form (validate the oracle).
# ----------------------------------------------------------------------
def test_env_benchmark_matches_closed_form():
    env = StochasticBanditEnv(BERNOULLI_MEANS)
    eq = env.equilibrium()
    assert eq["best_arm"] == 3
    assert eq["optimal_mean"] == pytest.approx(0.9)
    assert np.allclose(eq["gaps"], [0.7, 0.5, 0.3, 0.0])


# ----------------------------------------------------------------------
# 1. Thompson concentrates on the best arm (both posteriors).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dist,means,reward,noise", [
    ("beta", BERNOULLI_MEANS, "bernoulli", 0.0),
    ("gaussian", BERNOULLI_MEANS, "bernoulli", 0.0),
    ("gaussian", GAUSSIAN_MEANS, "gaussian", 0.5),
])
@pytest.mark.parametrize("seed", [7, 13, 21, 99])
def test_learns_best_arm(dist, means, reward, noise, seed):
    T = 20_000
    best = int(np.argmax(means))
    env = StochasticBanditEnv(means, reward=reward, noise=noise)
    agent = Thompson(env, dist=dist)
    res = run_episode(env, [agent], T=T, seed=seed)

    acts = res.prices[:, 0]
    # (a) the agent's exploitation choice is the true best arm
    assert agent.greedy_arm() == best
    # (b) it PLAYS the best arm the vast majority of the time ...
    assert (acts == best).mean() >= 0.95
    # (c) ... and essentially always by the tail (last 25% of the horizon)
    assert (acts[int(0.75 * T):] == best).mean() >= 0.95
    # (d) the single most-pulled arm is the best arm
    counts = np.bincount(acts, minlength=len(means))
    assert int(np.argmax(counts)) == best


# ----------------------------------------------------------------------
# 2. Sublinear (logarithmic) regret -- the KNOWN Thompson/Lai-Robbins result.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dist", ["beta", "gaussian"])
def test_regret_is_sublinear(dist):
    means = BERNOULLI_MEANS
    seed = 7
    T = 20_000

    env = StochasticBanditEnv(means, reward="bernoulli")
    reg_T = _pseudo_regret(run_episode(env, [Thompson(env, dist=dist)],
                                       T=T, seed=seed).prices[:, 0], means)[-1]
    env2 = StochasticBanditEnv(means, reward="bernoulli")
    reg_2T = _pseudo_regret(run_episode(env2, [Thompson(env2, dist=dist)],
                                        T=2 * T, seed=seed).prices[:, 0], means)[-1]

    gaps = np.asarray(means, float).max() - np.asarray(means, float)
    uniform_regret = T * gaps.mean()          # regret of a uniform-random policy
    lr = _lai_robbins_coeff_bernoulli(means)  # Lai-Robbins log-regret constant

    # (a) average regret vanishes
    assert reg_T / T < 0.01
    # (b) MUCH better than uniform-random selection (a real, non-trivial bound)
    assert reg_T < 0.05 * uniform_regret
    # (c) logarithmic ORDER: within a constant factor of the Lai-Robbins bound
    #     (a linear-regret policy would exceed this by ~2 orders of magnitude).
    assert reg_T <= 8.0 * lr * np.log(T)
    # (d) doubling the horizon barely grows regret (log ~1.07x; linear would 2x)
    assert reg_2T / reg_T < 1.6


# ----------------------------------------------------------------------
# 3. Determinism under the seeding contract (same seed -> identical trace).
# ----------------------------------------------------------------------
def test_determinism():
    means = BERNOULLI_MEANS
    e1 = StochasticBanditEnv(means); e2 = StochasticBanditEnv(means)
    r1 = run_episode(e1, [Thompson(e1, dist="beta")], T=3000, seed=42)
    r2 = run_episode(e2, [Thompson(e2, dist="beta")], T=3000, seed=42)
    assert np.array_equal(r1.prices, r2.prices)
    assert np.array_equal(r1.profits, r2.profits)


# ----------------------------------------------------------------------
# 4. update() consumes NO rng (Agent contract); act() ignores the observation.
# ----------------------------------------------------------------------
def test_update_consumes_no_rng():
    rng = np.random.default_rng(0)
    agent = Thompson(n_actions=4, dist="gaussian")
    agent.reset(rng)
    state_before = rng.bit_generator.state
    agent.update(obs=None, action=2, reward=1.0, next_obs=None)
    assert rng.bit_generator.state == state_before   # update drew nothing


# ----------------------------------------------------------------------
# 5. Env-agnostic: arm count is read from the env, no hardcoding of K.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("K", [2, 5, 10])
def test_reads_arm_count_from_env(K):
    means = np.linspace(0.1, 0.9, K)
    env = StochasticBanditEnv(means)
    agent = Thompson(env, dist="beta")
    assert agent.K == K
    res = run_episode(env, [agent], T=8000, seed=3)
    assert agent.greedy_arm() == int(np.argmax(means))
    assert (res.prices[:, 0] == int(np.argmax(means))).mean() >= 0.9


# ----------------------------------------------------------------------
# 6. Drops into a real game env unchanged (interface-generality smoke).
# ----------------------------------------------------------------------
def test_runs_on_bertrand_env():
    env = BertrandEnv(n=2, K=7, c=1.0)
    agents = [Thompson(env, dist="gaussian") for _ in range(2)]
    res = run_episode(env, agents, T=500, seed=1)
    assert res.prices.shape == (500, 2)
    assert res.prices.min() >= 0 and res.prices.max() < env.K
    assert np.isfinite(res.profits).all()
