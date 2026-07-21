"""Thompson-sampling (posterior-sampling) bandit agent.

Thompson sampling maintains a Bayesian posterior over each arm's mean reward and,
at every step, draws one sample from each arm's posterior and plays the arm with
the largest sample. The posterior sharpens around the true means as data arrives,
so the sampled maximiser lands on the best arm with probability -> 1: the agent
provably concentrates its pulls on the optimal action and its cumulative regret
grows only logarithmically in the horizon (Thompson 1933; Agrawal & Goyal 2012,
2013 -- ``O(sqrt(K T ln T))`` worst-case / ``O((sum_i 1/Delta_i) ln T)``
problem-dependent, matching the Lai-Robbins lower bound up to constants).

Two conjugate posteriors are supported (contract Section 4.1: "Gaussian/Beta"):

  * ``dist="gaussian"`` -- Normal-mean model with known observation variance.
    Prior ``N(prior_mean, prior_var)``, likelihood variance ``obs_var``. After
    ``n_a`` pulls of arm ``a`` with reward-sum ``S_a`` the posterior is
    ``N(mu_a, v_a)`` with precision ``1/prior_var + n_a/obs_var``,
    ``mu_a = (prior_mean/prior_var + S_a/obs_var) * v_a``, ``v_a = 1/precision``.
    Defaults (``prior_mean=0, prior_var=1, obs_var=1``) give the Agrawal-Goyal
    ``N(0,1)``-prior sampler ``theta_a ~ N(S_a/(1+n_a), 1/(1+n_a))``. Handles
    rewards of any sign/scale (use for general bounded rewards).

  * ``dist="beta"`` -- Beta-Bernoulli model for rewards in ``[0, 1]``. Prior
    ``Beta(alpha0, beta0)``; a reward ``r`` updates ``alpha += r``, ``beta += 1-r``
    (exact Bayes for Bernoulli rewards; ``r`` is clipped to ``[0,1]`` so it stays
    valid for any bounded-reward env). Posterior sample ``theta_a ~ Beta(alpha_a,
    beta_a)``.

The agent is a genuine bandit: it IGNORES the observation (no per-state table) and
reads only how many discrete actions the env exposes, so it is env-agnostic -- it
runs on any :class:`~econgym.core.EconEnv` whose per-agent action set is
``Discrete`` (Bertrand price index, a stochastic-bandit arm, ...). Following the
Agent contract, ``act`` draws from the shared ``rng`` (exactly ``K`` samples every
step -- this is the one agent that samples every step by design) and ``update``
consumes NO ``rng``.
"""
from __future__ import annotations

import numpy as np

from .base import Agent


class Thompson(Agent):
    """Thompson-sampling bandit over an env's ``K`` discrete actions.

    Parameters
    ----------
    env : EconEnv, optional
        Source of the arm count. ``env.K`` is used if present, else
        ``env.action_space[0].n``. Ignored if ``n_actions`` is given.
    n_actions : int, optional
        Explicit number of arms (overrides ``env``). Exactly one of ``env`` /
        ``n_actions`` must be supplied.
    dist : {"gaussian", "beta"}
        Posterior family (see module docstring).
    prior_mean, prior_var, obs_var : float
        Gaussian-mode hyperparameters (defaults reproduce the Agrawal-Goyal
        ``N(0,1)``-prior sampler).
    alpha0, beta0 : float
        Beta-mode prior pseudo-counts (default ``1, 1`` = uniform prior).
    """

    def __init__(self, env=None, *, n_actions: int | None = None,
                 dist: str = "gaussian",
                 prior_mean: float = 0.0, prior_var: float = 1.0,
                 obs_var: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0) -> None:
        self.K = self._infer_K(env, n_actions)
        if dist not in ("gaussian", "beta"):
            raise ValueError(f"dist must be 'gaussian' or 'beta', got {dist!r}")
        self.dist = dist
        # Gaussian hyperparameters
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.obs_var = float(obs_var)
        # Beta hyperparameters
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        # per-arm state (allocated in reset)
        self.counts = None      # n_a       (gaussian)
        self.sums = None        # S_a       (gaussian)
        self.alpha = None       # alpha_a   (beta)
        self.beta = None        # beta_a    (beta)
        self._pol_snap = None

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_K(env, n_actions) -> int:
        if n_actions is not None:
            return int(n_actions)
        if env is None:
            raise ValueError("Thompson needs either env or n_actions")
        if hasattr(env, "K"):
            return int(env.K)
        space = env.action_space[0]
        if not hasattr(space, "n"):
            raise ValueError(
                "Thompson requires a Discrete action space (with attribute .n)"
            )
        return int(space.n)

    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        """Reset posterior to the prior. Consumes NO ``rng`` (the sampling that
        defines the reproducible stream happens in ``act``); ``track_conv`` is
        accepted for a uniform signature and does not branch the rng stream."""
        self.counts = np.zeros(self.K, dtype=np.float64)
        self.sums = np.zeros(self.K, dtype=np.float64)
        self.alpha = np.full(self.K, self.alpha0, dtype=np.float64)
        self.beta = np.full(self.K, self.beta0, dtype=np.float64)
        self._pol_snap = None

    # ------------------------------------------------------------------
    def _posterior_mean(self) -> np.ndarray:
        """Current posterior mean per arm (the greedy value estimate)."""
        if self.dist == "gaussian":
            prec = 1.0 / self.prior_var + self.counts / self.obs_var
            return (self.prior_mean / self.prior_var + self.sums / self.obs_var) / prec
        return self.alpha / (self.alpha + self.beta)

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        """Draw one posterior sample per arm and play the arg-max (first-max
        tie-break, no extra rng). The observation is ignored (bandit)."""
        if self.dist == "gaussian":
            prec = 1.0 / self.prior_var + self.counts / self.obs_var
            var = 1.0 / prec
            mean = (self.prior_mean / self.prior_var + self.sums / self.obs_var) * var
            samples = mean + np.sqrt(var) * rng.standard_normal(self.K)
        else:  # beta
            samples = rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        """Bayesian posterior update for the pulled arm. Consumes NO ``rng``."""
        a = int(action)
        r = float(reward)
        if self.dist == "gaussian":
            self.counts[a] += 1.0
            self.sums[a] += r
        else:  # beta: reward in [0,1]; clip so any bounded-reward env stays valid
            rc = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
            self.alpha[a] += rc
            self.beta[a] += 1.0 - rc

    # ------------------------------------------------------------------
    def greedy_arm(self) -> int:
        """Arm with the highest posterior mean (the exploitation choice)."""
        return int(np.argmax(self._posterior_mean()))

    # -- optional convergence hooks (scalar greedy-arm comparison) --
    def snapshot_policy(self) -> None:
        self._pol_snap = self.greedy_arm()

    def policy_stability(self) -> float:
        if self._pol_snap is None:
            return 1.0
        return float(self._pol_snap == self.greedy_arm())

    # -- visitation diagnostics --
    def cells_visited(self):
        return int((self.counts > 0).sum()) if self.counts is not None else None

    def total_cells(self):
        return int(self.K)

    def min_visit(self):
        if self.counts is None:
            return None
        nz = self.counts[self.counts > 0]
        return int(nz.min()) if nz.size else 0
