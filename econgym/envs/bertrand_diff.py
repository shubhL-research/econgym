"""Differentiated-good (linear-demand) Bertrand price competition.

Model (contract v1, section 3.2)
--------------------------------
``n`` firms simultaneously set prices ``p_i``. Demand for firm ``i`` is linear
in own and rival prices,

    q_i = alpha - beta * p_i + gamma * sum_{j != i} p_j ,      0 < gamma < beta ,

with constant marginal cost ``c`` and profit ``pi_i = (p_i - c) * q_i`` (demand
clamped at 0 in the simulator; the interior equilibrium keeps ``q_i > 0``).

Closed-form benchmark -- FOC derivation (the single source of truth)
--------------------------------------------------------------------
Profit ``pi_i = (p_i - c)(alpha - beta p_i + gamma S_{-i})`` with
``S_{-i} = sum_{j != i} p_j``. The first-order condition
``d pi_i / d p_i = 0`` is

    (alpha - beta p_i + gamma S_{-i}) + (p_i - c)(-beta) = 0
    alpha + beta c + gamma S_{-i} - 2 beta p_i = 0
    =>  BR_i(S_{-i}) = (alpha + beta c + gamma S_{-i}) / (2 beta) .   [best-response map]

Imposing symmetry ``p_i = p*`` (so ``S_{-i} = (n-1) p*``):

    2 beta p* = alpha + beta c + gamma (n-1) p*
    =>  p* = (alpha + beta c) / (2 beta - gamma (n-1)) .   [symmetric Nash price]

Derived quantities ``q_i* = alpha - beta p* + gamma (n-1) p*`` and
``pi_i* = (p* - c) q_i*``. With the defaults ``alpha=10, beta=2, gamma=1, c=1,
n=2``: ``p* = (10 + 2)/(4 - 1) = 4.0``, ``q* = 10 - 8 + 4 = 6``,
``pi* = (4 - 1)*6 = 18``.

The physics (``step``) implements ``pi_i = (p_i - c) * max(0, q_i)`` exactly;
``equilibrium()`` delegates to :func:`econgym.solvers.closed_form.bertrand_diff_nash`
so the env and its benchmark can never disagree.
"""
from __future__ import annotations

import numpy as np

from ..core import Box, Discrete, EconEnv


class BertrandDiffEnv(EconEnv):
    """Differentiated-good (linear-demand) Bertrand price-competition oligopoly.

    Subclasses :class:`~econgym.core.EconEnv`. Simultaneous-move: every firm sets
    a price at once; ``step`` returns per-firm profits. Infinite-horizon /
    repeated (``terminated == truncated == False``); the runner's ``T`` bounds
    the horizon externally.

    Parameters
    ----------
    n : int
        Number of firms (default 2).
    alpha : float
        Demand intercept (default 10.0). Must be > 0.
    beta : float
        Own-price demand slope (default 2.0). Must be > 0.
    gamma : float
        Cross-price demand slope (default 1.0). Must satisfy ``0 < gamma < beta``.
    c : float
        Constant marginal cost (default 1.0).
    G : int
        Number of points on the discretised price grid ``linspace(c, p_cap, G)``
        (default 101), exposed as :attr:`pgrid` for index-based agents.
    p_cap : float, optional
        Upper bound of the native price action / grid. Defaults to
        ``alpha/beta + c`` (the choke price ``alpha/beta`` marked up by cost),
        which comfortably brackets the interior equilibrium.

    Notes
    -----
    Interior/stability requires ``2*beta - gamma*(n-1) > 0``, equivalently the
    best-response contraction factor ``gamma*(n-1)/(2*beta) < 1``. The
    constructor rejects parameters outside this regime (the closed-form
    symmetric Nash would otherwise be non-interior or unstable). Defaults give
    ``2*2 - 1*1 = 3 > 0`` (contraction factor ``0.25``).

    Attributes
    ----------
    pgrid : np.ndarray
        Discretised price grid ``np.linspace(c, p_cap, G)``.
    action_space : list[Box]
        Per-firm NATIVE price action, ``Box(c, p_cap)`` (length ``n``).
    action_space_discrete : list[Discrete]
        Per-firm index action over :attr:`pgrid` (length ``n``), for index-based
        agents that map an index to ``pgrid[index]``.
    observation_space : list[Box]
        Per-firm observation = the other ``n-1`` firms' previous prices
        (memory-1), ``Box(c, p_cap, shape=(n-1,))``.
    n, alpha, beta, gamma, c, G, p_cap
        Exposed as read attributes.
    """

    metadata = {"name": "BertrandDiff", "simultaneous": True}

    def __init__(self, n: int = 2, alpha: float = 10.0, beta: float = 2.0,
                 gamma: float = 1.0, c: float = 1.0, G: int = 101,
                 p_cap: float | None = None) -> None:
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.c = float(c)
        self.G = int(G)
        if self.n < 2:
            raise ValueError(f"bertrand_diff needs n >= 2, got n={self.n}")
        if self.alpha <= 0.0:
            raise ValueError(f"demand intercept alpha must be > 0, got alpha={self.alpha}")
        if self.beta <= 0.0:
            raise ValueError(f"own-price slope beta must be > 0, got beta={self.beta}")
        if not (0.0 < self.gamma < self.beta):
            raise ValueError(
                f"bertrand_diff requires 0 < gamma < beta (substitutes, own-price "
                f"dominant); got gamma={self.gamma}, beta={self.beta}"
            )
        # Interior/stability: 2*beta - gamma*(n-1) > 0  <=>  contraction < 1.
        denom = 2.0 * self.beta - self.gamma * (self.n - 1)
        if denom <= 0.0:
            raise ValueError(
                f"non-interior differentiated Bertrand: require "
                f"2*beta - gamma*(n-1) > 0, got 2*{self.beta} - {self.gamma}*"
                f"{self.n - 1} = {denom} (contraction factor "
                f"{self.gamma * (self.n - 1) / (2.0 * self.beta)} >= 1)"
            )
        self.p_cap = float(p_cap) if p_cap is not None else self.alpha / self.beta + self.c
        if self.p_cap <= self.c:
            raise ValueError(f"p_cap must exceed c; got p_cap={self.p_cap}, c={self.c}")
        self.pgrid = np.linspace(self.c, self.p_cap, self.G)
        self._prev = None
        # NATIVE continuous price action; discretised index grid = self.pgrid.
        self.action_space = [Box(low=self.c, high=self.p_cap) for _ in range(self.n)]
        self.action_space_discrete = [Discrete(self.G) for _ in range(self.n)]
        # Observation = others' previous prices (memory-1).
        self.observation_space = [
            Box(low=self.c, high=self.p_cap, shape=(self.n - 1,))
            for _ in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Observation encoding (memory-1: each firm sees rivals' prices)
    # ------------------------------------------------------------------
    def _obs(self, prices) -> list:
        """Per-firm obs from a joint price profile ``prices``.

        Firm ``i`` observes the vector of the other ``n-1`` firms' prices (agent
        order ``0..n-1`` skipping self).
        """
        p = np.asarray(prices, dtype=np.float64)
        return [
            np.array([p[j] for j in range(self.n) if j != i], dtype=np.float64)
            for i in range(self.n)
        ]

    def _reset(self) -> list:
        """Draw an initial price profile from the shared ``self.rng``.

        Consumes exactly ONE draw (``rng.uniform(c, p_cap, size=n)``) -- honoring
        the seeding contract -- stores it as the previous profile, and returns
        the per-firm observation list.
        """
        prev = self.rng.uniform(self.c, self.p_cap, size=self.n)
        self._prev = prev
        return self._obs(prev)

    # ------------------------------------------------------------------
    # Market physics (pure functions of the joint price profile)
    # ------------------------------------------------------------------
    def demand(self, prices) -> np.ndarray:
        """Vector of demands ``q_i = alpha - beta p_i + gamma * sum_{j!=i} p_j``.

        Clamped at 0 (a firm priced out of the market serves no demand). Pure
        function of the joint price profile; used by :meth:`profit` and ``step``.
        """
        p = np.asarray(prices, dtype=np.float64)
        cross = p.sum() - p                      # S_{-i} = sum_{j != i} p_j
        q = self.alpha - self.beta * p + self.gamma * cross
        return np.maximum(q, 0.0)

    def profit(self, prices) -> np.ndarray:
        """Vector of profits ``pi_i = (p_i - c) * max(0, q_i)``.

        Pure function of the joint price profile; used by ``step`` and by the
        validation tests (deviation / physics checks).
        """
        p = np.asarray(prices, dtype=np.float64)
        return (p - self.c) * self.demand(p)

    def best_response(self, prices) -> np.ndarray:
        """Vector of myopic best-response prices to the current profile.

        ``BR_i(S_{-i}) = (alpha + beta c + gamma S_{-i}) / (2 beta)`` with
        ``S_{-i} = sum_{j != i} p_j``. This is the FOC best-response map; its
        symmetric fixed point is the closed-form Nash price. Used by the
        best-response-iteration convergence test.
        """
        p = np.asarray(prices, dtype=np.float64)
        cross = p.sum() - p                      # S_{-i}
        return (self.alpha + self.beta * self.c + self.gamma * cross) / (2.0 * self.beta)

    def step(self, actions):
        """Advance one period.

        Parameters
        ----------
        actions : sequence of ``n`` native prices.

        Returns
        -------
        obs : list length n
            ``obs[i]`` = the rival firms' prices this period.
        rewards : np.ndarray length n
            ``pi_i = (p_i - c) * max(0, q_i)``.
        terminated, truncated : bool
            Always ``False`` (repeated price competition; ``T`` bounds it).
        info : dict
            ``{'prices', 'demand'}``.
        """
        p = np.asarray(actions, dtype=np.float64)
        q = self.demand(p)
        rewards = (p - self.c) * q
        self._prev = p
        info = {"prices": p, "demand": q}
        return self._obs(p), rewards, False, False, info

    # ------------------------------------------------------------------
    # Closed-form benchmark
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM symmetric interior Nash benchmark.

        Delegates to :func:`econgym.solvers.closed_form.bertrand_diff_nash`
        (single source of truth) so the env and its benchmark can never
        disagree. Returns ``{"p", "q_i", "profit_i", "contraction_factor"}``:
        ``p = (alpha + beta c)/(2 beta - gamma(n-1))``,
        ``q_i = alpha - beta p + gamma(n-1)p``, ``profit_i = (p - c) q_i``.
        """
        # Lazy import keeps env import robust even while the shared solvers
        # module is being written concurrently; delegation still binds at call.
        from ..solvers.closed_form import bertrand_diff_nash
        return bertrand_diff_nash(self.alpha, self.beta, self.gamma, self.c, self.n)
