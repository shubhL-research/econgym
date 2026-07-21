"""Homogeneous-good Bertrand market physics.

Extracted *verbatim* (semantics-preserving) from Paper-1's `simulation.py`
(`make_grid`, `step_profits`, `nash_profit`, `monopoly_profit`). The lowest
price serves one unit of inelastic demand; ties split equally; prices live on a
discrete linear grid.

The env owns only the *physics*: it holds no learning state and consumes RNG
only in ``reset`` (a single ``rng.integers(0, K, size=n)`` draw for the initial
joint price profile). Observations are memory-1: each agent sees its
opponent(s)' previous-period price index(es).
"""
from __future__ import annotations

import numpy as np

from ..core import Discrete, EconEnv


def make_grid(K, p_min=0.0, p_max=10.0):
    """Discrete linear price grid with ``K`` points on ``[p_min, p_max]``."""
    return np.linspace(p_min, p_max, K)


def step_profits(action_idx, grid, c):
    """Bertrand allocation: lowest price serves 1 unit of demand, ties split."""
    prices = grid[action_idx]
    pmin = prices.min()
    winners = prices == pmin
    q = np.where(winners, 1.0 / winners.sum(), 0.0)
    return (prices - c) * q


class BertrandEnv(EconEnv):
    """Discrete-price homogeneous-good Bertrand oligopoly.

    Subclasses :class:`~econgym.core.EconEnv`: the market physics, RNG usage, and
    numerical behaviour are IDENTICAL to v0 -- only the interface is generalised
    (``reset`` resolves the shared episode ``rng`` in the base and delegates to
    ``_reset``; ``step`` now returns the 5-tuple ``(obs, rewards, terminated,
    truncated, info)`` with ``terminated == truncated == False`` always).

    Parameters
    ----------
    n : int
        Number of firms.
    K : int
        Number of price-grid points.
    c : float
        Marginal cost.
    p_min, p_max : float
        Grid endpoints.

    Attributes
    ----------
    grid : np.ndarray
        The price grid, ``np.linspace(p_min, p_max, K)``.
    action_space : list[Discrete]
        Per-firm price-index action space (length ``n``, each ``Discrete(K)``).
    observation_space : list[Discrete]
        Per-firm opponent-profile observation space (length ``n``).
    n, K, c, p_min, p_max
        Exposed as read attributes.
    """

    metadata = {"name": "Bertrand", "simultaneous": True}

    def __init__(self, n: int = 2, K: int = 7, c: float = 1.0,
                 p_min: float = 0.0, p_max: float = 10.0) -> None:
        self.n = int(n)
        self.K = int(K)
        self.c = float(c)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.grid = make_grid(self.K, self.p_min, self.p_max)
        self._prev = None
        # Per-agent spaces (index = agent id). Action = price index in {0..K-1};
        # observation = the opponent price profile (K**(n-1) distinct profiles).
        self.action_space = [Discrete(self.K) for _ in range(self.n)]
        self.observation_space = [
            Discrete(self.K ** (self.n - 1)) for _ in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Observation encoding (memory-1, opponent price)
    # ------------------------------------------------------------------
    def _obs(self, idx):
        """Per-agent observation from a joint price-index profile ``idx``.

        For ``n == 2`` this returns ``[idx[1], idx[0]]`` (plain ints) so that
        agent 0 sees agent 1's price and vice versa -- byte-matching the
        original ``s = [prev[1], prev[0]]`` / ``ns = [a1, a0]``.

        For ``n > 2`` each agent sees a tuple of the other ``n-1`` agents'
        price indices (agent order ``0..n-1`` skipping self) -- a faithful
        generalisation (no original to match).
        """
        if self.n == 2:
            return [int(idx[1]), int(idx[0])]
        return [tuple(int(idx[j]) for j in range(self.n) if j != i)
                for i in range(self.n)]

    def _reset(self) -> list:
        """Draw the initial joint price profile and return per-agent obs.

        Called by :meth:`EconEnv.reset` with ``self.rng`` already resolved. Draws
        exactly ONE ``self.rng.integers(0, K, size=n)`` call -- the SAME single
        draw v0 made -- stores it, and returns ``[obs_0, ..., obs_{n-1}]``.
        """
        prev = self.rng.integers(0, self.K, size=self.n)
        self._prev = prev
        return self._obs(prev)

    def step(self, actions):
        """Advance one period.

        Parameters
        ----------
        actions : sequence of n price INDICES (ints).

        Returns
        -------
        obs : list length n
            ``obs[i]`` = opponent price index(es) for agent ``i``.
        rewards : np.ndarray length n
            Per-firm profit this period (Bertrand allocation).
        terminated : bool
            Always ``False`` (Bertrand is an infinite-horizon repeated game).
        truncated : bool
            Always ``False`` (the runner's ``T`` bounds the horizon externally).
        info : dict
            At least ``{'prices': np.ndarray of chosen price indices}``.
        """
        idx = np.asarray(actions, dtype=np.int64)
        rewards = step_profits(idx, self.grid, self.c)
        self._prev = idx
        obs = self._obs(idx)
        info = {"prices": idx}
        return obs, rewards, False, False, info

    # ------------------------------------------------------------------
    # Static / discrete benchmarks (verbatim formulas)
    # ------------------------------------------------------------------
    def nash_profit(self) -> float:
        """Per-firm profit at the discrete Bertrand-Nash benchmark:
        ``(p_comp - c) / n`` with ``p_comp = min{p in grid : p >= c}``
        (``grid.max()`` if none). Returns 0 only if a grid point equals ``c``."""
        above = self.grid[self.grid >= self.c]
        p_comp = above.min() if above.size else self.grid.max()
        return float((p_comp - self.c) / self.n)

    def monopoly_profit(self) -> float:
        """Per-firm profit at symmetric joint monopoly: ``(grid.max() - c) / n``."""
        return float((self.grid.max() - self.c) / self.n)

    def _p_comp(self) -> float:
        """Discrete competitive (Bertrand-Nash) price: ``min{p in grid : p >= c}``
        (``grid.max()`` if no grid point reaches ``c``). This is the price behind
        :meth:`nash_profit`."""
        above = self.grid[self.grid >= self.c]
        return float(above.min() if above.size else self.grid.max())

    def equilibrium(self) -> dict:
        """CLOSED-FORM (discrete-grid) benchmark for this Bertrand market.

        Homogeneous-good Bertrand: the *continuous* Nash is price = marginal cost
        with zero profit; on the discrete grid the competitive price is
        ``p_comp = min{p in grid : p >= c}`` and per-firm Nash profit is
        ``(p_comp - c) / n`` (exactly ``0`` when a grid point equals ``c``). The
        symmetric joint-monopoly benchmark caps prices at ``grid.max()`` for a
        per-firm profit of ``(grid.max() - c) / n``.
        """
        return {
            "nash_price": self._p_comp(),
            "nash_profit_per_firm": self.nash_profit(),        # (p_comp - c)/n
            "monopoly_price": float(self.grid.max()),
            "monopoly_profit_per_firm": self.monopoly_profit(),  # (p_max - c)/n
            "note": ("homogeneous Bertrand: continuous Nash p=c, pi=0; "
                     "discrete-grid p_comp = min{p in grid : p>=c}"),
        }
