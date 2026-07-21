"""n-firm Cournot quantity competition.

Model (contract v1, section 3.1)
--------------------------------
``n`` firms simultaneously choose quantities ``q_i >= 0``. Market (inverse)
demand is linear with a non-negativity floor,

    P = max(0, a - b*Q),        Q = sum_i q_i,

marginal cost is a constant ``c``, and each firm's payoff is

    pi_i = (P - c) * q_i.

Closed-form benchmark (the single source of truth)
--------------------------------------------------
Firm ``i`` maximises ``(a - b(q_i + Q_{-i}) - c) q_i``; the FOC
``a - b*Q_{-i} - 2*b*q_i - c = 0`` gives the best-response map

    BR_i(Q_{-i}) = max(0, (a - c - b*Q_{-i}) / (2b)),   Q_{-i} = sum_{j!=i} q_j.

Imposing symmetry ``q_j = q_i* for all j`` (so ``Q_{-i} = (n-1) q_i*``) yields the
textbook symmetric interior Nash

    q_i* = (a - c) / (b(n+1)),   Q* = n(a-c)/(b(n+1)),
    P*   = (a + n c)/(n+1),      pi_i* = (a - c)^2 / (b(n+1)^2).

The physics (``step``) implements ``pi_i = (max(0, a - b*Q) - c) q_i`` exactly;
``equilibrium()`` delegates to :func:`econgym.solvers.closed_form.cournot_nash`
so the env and its benchmark can never disagree.

A note on best-response dynamics. The *simultaneous* best-response map here is
**not** a contraction for ``n >= 3``: its Jacobian has eigenvalue ``-(n-1)/2``
along the aggregate direction (neutrally stable 2-cycle at ``n = 3``, divergent
for ``n >= 4``). A *relaxed* (damped) iteration
``x <- (1-lambda) x + lambda * BR(x)`` with ``lambda in (0, 4/(n+1))`` has the
**same** fixed point (the Nash above) and converges -- this is what the
validation test uses.
"""
from __future__ import annotations

import numpy as np

from ..core import Box, EconEnv
from ..solvers.closed_form import cournot_nash

__all__ = ["CournotEnv", "cournot_nash"]


class CournotEnv(EconEnv):
    """``n``-firm homogeneous-good Cournot quantity-competition environment.

    Subclasses :class:`~econgym.core.EconEnv`. Simultaneous-move: every firm
    picks a quantity ``q_i >= 0`` at once and ``step`` returns per-firm profits.
    Infinite-horizon / repeated (``terminated == truncated == False``); the
    runner's ``T`` bounds the horizon externally.

    The **native** action is a continuous quantity, ``Box(0, q_max)`` with the
    choke bound ``q_max = (a-c)/b`` (a single firm never profitably produces more
    than this). A discretised ``linspace(0, q_max, G)`` grid is also exposed as
    :attr:`qgrid` for index-based agents.

    Parameters
    ----------
    n : int
        Number of firms (default 3; ``n >= 1``, ``n = 1`` is the monopoly limit).
    a : float
        Demand intercept (choke price); requires ``a > c`` for an interior
        equilibrium.
    b : float
        Demand slope (``b > 0``).
    c : float
        Constant marginal cost.
    G : int
        Number of points on the discretised quantity grid ``linspace(0, q_max,
        G)`` (default 101), exposed as :attr:`qgrid` for index-based agents.

    Attributes
    ----------
    q_max : float
        Choke/monopoly quantity bound ``(a - c)/b``.
    qgrid : np.ndarray
        Discretised quantity grid ``np.linspace(0, q_max, G)``.
    action_space : list[Box]
        Per-firm NATIVE quantity action ``Box(0, q_max)`` (length ``n``).
    observation_space : list[Box]
        Per-firm observation = the other ``n-1`` firms' previous quantities
        (memory-1), ``Box(0, q_max, shape=(n-1,))``.
    n, a, b, c, G
        Exposed as read attributes.
    """

    metadata = {"name": "Cournot", "simultaneous": True}

    def __init__(self, n: int = 3, a: float = 100.0, b: float = 1.0,
                 c: float = 10.0, G: int = 101) -> None:
        self.n = int(n)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.G = int(G)
        if self.n < 1:
            raise ValueError(f"cournot needs n >= 1 firms, got n={self.n}")
        if self.b <= 0.0:
            raise ValueError(f"demand slope b must be > 0, got b={self.b}")
        if self.a <= self.c:
            raise ValueError(
                f"cournot requires a > c for an interior equilibrium, got "
                f"a={self.a}, c={self.c}"
            )
        self.q_max = (self.a - self.c) / self.b          # choke / monopoly bound
        self.qgrid = np.linspace(0.0, self.q_max, self.G)
        self._prev = None
        # NATIVE continuous quantity action; discretised grid = self.qgrid.
        self.action_space = [
            Box(low=0.0, high=self.q_max) for _ in range(self.n)
        ]
        # Observation = the other firms' previous quantities (memory-1).
        self.observation_space = [
            Box(low=0.0, high=self.q_max, shape=(self.n - 1,))
            for _ in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Observation encoding (memory-1: each firm sees rivals' quantities)
    # ------------------------------------------------------------------
    def _obs(self, quantities) -> list:
        """Per-agent obs from a joint quantity profile.

        Firm ``i`` observes the vector of the other ``n-1`` firms' quantities
        (agent order ``0..n-1`` skipping self). For ``n == 1`` this is an empty
        array.
        """
        q = np.asarray(quantities, dtype=np.float64)
        return [
            np.array([q[j] for j in range(self.n) if j != i], dtype=np.float64)
            for i in range(self.n)
        ]

    def _reset(self) -> list:
        """Draw an initial quantity profile from the shared ``self.rng``.

        Consumes exactly ONE draw (``rng.uniform(0, q_max, size=n)``) -- honoring
        the seeding contract -- stores it as the previous profile, and returns
        the per-agent observation list.
        """
        prev = self.rng.uniform(0.0, self.q_max, size=self.n)
        self._prev = prev
        return self._obs(prev)

    # ------------------------------------------------------------------
    # Physics (pure functions of the joint quantity profile)
    # ------------------------------------------------------------------
    def price(self, quantities) -> float:
        """Market price ``P = max(0, a - b*Q)`` for a joint quantity profile."""
        q = np.maximum(np.asarray(quantities, dtype=np.float64), 0.0)
        return float(max(0.0, self.a - self.b * q.sum()))

    def profit(self, quantities) -> np.ndarray:
        """Vector of per-firm profits ``pi_i = (max(0, a - b*Q) - c) * q_i``.

        Pure function of the joint quantity profile; used by ``step`` and by the
        validation tests. Negative quantities are floored at 0 (unphysical);
        prices are floored at 0 by the demand curve, so a firm that over-produces
        into the ``P = 0`` region earns the (negative) ``-c * q_i``.
        """
        q = np.maximum(np.asarray(quantities, dtype=np.float64), 0.0)
        P = max(0.0, self.a - self.b * q.sum())
        return (P - self.c) * q

    def best_response(self, quantities) -> np.ndarray:
        """Vector of myopic best responses ``BR_i(Q_{-i})`` to a joint profile.

        ``BR_i(Q_{-i}) = max(0, (a - c - b*Q_{-i}) / (2b))`` with
        ``Q_{-i} = sum_{j != i} q_j``. This is the map whose fixed point is the
        Nash equilibrium (used by the best-response-iteration validation, with
        relaxation -- see the module docstring).
        """
        q = np.maximum(np.asarray(quantities, dtype=np.float64), 0.0)
        Q_minus = q.sum() - q
        return np.maximum(0.0, (self.a - self.c - self.b * Q_minus) / (2.0 * self.b))

    def step(self, actions):
        """Advance one period.

        Parameters
        ----------
        actions : sequence of ``n`` native quantities in ``[0, q_max]``.

        Returns
        -------
        obs : list length n
            ``obs[i]`` = the other firms' quantities this period.
        rewards : np.ndarray length n
            ``pi_i = (max(0, a - b*Q) - c) * q_i``.
        terminated, truncated : bool
            Always ``False`` (repeated Cournot; the runner's ``T`` bounds it).
        info : dict
            ``{'quantities', 'Q', 'P'}`` -- the realised profile, aggregate
            quantity, and market price.
        """
        q = np.maximum(np.asarray(actions, dtype=np.float64), 0.0)
        Q = float(q.sum())
        P = max(0.0, self.a - self.b * Q)
        rewards = (P - self.c) * q
        self._prev = q
        info = {"quantities": q, "Q": Q, "P": P}
        return self._obs(q), rewards, False, False, info

    # ------------------------------------------------------------------
    # Closed-form benchmark
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM benchmark: delegates to
        :func:`econgym.solvers.closed_form.cournot_nash`.

        Returns ``{"q_i": (a-c)/(b(n+1)), "Q": n(a-c)/(b(n+1)),
        "P": (a+nc)/(n+1), "profit_i": (a-c)^2/(b(n+1)^2)}``.
        """
        return cournot_nash(self.a, self.b, self.c, self.n)
