"""Linear public-goods (Voluntary Contribution Mechanism) game.

Model (contract v1, section 3.5)
--------------------------------
``n`` players each hold an endowment ``w`` and simultaneously contribute
``c_i in [0, w]`` to a common pool. The pool is scaled by a **marginal
per-capita return** ``r`` and shared equally, so each player's payoff is

    pi_i = w - c_i + r * sum_j c_j        (the j-sum INCLUDES i)

with the social-dilemma condition ``1/n < r < 1``.

Closed-form benchmark (the single source of truth)
--------------------------------------------------
Own-contribution marginal payoff:

    d pi_i / d c_i = -1 + r < 0          (since r < 1)

so contributing anything strictly lowers a player's own payoff regardless of
what everyone else does: the unique Nash equilibrium is **free-riding**
``c_i = 0`` for all ``i`` (a strictly dominant strategy), giving ``pi_i = w``.

Social welfare, however, is

    sum_i pi_i = n*w + (r*n - 1) * sum_j c_j

whose slope in every ``c_j`` is ``r*n - 1 > 0`` (since ``r > 1/n``), so welfare
is maximised at **full contribution** ``c_i = w`` for all ``i``, giving
``pi_i = r*n*w``. The per-player Nash-vs-optimum gap is ``(r*n - 1)*w``.

With defaults ``n=4, w=20, r=0.4``: ``nash_payoff = 20``,
``optimum_payoff = 0.4*4*20 = 32``, ``gap_per_player = 12``.

The physics (``step``) implements ``pi_i = w - c_i + r * sum_j c_j`` exactly;
``equilibrium()`` delegates to
:func:`econgym.solvers.closed_form.public_goods_nash` (the single source of
truth) so the env and its benchmark can never disagree. That function is
re-exported here (and at the package root) as ``public_goods_nash``.
"""
from __future__ import annotations

import numpy as np

from ..core import Box, EconEnv
from ..solvers.closed_form import public_goods_nash


class PublicGoodsEnv(EconEnv):
    """Linear public-goods (VCM) contribution game.

    Subclasses :class:`~econgym.core.EconEnv`. Simultaneous-move: every player
    chooses a contribution ``c_i in [0, w]`` at once; ``step`` returns per-agent
    payoffs. Infinite-horizon / repeated (``terminated == truncated == False``);
    the runner's ``T`` bounds the horizon externally.

    Parameters
    ----------
    n : int
        Number of players (default 4).
    w : float
        Per-player endowment (default 20.0).
    r : float
        Marginal per-capita return / MPCR (default 0.4). Must satisfy the
        social-dilemma condition ``1/n < r < 1`` (below ``1/n`` full
        contribution is not socially optimal; above ``1`` contributing is
        privately optimal -- either way the closed-form Nash/optimum corners
        would not hold, so the constructor rejects it).
    G : int
        Number of points on the discretised contribution grid ``linspace(0, w,
        G)`` (default 21), exposed as :attr:`cgrid` for index-based agents.

    Attributes
    ----------
    cgrid : np.ndarray
        Discretised contribution grid ``np.linspace(0, w, G)``.
    action_space : list[Box]
        Per-player NATIVE contribution action, ``Box(0, w)`` (length ``n``).
    observation_space : list[Box]
        Per-player observation = the other ``n-1`` players' previous
        contributions (memory-1), ``Box(0, w, shape=(n-1,))``.
    n, w, r, G
        Exposed as read attributes.
    """

    metadata = {"name": "PublicGoods", "simultaneous": True}

    def __init__(self, n: int = 4, w: float = 20.0, r: float = 0.4,
                 G: int = 21) -> None:
        self.n = int(n)
        self.w = float(w)
        self.r = float(r)
        self.G = int(G)
        if self.n < 2:
            raise ValueError(f"public_goods needs n >= 2, got n={self.n}")
        if self.w <= 0.0:
            raise ValueError(f"endowment w must be > 0, got w={self.w}")
        # Social-dilemma regime: 1/n < r < 1 (see class docstring). Outside this
        # band the closed-form Nash/optimum corners do not both hold, so refuse.
        if not (1.0 / self.n < self.r < 1.0):
            raise ValueError(
                f"public_goods requires 1/n < r < 1 (social dilemma); got "
                f"r={self.r} with 1/n={1.0 / self.n} (n={self.n})"
            )
        self.cgrid = np.linspace(0.0, self.w, self.G)
        self._prev = None
        # NATIVE continuous contribution action; discretised grid = self.cgrid.
        self.action_space = [Box(low=0.0, high=self.w) for _ in range(self.n)]
        # Observation = others' previous contributions (memory-1).
        self.observation_space = [
            Box(low=0.0, high=self.w, shape=(self.n - 1,))
            for _ in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Observation encoding (memory-1: each agent sees others' contributions)
    # ------------------------------------------------------------------
    def _obs(self, contribs) -> list:
        """Per-agent obs from a joint contribution profile ``contribs``.

        Agent ``i`` observes the vector of the other ``n-1`` players'
        contributions (agent order ``0..n-1`` skipping self).
        """
        c = np.asarray(contribs, dtype=np.float64)
        return [
            np.array([c[j] for j in range(self.n) if j != i], dtype=np.float64)
            for i in range(self.n)
        ]

    def _reset(self) -> list:
        """Draw an initial contribution profile from the shared ``self.rng``.

        Consumes exactly ONE draw (``rng.uniform(0, w, size=n)``) -- honoring
        the seeding contract -- stores it as the previous profile, and returns
        the per-agent observation list.
        """
        prev = self.rng.uniform(0.0, self.w, size=self.n)
        self._prev = prev
        return self._obs(prev)

    def payoff(self, contributions) -> np.ndarray:
        """Vector of payoffs ``pi_i = w - c_i + r * sum_j c_j`` (j-sum includes i).

        Pure function of the joint contribution profile; used by ``step`` and by
        the validation tests. Contributions are clipped to ``[0, w]``.
        """
        c = np.clip(np.asarray(contributions, dtype=np.float64), 0.0, self.w)
        return self.w - c + self.r * c.sum()

    def step(self, actions):
        """Advance one period.

        Parameters
        ----------
        actions : sequence of ``n`` native contributions in ``[0, w]``.

        Returns
        -------
        obs : list length n
            ``obs[i]`` = the other players' contributions this period.
        rewards : np.ndarray length n
            ``pi_i = w - c_i + r * sum_j c_j``.
        terminated, truncated : bool
            Always ``False`` (repeated VCM; the runner's ``T`` bounds it).
        info : dict
            ``{'contributions', 'total', 'public_good', 'per_capita_return'}``.
        """
        c = np.clip(np.asarray(actions, dtype=np.float64), 0.0, self.w)
        total = float(c.sum())
        rewards = self.w - c + self.r * total
        self._prev = c
        info = {
            "contributions": c,
            "total": total,
            "public_good": self.r * total,     # value returned to EACH player
            "per_capita_return": self.r,
        }
        return self._obs(c), rewards, False, False, info

    # ------------------------------------------------------------------
    # Closed-form benchmark
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM benchmark: delegates to :func:`public_goods_nash`.

        Returns ``nash_contribution=0``, ``nash_payoff=w``,
        ``optimum_contribution=w``, ``optimum_payoff=r*n*w``,
        ``gap_per_player=(r*n-1)*w``.
        """
        return public_goods_nash(self.w, self.r, self.n)
