"""Second-price (Vickrey) sealed-bid auction.

``n`` bidders draw iid private values ``v_i ~ Uniform[0, 1]`` (revealed to their
owner at ``reset``) and simultaneously submit bids ``b_i >= 0``. The highest bid
wins the single item and **pays the second-highest bid**; ties for the top bid
split the item uniformly at random (a measure-zero event under continuous
values). The winner's payoff is ``v_i - (second-highest bid)``; every loser
earns ``0``.

The key economic fact this env encodes is that truthful bidding ``b(v) = v`` is a
*weakly dominant* strategy -- stronger than a mere Bayes-Nash equilibrium. Fix
any opponent bids and let ``m`` be the highest of them; then a bidder's payoff as
a function of their own bid ``b`` depends on the opponents ONLY through ``m``:
win (pay ``m``) iff ``b > m``, lose iff ``b < m``. Truthful ``b = v`` wins exactly
the profitable auctions (``v > m`` -> surplus ``v - m > 0``) and skips the
unprofitable ones, so no deviation can do strictly better against any opponent
profile. This dominance is ``n``-independent; ``n`` enters only the expected
revenue ``E[R] = (n-1)/(n+1)`` (the mean second-order statistic of ``n`` iid
``U[0,1]`` draws), which by revenue equivalence matches the first-price auction.

One-shot game: ``step`` returns ``terminated=True``. The env owns only the
auction physics and consumes RNG solely in ``reset`` (one ``rng.uniform`` draw of
the ``n`` private values) and, only on an exact top-bid tie, one integer draw to
break it.
"""
from __future__ import annotations

import numpy as np

from ..core import Box, Discrete, EconEnv
from ..solvers import closed_form


class SecondPriceEnv(EconEnv):
    """Second-price (Vickrey) sealed-bid auction with iid ``Uniform[0,1]`` values.

    Parameters
    ----------
    n : int
        Number of bidders. Default 2.
    G : int
        Size of the optional discrete bid grid ``env.bgrid = linspace(0, 1, G)``
        exposed for index-based ``Agent`` s. The *native* action is the
        continuous bid (a ``Box(0, 1)``); the grid is a convenience only.

    Attributes
    ----------
    action_space : list[Box]
        Per-bidder native bid space, each ``Box(0.0, 1.0)`` (a scalar in
        ``[0, 1]``). Bids outside the box are accepted by ``step`` (not clipped);
        the space documents the equilibrium-relevant range.
    observation_space : list[Box]
        Per-bidder own-value space, each ``Box(0.0, 1.0)``.
    bgrid : np.ndarray
        Optional discrete bid grid ``linspace(0, 1, G)`` for index agents.
    G : int
        Number of grid points.
    """

    metadata = {"name": "SecondPrice", "simultaneous": True}

    def __init__(self, n: int = 2, G: int = 101) -> None:
        self.n = int(n)
        self.G = int(G)
        # Native action = a bid in [0, 1]; observation = own private value in [0, 1].
        self.action_space = [Box(low=0.0, high=1.0) for _ in range(self.n)]
        self.observation_space = [Box(low=0.0, high=1.0) for _ in range(self.n)]
        # Optional discrete bid grid for index-based agents (convenience only).
        self.bgrid = np.linspace(0.0, 1.0, self.G)
        self._values = None

    # ------------------------------------------------------------------
    # Reset: draw the iid private values, reveal each to its owner.
    # ------------------------------------------------------------------
    def _reset(self) -> list:
        """Draw ``n`` iid ``Uniform[0,1]`` private values from the shared ``rng``
        and return them as the per-agent observation list (each bidder observes
        only its own value)."""
        self._values = self.rng.uniform(0.0, 1.0, size=self.n)
        return [float(v) for v in self._values]

    # ------------------------------------------------------------------
    # Step: run the one-shot auction.
    # ------------------------------------------------------------------
    def step(self, actions):
        """Run the sealed-bid auction for one (and only) round.

        Parameters
        ----------
        actions : sequence of ``n`` native bids (floats). Index-based agents
            should map their action index through ``env.bgrid`` before calling.

        Returns
        -------
        obs : list length n
            The bidders' own values (unchanged; a terminal observation).
        rewards : np.ndarray length n
            ``v_i - second_price`` for the winner, ``0`` for everyone else.
        terminated : bool
            Always ``True`` (one-shot game).
        truncated : bool
            Always ``False``.
        info : dict
            ``{"winner": int, "price": float, "bids": np.ndarray,
               "values": np.ndarray}``.
        """
        bids = np.asarray(actions, dtype=np.float64)
        if bids.shape != (self.n,):
            raise ValueError(f"expected {self.n} bids, got shape {bids.shape}")

        top = float(bids.max())
        winners = np.flatnonzero(bids == top)
        if winners.size == 1:
            winner = int(winners[0])
        else:  # exact top-bid tie -> split uniformly at random (measure zero)
            winner = int(winners[int(self.rng.integers(winners.size))])

        # Second-highest bid = the price the winner pays. Because the winner holds
        # the maximum bid, the second-highest bid overall equals the highest bid
        # among the opponents (== the max opponent bid `m`).
        second_price = float(np.sort(bids)[-2])

        rewards = np.zeros(self.n, dtype=np.float64)
        rewards[winner] = float(self._values[winner]) - second_price

        obs = [float(v) for v in self._values]
        info = {
            "winner": winner,
            "price": second_price,
            "bids": bids,
            "values": np.asarray(self._values, dtype=np.float64),
        }
        return obs, rewards, True, False, info

    # ------------------------------------------------------------------
    # Closed-form benchmark (delegated -- single source of truth).
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM benchmark: truthful ``b(v)=v`` is weakly dominant and the
        expected revenue is ``(n-1)/(n+1)``.

        Delegates to :func:`econgym.solvers.closed_form.second_price_bne` so the
        env and solver can never disagree.
        """
        return closed_form.second_price_bne(self.n)
