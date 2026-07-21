"""First-price sealed-bid auction (iid Uniform[0,1] private values).

Model (CONTRACT §3.3)
---------------------
``n`` bidders draw iid private values ``v_i ~ Uniform[0, 1]`` (revealed to bidder
``i`` at ``reset``). Each submits a bid ``b_i >= 0``. The **highest bid wins and
pays its own bid** (that is what makes it *first-price* / pay-your-bid); ties are
split by choosing a winner uniformly at random among the tied top bids (a
measure-zero event under continuous values). Payoffs::

    winner i :  v_i - b_i
    losers   :  0

The env owns only the *physics*. It consumes RNG in ``reset`` (one
``rng.uniform(0, 1, size=n)`` draw for the private values) and, only if a genuine
tie for the top bid occurs, one ``rng.choice`` to break it. The auction is
**one-shot**: ``step`` returns ``terminated=True``.

Closed-form benchmark (symmetric Bayes-Nash equilibrium)
--------------------------------------------------------
With ``n`` symmetric bidders and iid ``U[0,1]`` values, the symmetric BNE bid
function and the resulting expected seller revenue are::

    b(v)          = ((n - 1) / n) * v
    E[revenue]    = (n - 1) / (n + 1)

Derivation of ``b(v)``.  Suppose every opponent plays ``b(v)=((n-1)/n) v``.  A
bidder with value ``v`` bidding ``b`` wins iff *every* opponent's value satisfies
``v_j < b * n/(n-1)``, which for ``b in [0, (n-1)/n]`` happens with probability
``(b * n/(n-1))^{n-1}``.  Expected payoff

    U(b) = (v - b) * (b * n/(n-1))^{n-1}.

``dU/db = 0`` gives ``m v - b(m+1) = 0`` with ``m = n-1``, i.e.
``b = (n-1) v / n`` -- so the BNE is a mutual best response.

Derivation of ``E[revenue]``.  Revenue is the winner's (highest) bid,
``((n-1)/n) * max_i v_i``.  The maximum of ``n`` iid ``U[0,1]`` draws has mean
``n/(n+1)``, so ``E[revenue] = ((n-1)/n)(n/(n+1)) = (n-1)/(n+1)``.  (This equals
the second-price auction's expected revenue -- the revenue-equivalence theorem.)

:func:`econgym.solvers.closed_form.first_price_bne` is the single source of truth
for these formulas; ``FirstPriceAuctionEnv.equilibrium()`` delegates to it so the
env and its benchmark can never disagree. It is re-exported here (and at the
package root) as ``first_price_bne`` for convenience.
"""
from __future__ import annotations

import numpy as np

from ..core import Box, EconEnv
from ..solvers.closed_form import first_price_bne


class FirstPriceAuctionEnv(EconEnv):
    """First-price sealed-bid auction with iid ``Uniform[0,1]`` private values.

    Subclasses :class:`~econgym.core.EconEnv`. Simultaneous-move, one-shot.

    Parameters
    ----------
    n : int
        Number of bidders.
    G : int
        Size of the optional discrete bid grid ``linspace(0, 1, G)`` exposed as
        :attr:`bid_grid` for index-based agents. The *native* action is the
        continuous bid (a ``Box(0, 1)``); ``step`` accepts native bid values.

    Attributes
    ----------
    action_space : list[Box]
        Per-bidder native bid space (length ``n``, each ``Box(0.0, 1.0)``).
    observation_space : list[Box]
        Per-bidder own-value space (length ``n``, each ``Box(0.0, 1.0)``).
    bid_grid : np.ndarray
        Optional discretized bid grid ``linspace(0, 1, G)`` for index agents.
    """

    metadata = {"name": "FirstPriceAuction", "simultaneous": True}

    def __init__(self, n: int = 2, G: int = 101) -> None:
        self.n = int(n)
        self.G = int(G)
        self.bid_grid = np.linspace(0.0, 1.0, self.G)
        self._values: np.ndarray | None = None
        # Native action = a bid in [0, 1]; observation = own private value in [0, 1].
        self.action_space = [Box(0.0, 1.0) for _ in range(self.n)]
        self.observation_space = [Box(0.0, 1.0) for _ in range(self.n)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_bids(self, indices) -> np.ndarray:
        """Map discrete bid-grid indices to native bid values (for index agents)."""
        return self.bid_grid[np.asarray(indices, dtype=np.int64)]

    # ------------------------------------------------------------------
    # EconEnv hooks
    # ------------------------------------------------------------------
    def _reset(self) -> list:
        """Draw the ``n`` iid ``U[0,1]`` private values and reveal each to its owner.

        Called by :meth:`EconEnv.reset` with ``self.rng`` already resolved. Draws
        exactly ONE ``self.rng.uniform(0, 1, size=n)`` call. Returns the per-agent
        observation list ``[v_0, ..., v_{n-1}]`` (each bidder sees only its own value).
        """
        self._values = self.rng.uniform(0.0, 1.0, size=self.n)
        return [float(v) for v in self._values]

    def step(self, actions):
        """Run the auction.

        Parameters
        ----------
        actions : sequence of n native bids (floats). ``env.to_bids(indices)`` maps
            discrete grid indices to bids for index-based agents.

        Returns
        -------
        obs : list length n
            The private values (terminal observation; the auction is one-shot).
        rewards : np.ndarray length n
            ``v_i - b_i`` for the winner, ``0`` for everyone else.
        terminated : bool
            Always ``True`` (one-shot auction).
        truncated : bool
            Always ``False``.
        info : dict
            ``{"values", "bids", "winner", "price"/"revenue" (winner's paid bid),
            "highest_bid"}``.
        """
        if self._values is None:
            raise RuntimeError("step() called before reset()")
        bids = np.asarray(actions, dtype=float)
        if bids.shape != (self.n,):
            raise ValueError(f"expected {self.n} bids, got shape {bids.shape}")
        values = self._values

        top = bids.max()
        tied = np.flatnonzero(bids == top)
        winner = int(tied[0]) if tied.size == 1 else int(self.rng.choice(tied))

        rewards = np.zeros(self.n, dtype=float)
        rewards[winner] = values[winner] - bids[winner]      # first-price: pay own bid
        revenue = float(bids[winner])                        # == top

        obs = [float(v) for v in values]
        info = {
            "values": values.copy(),
            "bids": bids,
            "winner": winner,
            "price": revenue,          # what the winner pays == seller revenue
            "revenue": revenue,
            "highest_bid": float(top),
        }
        return obs, rewards, True, False, info

    # ------------------------------------------------------------------
    # Closed-form benchmark
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM symmetric Bayes-Nash benchmark for this auction.

        Delegates to :func:`first_price_bne` (the single source of truth) so the
        env and its benchmark can never disagree.  Returns
        ``{"bid_fn", "bid_slope", "expected_revenue", "note"}`` where
        ``bid_fn(v) = ((n-1)/n) v`` and ``expected_revenue = (n-1)/(n+1)``.
        """
        eq = first_price_bne(self.n)
        eq["note"] = (
            "iid U[0,1] first-price auction: symmetric BNE b(v)=((n-1)/n) v; "
            "E[revenue]=(n-1)/(n+1) (equals second-price by revenue equivalence)."
        )
        return eq
