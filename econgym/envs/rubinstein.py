"""Rubinstein alternating-offers bargaining.

Two players split a unit pie via alternating offers with a common per-round
discount ``delta``: player 1 (agent ``0``) proposes in odd rounds, player 2
(agent ``1``) in even rounds. A proposer offers a split; the responder accepts
(payoffs realized, discounted by ``delta**(round-1)``) or rejects (roles swap,
the game continues).

The equilibrium is analytic (unique subgame-perfect equilibrium):

    proposer share  = 1 / (1 + delta)
    responder share = delta / (1 + delta)     (agreement in round 1, no delay).

``equilibrium()`` delegates to :func:`econgym.solvers.closed_form.rubinstein_split`
so the env and the solver can never disagree (single source of truth). The env's
role is to host the *physics* (offer / accept / reject accounting with the
correct discounting) that the deviation and equilibrium checks exercise; it holds
no learning state.

Simultaneous-API rendering of a sequential game
-----------------------------------------------
``EconEnv`` is a simultaneous-move, list-of-actions base. Rubinstein is inherently
sequential and the proposer/responder roles *alternate* each round, so a single
fixed per-agent action type (a ``Box`` offer for one, a ``Discrete`` accept for
the other) cannot be assigned statically. Instead **each agent submits a length-2
action** ``[offer_keep, accept_signal]`` every step, and ``step`` reads only the
component that matters for that agent's current role:

  * the current **proposer**'s ``offer_keep`` (component 0) = the share it
    proposes to KEEP for itself (so the responder is offered ``1 - offer_keep``);
  * the current **responder**'s ``accept_signal`` (component 1) = accept iff
    ``>= 0.5``.

``info`` always carries whose turn it is (``proposer``) and the round. This is the
faithful way to express the alternating game through the simultaneous list-action
contract; the closed-form SPE, the responder indifference, and the
no-profitable-deviation property are all recovered exactly (see
``tests/test_env_rubinstein.py``).
"""
from __future__ import annotations

import numpy as np

from ..core import Box, EconEnv
from ..solvers import closed_form


class RubinsteinEnv(EconEnv):
    """Two-player Rubinstein alternating-offers bargaining over a unit pie.

    Parameters
    ----------
    delta : float
        Common per-round discount factor, ``0 <= delta < 1`` (default ``0.9``).
    max_rounds : int or None
        Optional horizon. If set and the players reach round ``max_rounds`` without
        agreement, the next rejection truncates the episode (``truncated=True``)
        with a zero (disagreement) payoff. ``None`` (default) leaves the game
        open-ended -- the runner's ``T`` then bounds it externally.

    Attributes
    ----------
    n : int
        Always ``2``.
    delta : float
        The per-round discount factor.
    round : int
        Current bargaining round (1-indexed); ``round`` odd => agent 0 proposes,
        ``round`` even => agent 1 proposes.
    action_space : list[Box]
        Length 2; each ``Box([0,0], [1,1])`` = ``[offer_keep, accept_signal]``.
    observation_space : list[Box]
        Length 2; each ``Box([0,1], [1, +inf])`` = ``[proposer_id, round]`` (the
        public state, identical for both agents).
    """

    metadata = {"name": "Rubinstein", "simultaneous": True}

    def __init__(self, delta: float = 0.9, max_rounds: int | None = None) -> None:
        self.n = 2
        self.delta = float(delta)
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(
                f"RubinsteinEnv requires 0 <= delta < 1, got delta={self.delta}."
            )
        if max_rounds is not None:
            max_rounds = int(max_rounds)
            if max_rounds < 1:
                raise ValueError(f"max_rounds must be >= 1, got {max_rounds}.")
        self.max_rounds = max_rounds
        self.round = 1
        self._done = False

        # Each agent submits [offer_keep in [0,1], accept_signal in [0,1]].
        self.action_space = [
            Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
            for _ in range(self.n)
        ]
        # Public observation: [proposer_id in {0,1}, round >= 1].
        round_hi = float(self.max_rounds) if self.max_rounds is not None else np.inf
        self.observation_space = [
            Box(low=np.array([0.0, 1.0]), high=np.array([1.0, round_hi]))
            for _ in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Role / observation helpers
    # ------------------------------------------------------------------
    def _proposer(self) -> int:
        """Current proposer: agent 0 in odd rounds, agent 1 in even rounds."""
        return 0 if (self.round % 2 == 1) else 1

    def _obs(self) -> list:
        """Public state ``[proposer_id, round]`` seen by both agents."""
        pub = np.array([float(self._proposer()), float(self.round)], dtype=np.float64)
        return [pub.copy(), pub.copy()]

    # ------------------------------------------------------------------
    # EconEnv hooks
    # ------------------------------------------------------------------
    def _reset(self) -> list:
        """Reset to round 1 (agent 0 proposes). Deterministic: the bargaining
        game has no stochastic initial state, so ``self.rng`` (already set by
        :meth:`EconEnv.reset`) is intentionally not consumed."""
        self.round = 1
        self._done = False
        return self._obs()

    def step(self, actions):
        """Advance one bargaining round.

        Parameters
        ----------
        actions : sequence of 2 array-likes
            ``actions[i] = [offer_keep, accept_signal]`` for agent ``i``. Only the
            current proposer's ``offer_keep`` and the current responder's
            ``accept_signal`` are consulted (component 0 and component 1
            respectively); the unused components are ignored.

        Returns
        -------
        obs : list length 2
            Public state ``[proposer_id, round]`` for each agent (post-transition).
        rewards : np.ndarray length 2
            Discounted realized payoffs this step: zero except on the accepting
            round, where proposer gets ``delta**(round-1)*offer_keep`` and
            responder ``delta**(round-1)*(1-offer_keep)``.
        terminated : bool
            ``True`` on acceptance (a deal is struck).
        truncated : bool
            ``True`` only if ``max_rounds`` is set and reached without agreement.
        info : dict
            ``{"proposer", "responder", "round", "offer_keep",
            "responder_offered", "accepted", "discount"}``.
        """
        if self._done:
            raise RuntimeError("step() called after termination; call reset() first.")

        p = self._proposer()
        r = 1 - p
        cur_round = self.round

        a_p = np.asarray(actions[p], dtype=np.float64).reshape(-1)
        a_r = np.asarray(actions[r], dtype=np.float64).reshape(-1)
        offer_keep = float(np.clip(a_p[0], 0.0, 1.0))       # proposer's own share
        responder_offered = 1.0 - offer_keep
        accept = bool(a_r[-1] >= 0.5)                        # responder's accept bit

        rewards = np.zeros(self.n, dtype=np.float64)
        discount = self.delta ** (cur_round - 1)

        if accept:
            rewards[p] = discount * offer_keep
            rewards[r] = discount * responder_offered
            self._done = True
            terminated, truncated = True, False
        else:
            # Rejection: roles swap, next round begins.
            self.round = cur_round + 1
            terminated = False
            truncated = bool(
                self.max_rounds is not None and self.round > self.max_rounds
            )
            if truncated:
                self._done = True  # disagreement: both keep zero

        info = {
            "proposer": p,
            "responder": r,
            "round": cur_round,
            "offer_keep": offer_keep,
            "responder_offered": responder_offered,
            "accepted": accept,
            "discount": discount,
        }
        return self._obs(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Closed-form benchmark (single source of truth: the solver)
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """Unique subgame-perfect equilibrium, delegated to
        :func:`econgym.solvers.closed_form.rubinstein_split` so the env and the
        solver can never disagree.

        Returns ``{"proposer_share": 1/(1+delta),
        "responder_share": delta/(1+delta), "agreement_round": 1}``.
        """
        return closed_form.rubinstein_split(self.delta)
