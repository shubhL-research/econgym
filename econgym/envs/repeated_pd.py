"""Infinitely-repeated Prisoner's Dilemma with discounting.

A 2-player simultaneous-move stage Prisoner's Dilemma, repeated indefinitely with
a common per-period discount factor ``delta in (0, 1)``. The env owns only the
*physics* (the stage bimatrix and a memory-1 observation of the previous joint
action); it holds no learning state and consumes RNG only in ``reset`` (a single
``rng.integers(0, 2, size=2)`` draw for the initial joint action profile), in
keeping with the shared seeding contract.

Stage bimatrix (own payoff; row = own action C/D, col = opponent action C/D)::

                   opp C     opp D
        own C    (R, R)    (S, T)
        own D    (T, S)    (P, P)

with the PD orderings ``T > R > P > S`` (so ``D`` strictly dominates ``C`` in the
stage game) and efficiency ``2R > T + S`` (mutual cooperation is the efficient
outcome, worth more than alternating exploitation).

Actions are ``0 = Cooperate`` and ``1 = Defect``. Observations are memory-1 and
carry BOTH players' previous actions (encoded as a single ``Discrete(4)`` index
``2 * own_prev + opp_prev``), so grim-trigger / tit-for-tat agents are
representable. ``step`` returns ``terminated == truncated == False`` (the game is
infinite-horizon; the runner's ``T`` and the env's ``delta`` model the
horizon/discounting in downstream analysis).

Closed-form benchmark
---------------------
One-shot Nash is ``(D, D)``. Grim-trigger cooperation is a subgame-perfect
equilibrium of the repeated game iff ``delta >= (T - R) / (T - P)`` -- the
folk-theorem threshold, delegated to
:func:`econgym.solvers.closed_form.repeated_pd_threshold` as the single source of
truth. With the classic defaults ``R=3, T=5, P=1, S=0`` the threshold is
``(5 - 3) / (5 - 1) = 0.5``.
"""
from __future__ import annotations

import numpy as np

from ..core import Discrete, EconEnv
from ..solvers.closed_form import repeated_pd_threshold


class RepeatedPDEnv(EconEnv):
    """2-player infinitely-repeated Prisoner's Dilemma with discounting.

    Parameters
    ----------
    n : int
        Number of players. The stage PD is intrinsically 2-player; ``n`` must be
        ``2`` (kept as an argument only for a uniform env constructor surface).
    R, T, P, S : float
        Stage payoffs -- mutual-cooperate Reward, Temptation (defect vs a
        cooperator), mutual-defect Punishment, Sucker (cooperate vs a defector).
        Must satisfy the PD orderings ``T > R > P > S`` and efficiency
        ``2 * R > T + S``; the constructor rejects violations.
    delta : float
        Common per-period discount factor, in the open interval ``(0, 1)``.

    Attributes
    ----------
    action_space : list[Discrete]
        Per-player action space, length ``n`` (each ``Discrete(2)``: 0=C, 1=D).
    observation_space : list[Discrete]
        Per-player memory-1 observation space, length ``n`` (each ``Discrete(4)``:
        the previous joint action profile encoded as ``2*own_prev + opp_prev``).
    """

    metadata = {"name": "RepeatedPD", "simultaneous": True}

    # action encoding
    COOPERATE = 0
    DEFECT = 1
    _ACTION_LABEL = {0: "C", 1: "D"}

    def __init__(self, n: int = 2, R: float = 3.0, T: float = 5.0,
                 P: float = 1.0, S: float = 0.0, delta: float = 0.8) -> None:
        self.n = int(n)
        if self.n != 2:
            raise ValueError(
                f"RepeatedPDEnv is a 2-player stage game; got n={self.n}."
            )
        self.R = float(R)
        self.T = float(T)
        self.P = float(P)
        self.S = float(S)
        self.delta = float(delta)

        # --- validate the Prisoner's-Dilemma structure (physics faithfulness) ---
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(
                "PD payoffs must satisfy T > R > P > S, got "
                f"T={self.T}, R={self.R}, P={self.P}, S={self.S}."
            )
        if not (2.0 * self.R > self.T + self.S):
            raise ValueError(
                "PD efficiency requires 2*R > T + S (so mutual cooperation beats "
                f"alternating exploitation), got 2*{self.R}={2.0 * self.R} "
                f"<= T+S={self.T + self.S}."
            )
        if not (0.0 < self.delta < 1.0):
            raise ValueError(
                f"discount factor delta must lie in (0, 1), got delta={self.delta}."
            )

        # Stage payoff matrix indexed [own_action][opp_action] (0=C, 1=D):
        #   [C,C]=R  [C,D]=S
        #   [D,C]=T  [D,D]=P
        self._payoff = np.array([[self.R, self.S],
                                 [self.T, self.P]], dtype=np.float64)

        self._prev = None
        self.action_space = [Discrete(2) for _ in range(self.n)]
        # memory-1 observation: previous joint action profile from each agent's
        # own perspective -> 4 distinct states (2*own_prev + opp_prev).
        self.observation_space = [Discrete(4) for _ in range(self.n)]

    # ------------------------------------------------------------------
    # Observation encoding (memory-1, both players' previous actions)
    # ------------------------------------------------------------------
    def _obs(self, idx):
        """Per-agent observation from a joint action-index profile ``idx``.

        Agent ``i`` observes ``2 * own_prev + opp_prev`` in ``{0, 1, 2, 3}`` (own
        action high bit, opponent low bit). This exposes BOTH previous actions:
        ``opp_prev = obs % 2`` (tit-for-tat) and ``obs == 0`` iff both cooperated
        last period (grim trigger).
        """
        a0, a1 = int(idx[0]), int(idx[1])
        return [2 * a0 + a1, 2 * a1 + a0]

    def _reset(self) -> list:
        """Draw the initial joint action profile and return per-agent obs.

        Called by :meth:`EconEnv.reset` with ``self.rng`` already resolved. Draws
        exactly ONE ``self.rng.integers(0, 2, size=2)`` call (the initial joint
        action profile), stores it, and returns ``[obs_0, obs_1]``.
        """
        prev = self.rng.integers(0, 2, size=2)
        self._prev = prev
        return self._obs(prev)

    def step(self, actions):
        """Play one stage of the PD.

        Parameters
        ----------
        actions : sequence of 2 action INDICES (0=Cooperate, 1=Defect).

        Returns
        -------
        obs : list length 2
            Memory-1 observation of the just-played joint profile, per agent.
        rewards : np.ndarray length 2
            ``[payoff(a0 | a1), payoff(a1 | a0)]`` from the stage bimatrix.
        terminated : bool
            Always ``False`` (infinite-horizon repeated game).
        truncated : bool
            Always ``False`` (the runner's ``T`` bounds the horizon externally).
        info : dict
            ``{"actions": np.ndarray of the two chosen action indices}``.
        """
        a0 = int(actions[0])
        a1 = int(actions[1])
        idx = np.array([a0, a1], dtype=np.int64)
        rewards = np.array([self._payoff[a0, a1], self._payoff[a1, a0]],
                           dtype=np.float64)
        self._prev = idx
        return self._obs(idx), rewards, False, False, {"actions": idx}

    # ------------------------------------------------------------------
    # Stage-game & repeated-game analytic helpers
    # ------------------------------------------------------------------
    def stage_payoff(self, own_action: int, opp_action: int) -> float:
        """Own stage payoff for playing ``own_action`` against ``opp_action``
        (0=Cooperate, 1=Defect)."""
        return float(self._payoff[int(own_action), int(opp_action)])

    def cooperation_value(self, delta: float | None = None) -> float:
        """Discounted value of the all-cooperate path: ``R / (1 - delta)``."""
        d = self.delta if delta is None else float(delta)
        return self.R / (1.0 - d)

    def deviation_value(self, delta: float | None = None) -> float:
        """Discounted value of a one-shot defection then grim punishment forever:
        ``T + delta * P / (1 - delta)``."""
        d = self.delta if delta is None else float(delta)
        return self.T + d * self.P / (1.0 - d)

    # ------------------------------------------------------------------
    # Closed-form benchmark (delegates to solvers.closed_form)
    # ------------------------------------------------------------------
    def equilibrium(self) -> dict:
        """CLOSED-FORM benchmark for this repeated PD.

        Delegates the folk-theorem grim-trigger threshold to
        :func:`econgym.solvers.closed_form.repeated_pd_threshold` (the single
        source of truth), so the env and solver can never disagree. Returns:

          * ``one_shot_nash``      -- ``("D", "D")`` (unique stage-game Nash).
          * ``grim_threshold``     -- ``delta* = (T - R) / (T - P)``.
          * ``coop_sustainable_at``-- ``{"delta": delta, "sustainable": bool}`` for
            THIS env's ``delta`` (grim cooperation is a SPE iff ``delta >= delta*``).
          * ``stage_payoffs``      -- ``{"R", "T", "P", "S"}``.
          * ``delta``              -- this env's discount factor.
        """
        cf = repeated_pd_threshold(self.T, self.R, self.P, self.S)
        threshold = cf["grim_threshold"]
        return {
            "one_shot_nash": cf["one_shot_nash"],   # ("D", "D")
            "grim_threshold": threshold,
            "coop_sustainable_at": {
                "delta": self.delta,
                "sustainable": bool(self.delta >= threshold),
            },
            "stage_payoffs": {"R": self.R, "T": self.T, "P": self.P, "S": self.S},
            "delta": self.delta,
        }
