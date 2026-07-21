"""Agent abstract base class.

An agent is a pure learning rule: it maps an observation to a price index
(``act``) and updates its internal state from a transition (``update``). All
randomness flows through the shared episode ``rng`` supplied by the runner; the
agent never owns a Generator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        """Return a price INDEX.

        ``t`` is the current step (drives the epsilon schedule). ``rng`` is the
        SHARED episode Generator -- draws here define the reproducible stream.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, obs, action: int, reward: float, next_obs) -> None:
        """Learning update for one transition. Consumes NO rng. A stateless
        agent may still update running statistics here (or in ``act``)."""
        raise NotImplementedError

    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        """Initialise internal state / tables. Called ONCE per episode, in agent
        order, BEFORE ``env.reset``. May consume ``rng`` (QLearner does;
        MeanBased does not).

        ``track_conv`` signals whether the runner will request convergence /
        visitation diagnostics for this episode. Agents may use it to skip
        allocating diagnostic-only state (e.g. QLearner's per-cell visit
        counter) when it will never be read; it MUST NOT change the rng stream.
        """
        return None

    # -- optional convergence hooks (used by the runner when track_conv=True) --
    def snapshot_policy(self) -> None:
        """Record the greedy policy at ``t = floor(0.9 * T)``."""
        return None

    def policy_stability(self) -> float:
        """Fraction of visited states whose greedy action is unchanged vs the
        snapshot (1.0 if no snapshot / no visited states)."""
        return 1.0

    # -- optional visitation diagnostics --
    def cells_visited(self):
        return None

    def total_cells(self):
        return None

    def min_visit(self):
        return None
