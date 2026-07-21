"""Stateless mean-based (bandit) benchmark learner.

Faithful to Paper-1's ``run_meanbased`` + ``_randargmax``: the agent tracks a
running mean payoff per price and plays epsilon-greedy over it. Exploration is
matched to the Q-learner (same annealed schedule). Exploitation uses a
**randomized** tie-break (``_randargmax``): when the greedy max is tied it draws
one ``rng.integers(n_ties)``. At ``t = 0`` the value table is all-zero, so every
step-0 exploit is a full tie and consumes one draw -- this must be preserved.

The observation is IGNORED (the learner is genuinely stateless).
"""
from __future__ import annotations

import numpy as np

from .base import Agent


def _randargmax(row, rng):
    """Argmax with a uniform random tie-break (verbatim from the original).

    Draws ``rng.integers(len(cands))`` ONLY when the max is tied (>1 candidate).
    """
    m = row.max()
    cands = np.flatnonzero(row == m)
    return cands[rng.integers(len(cands))] if cands.size > 1 else cands[0]


class MeanBased(Agent):
    def __init__(self, env, epsilon: float = 0.10, eps_decay: float | None = 3e-4):
        self.K = int(env.K)
        self.epsilon = float(epsilon)
        self.eps_decay = eps_decay
        self.sums = None
        self.counts = None
        self.val = None
        self._pol_snap = None

    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        # Consumes NO rng (the single init offset lives in env.reset -- an
        # intended, documented aggregate-level deviation; see README).
        # ``track_conv`` is accepted for a uniform Agent.reset signature but is
        # unused: the running-mean value table IS the learning state and must be
        # tracked every step regardless of whether diagnostics are requested.
        self.sums = np.zeros(self.K)
        self.counts = np.zeros(self.K)
        self.val = np.zeros(self.K)
        self._pol_snap = None

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        if self.eps_decay is None:
            e = self.epsilon
        else:
            e = self.epsilon / (1.0 + self.eps_decay * t)
        # epsilon-test rng.random() is drawn EVERY step, before the branch.
        if rng.random() < e:
            return int(rng.integers(self.K))            # explore
        return int(_randargmax(self.val, rng))          # exploit: random tie-break

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        self.counts[action] += 1
        self.sums[action] += reward
        self.val[action] = self.sums[action] / self.counts[action]

    # -- convergence hooks (scalar greedy-arm comparison, matches original) --
    def snapshot_policy(self) -> None:
        self._pol_snap = int(np.argmax(self.val))

    def policy_stability(self) -> float:
        if self._pol_snap is None:
            return 1.0
        return float(self._pol_snap == int(np.argmax(self.val)))

    # -- visitation diagnostics --
    def cells_visited(self):
        return int((self.counts > 0).sum())

    def total_cells(self):
        return int(self.K)

    def min_visit(self):
        nz = self.counts[self.counts > 0]
        return int(nz.min()) if nz.size else 0
