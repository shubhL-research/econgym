"""Memory-1, opponent-price-state, epsilon-greedy Q-learner.

Byte-identical (RNG and update) to Paper-1's ``run_qlearning``:

    Q(s, a) <- Q(s, a) + alpha * [ pi + gamma * max_a' Q(s', a') - Q(s, a) ]

with state ``s`` = the opponent's previous-period price index. Exploration is
epsilon-greedy with an annealed schedule ``eps_t = eps / (1 + eps_decay * t)``;
exploitation is a deterministic first-max ``np.argmax`` -- **no** random
tie-break, so exploit consumes no rng.
"""
from __future__ import annotations

import numpy as np

from .base import Agent


class QLearner(Agent):
    def __init__(self, env, alpha: float = 0.10, gamma: float = 0.95,
                 epsilon: float = 0.10, eps_decay: float | None = 3e-4,
                 q_init_std: float = 1e-6):
        """``K`` and the state-space size are read from ``env``.

        ``eps_decay=None`` -> constant epsilon. For ``n == 2`` the state space is
        ``K`` (single opponent price), so ``Q`` has shape ``(K, K)`` -- matching
        the original exactly. For ``n > 2`` the joint opponent profile is
        base-``K`` encoded into ``K**(n-1)`` states (a faithful generalisation).
        """
        self.K = int(env.K)
        self.n = int(env.n)
        self.n_states = self.K ** (self.n - 1)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.eps_decay = eps_decay
        self.q_init_std = float(q_init_std)
        self.Q = None
        self.visits = None
        self._track = False
        self._pol_snap = None

    # ------------------------------------------------------------------
    def _encode(self, obs) -> int:
        """Opponent price index(es) -> flat state index.

        For ``n == 2`` the env hands us a plain int (the single opponent's price
        index) which is returned unchanged. For ``n > 2`` we base-``K`` encode
        the tuple of opponents' indices.
        """
        if isinstance(obs, (int, np.integer)):
            return int(obs)
        s = 0
        mult = 1
        for v in obs:
            s += int(v) * mult
            mult *= self.K
        return s

    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        # ONE draw for THIS agent (the runner calls reset in agent order, so
        # agent 0's Q is drawn before agent 1's) -- byte-matches the original
        # ``Q = [rng.normal(0.0, 1e-6, size=(K, K)) for _ in range(2)]``.
        self.Q = rng.normal(0.0, self.q_init_std, size=(self.n_states, self.K))
        # The (n_states x K) visit counter is a DIAGNOSTIC-ONLY structure (read
        # by the convergence / visitation hooks). Allocate + increment it only
        # when diagnostics are requested -- matching the original's conditional
        # tracking and avoiding needless work at large n/K. It consumes no rng,
        # so gating it never perturbs the byte-exact stream.
        self._track = bool(track_conv)
        self.visits = (np.zeros((self.n_states, self.K), dtype=np.int64)
                       if self._track else None)
        self._pol_snap = None

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        s = self._encode(obs)
        if self.eps_decay is None:
            e = self.epsilon
        else:
            e = self.epsilon / (1.0 + self.eps_decay * t)
        # epsilon-test rng.random() is drawn EVERY step, before the branch.
        if rng.random() < e:
            a = int(rng.integers(self.K))          # explore
        else:
            a = int(np.argmax(self.Q[s]))          # exploit: first-max, NO rng
        if self._track:                            # diagnostics-only bookkeeping
            self.visits[s, a] += 1
        return a

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        s = self._encode(obs)
        ns = self._encode(next_obs)
        self.Q[s, action] += self.alpha * (
            reward + self.gamma * self.Q[ns].max() - self.Q[s, action]
        )

    # -- convergence hooks --
    def snapshot_policy(self) -> None:
        self._pol_snap = self.Q.argmax(axis=1).copy()

    def policy_stability(self) -> float:
        if self._pol_snap is None or self.visits is None:
            return 1.0
        seen = self.visits.sum(axis=1) > 0
        final_pol = self.Q.argmax(axis=1)
        denom = int(seen.sum())
        if not denom:
            return 1.0
        return float((self._pol_snap[seen] == final_pol[seen]).mean())

    # -- visitation diagnostics (only populated when track_conv=True) --
    def cells_visited(self):
        return None if self.visits is None else int((self.visits > 0).sum())

    def total_cells(self):
        return int(self.n_states * self.K)

    def min_visit(self):
        if self.visits is None:
            return None
        nz = self.visits[self.visits > 0]
        return int(nz.min()) if nz.size else 0
