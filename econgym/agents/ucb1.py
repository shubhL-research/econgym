"""UCB1 -- deterministic upper-confidence-bound bandit over a discrete action set.

The classic Auer, Cesa-Bianchi & Fischer (2002) index policy
("Finite-time Analysis of the Multiarmed Bandit Problem", *Machine Learning* 47).
On each step the agent plays

    argmax_a  [ mean_a + sqrt(c * ln(t) / n_a) ]                       (c = 2)

where ``mean_a`` is the running mean reward of arm ``a``, ``n_a`` the number of
times it has been pulled, and ``t = sum_a n_a`` the total pulls so far. Every arm
is pulled once before the index rule engages (the ``sqrt`` term is ``+inf`` for an
unplayed arm). With rewards bounded in ``[0, 1]`` the policy enjoys the KNOWN
gap-dependent regret bound (Theorem 1 of the paper)

    E[R_T]  <=  8 * sum_{i: Delta_i > 0} (ln T) / Delta_i
                +  (1 + pi^2 / 3) * sum_i Delta_i ,     Delta_i = mu* - mu_i ,

i.e. regret is ``O(K * ln T)`` -- **sublinear** in the horizon ``T``. That bound is
the closed-form answer ``tests/test_agent_ucb1.py`` validates against.

Design notes (honouring the shared Agent/EconEnv contract, CONTRACT_v1.md §1.4/§4.1)
-----------------------------------------------------------------------------------
* ``act`` returns an ACTION INDEX into the agent's ``Discrete`` action set. The
  discrete arm count ``K`` is read *through the EconEnv interface* --
  ``env.action_space[agent_id].n`` (with a fallback to ``env.K`` for the
  discretized Bertrand-style envs) -- so the agent is NOT hard-coded to one env.
* UCB1 is deterministic. Unlike the epsilon-greedy learners it does **not** draw
  the shared ``rng`` every step; it consumes ``rng`` *only to break ties* -- among
  the unplayed arms during the initialisation round-robin, and among equal indices
  thereafter (a measure-zero event for continuous reward distributions, but real
  for {0,1}/rational rewards). This preserves the "a draw every step is optional"
  discipline while still routing all randomness through the shared stream.
* ``reset`` consumes NO rng (the running-statistics tables are the learning state).
* The observation is IGNORED: a bandit is stateless w.r.t. the environment state.
"""
from __future__ import annotations

import numpy as np

from .base import Agent


def _num_actions(env, agent_id: int) -> int:
    """Discrete arm count read through the shared EconEnv interface.

    Primary source is the per-agent ``Discrete`` action space
    (``env.action_space[agent_id].n``, CONTRACT_v1.md §1.3); falls back to the
    ``env.K`` attribute exposed by the discretized Bertrand-style envs. Raises if
    neither yields a finite discrete count (UCB1 is a discrete-action policy).
    """
    aspace = getattr(env, "action_space", None)
    if aspace is not None:
        space = aspace[agent_id]
        n = getattr(space, "n", None)          # Discrete exposes .n; Box does not
        if n is not None:
            return int(n)
    if hasattr(env, "K"):
        return int(env.K)
    raise TypeError(
        "UCB1 needs a discrete action set: env.action_space"
        f"[{agent_id}] must be a Discrete space (or env must expose .K); "
        f"got action_space={aspace!r}"
    )


def _randargmax(row, rng):
    """Argmax with a uniform random tie-break drawn from the shared ``rng``.

    Draws ``rng.integers(n_ties)`` ONLY when the maximum is tied (>1 candidate),
    so a strict maximum consumes no randomness (mirrors ``MeanBased._randargmax``).
    """
    m = row.max()
    cands = np.flatnonzero(row == m)
    return int(cands[rng.integers(len(cands))]) if cands.size > 1 else int(cands[0])


class UCB1(Agent):
    """UCB1 index-policy bandit over ``K`` discrete actions.

    Parameters
    ----------
    env : EconEnv
        Any EconGym env exposing a ``Discrete`` action space for ``agent_id``
        (or a ``.K`` attribute). Only the arm count is read from it.
    c : float
        Exploration coefficient inside the confidence radius
        ``sqrt(c * ln t / n_a)``. The classic UCB1 uses ``c = 2`` (the value the
        Auer et al. regret bound is proved for); larger ``c`` explores more.
    agent_id : int
        Index of this agent in the env's per-agent space lists (0 for a
        single-agent bandit). Selects ``env.action_space[agent_id]``.
    """

    def __init__(self, env, c: float = 2.0, agent_id: int = 0):
        self.K = _num_actions(env, agent_id)
        self.c = float(c)
        self.agent_id = int(agent_id)
        self.counts = None       # n_a : pulls per arm
        self.sums = None         # cumulative reward per arm
        self.means = None        # running mean reward per arm
        self._pol_snap = None

    # ------------------------------------------------------------------
    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        """Zero the running statistics. Consumes NO rng (UCB1 draws only for
        tie-breaking inside ``act``); ``track_conv`` is accepted for a uniform
        signature but unused -- the reward statistics ARE the learning state and
        are maintained every step regardless of diagnostics."""
        self.counts = np.zeros(self.K, dtype=np.int64)
        self.sums = np.zeros(self.K, dtype=np.float64)
        self.means = np.zeros(self.K, dtype=np.float64)
        self._pol_snap = None

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        """Return the UCB1 arm index. Observation is ignored (stateless bandit).

        Initialisation: while any arm is unplayed, play one of them (random
        tie-break among the unplayed). Otherwise play
        ``argmax_a mean_a + sqrt(c * ln(total) / n_a)`` (random tie-break on
        exact index ties only)."""
        counts = self.counts
        unplayed = counts == 0
        if unplayed.any():
            # Round-robin init: prefer unplayed arms; random tie-break among them.
            return _randargmax(np.where(unplayed, 0.0, -np.inf), rng)
        total = float(counts.sum())            # >= K >= 1 once all arms are played
        bonus = np.sqrt(self.c * np.log(total) / counts)
        return _randargmax(self.means + bonus, rng)

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        """Fold one realised reward into arm ``action``'s running mean. No rng."""
        self.counts[action] += 1
        self.sums[action] += reward
        self.means[action] = self.sums[action] / self.counts[action]

    # -- convergence hooks (scalar greedy-arm comparison, matches MeanBased) --
    @property
    def greedy_arm(self) -> int:
        """The current empirical-best arm ``argmax_a mean_a``."""
        return int(np.argmax(self.means))

    def snapshot_policy(self) -> None:
        self._pol_snap = int(np.argmax(self.means))

    def policy_stability(self) -> float:
        if self._pol_snap is None:
            return 1.0
        return float(self._pol_snap == int(np.argmax(self.means)))

    # -- visitation diagnostics --
    def cells_visited(self):
        return int((self.counts > 0).sum())

    def total_cells(self):
        return int(self.K)

    def min_visit(self):
        nz = self.counts[self.counts > 0]
        return int(nz.min()) if nz.size else 0
