"""Fictitious-play / best-response learning agent.

Fictitious play (Brown 1951; Robinson 1951) is the canonical belief-based
learning rule of game theory. Each period a player forms a belief about every
opponent equal to the **empirical frequency of that opponent's past actions**,
then plays a myopic **best response** to that belief. Its central guarantee:

  * Fictitious play converges (the empirical play, hence the realised actions,
    settle at a Nash equilibrium) in every game with the *fictitious-play
    property* -- **potential games** (Monderer & Shapley 1996), **two-player
    zero-sum games** (Robinson 1951), **2xN games**, and **dominance-solvable
    games**.

Every environment in this suite that FP is applicable to falls in that class:

  * ``cournot``       -- an (ordinal/exact) potential game  -> FP -> Cournot Nash q*.
  * ``bertrand_diff`` -- a potential game (linear-quadratic) -> FP -> Nash price p*.
  * ``public_goods``  -- dominant strategy (free-ride)       -> FP -> c = 0.
  * ``repeated_pd``   -- the STAGE game is dominance-solvable -> FP -> (D, D).

This agent implements FP **through the shared EconEnv interface** -- it is not
hard-wired to any one environment. It discovers, from the ``env`` it is handed:

  * its own discrete action grid (``qgrid`` / ``pgrid`` / ``cgrid`` for the
    continuous envs, whose native action is a :class:`~econgym.core.Box`; or the
    identity grid ``{0, ..., K-1}`` for a :class:`~econgym.core.Discrete` action);
  * a payoff oracle -- the env's pure per-agent stage-payoff function
    (``profit`` / ``payoff``) for the continuous (aggregative) games, or its
    ``best_response`` map when exposed, or the ``stage_payoff`` bimatrix for the
    discrete matrix games.

It obeys the v0 :class:`~econgym.agents.base.Agent` contract verbatim: ``act``
returns an **action index** into that grid, ``update`` folds a transition into
the belief, ``reset`` (re)initialises the belief and consumes **no** ``rng``. It
therefore slots into ``run_episode`` unchanged; for the continuous envs (whose
``step`` consumes *native* actions, not indices) map the returned index through
:attr:`grid` / :meth:`native` -- exactly what an index-based agent does on a
continuous env.

Belief representations
----------------------
* **continuous / aggregative games** (Cournot, differentiated Bertrand, public
  goods): the opponents enter every payoff only through an aggregate that is
  *affine* in their actions, so the belief's **mean** opponent profile is a
  sufficient statistic -- ``E_belief[pi_i(a, .)] = pi_i(a, mean opponents)`` for
  each own action ``a``. The agent tracks the running mean of the observed
  opponent-action vector and best-responds to it (via the env's ``best_response``
  map when available, else by ``argmax`` of the payoff oracle over its grid).
  This is the standard continuous/stochastic fictitious play whose ``1/t``
  averaging supplies the damping that makes it converge even where the *undamped*
  simultaneous best-response map is not a contraction (Cournot with ``n >= 3``).
* **discrete matrix games** (repeated PD stage game): the agent keeps the full
  empirical frequency vector over each opponent's actions and best-responds to
  the mixed belief exactly, ``argmax_a sum_o count[o] * payoff(a, o)``.

``belief="mean"`` (default) is fictitious play (best response to the whole
history's empirical distribution). ``belief="last"`` degrades it to the myopic
single-step **best response** to the opponents' most recent action (Cournot
adjustment / Cournot best-reply dynamics) -- exposed as :class:`BestResponse`.
"""
from __future__ import annotations

import numpy as np

from ..core import Discrete
from .base import Agent

# Attribute names the continuous envs use for their discretised action grid, in
# discovery order (Cournot quantity, differentiated-Bertrand price, public-goods
# contribution, then a generic ``grid``).
_GRID_ATTRS = ("qgrid", "pgrid", "cgrid", "grid")
# Pure per-agent stage-payoff methods, in discovery order.
_PAYOFF_METHODS = ("profit", "payoff", "payoffs")


class FictitiousPlay(Agent):
    """Belief-based fictitious-play best-responder over a discrete action grid.

    Parameters
    ----------
    env : EconEnv
        The environment this agent plays in. Read (never mutated) to discover the
        action grid, the payoff oracle and, when present, the ``best_response``
        map. ``env.n`` fixes the number of players.
    player_id : int
        This agent's index ``i`` in ``0..n-1`` (which payoff row it owns and where
        its own action sits in a joint profile). In a symmetric game create one
        instance per seat: ``[FictitiousPlay(env, i) for i in range(env.n)]``.
    belief : {"mean", "last"}
        ``"mean"`` (default) = fictitious play (best response to the empirical
        distribution of ALL past opponent actions). ``"last"`` = myopic best
        response to the opponents' most recent action (see :class:`BestResponse`).
    grid : array-like, optional
        Explicit discrete action grid (native action values). Overrides
        auto-discovery; required only if the env exposes no recognised grid
        attribute and the action space is continuous.
    payoff : callable, optional
        Explicit payoff oracle. For a continuous env: ``payoff(profile) ->
        per-agent reward vector`` (same signature as ``env.profit``). For a
        discrete env: ``payoff(own_idx, opp_idx) -> own reward`` (same signature
        as ``env.stage_payoff``). Overrides auto-discovery.

    Attributes
    ----------
    grid : np.ndarray
        The native action values, one per action index. ``native(idx) ==
        grid[idx]``. For a discrete action space this is the identity
        ``[0, 1, ..., K-1]`` (index == native action).
    K : int
        Number of discrete actions.
    mode : {"continuous", "discrete"}
        Which belief/best-response machinery is in use (set from the action
        space type at construction).
    """

    def __init__(self, env, player_id: int = 0, belief: str = "mean",
                 grid=None, payoff=None) -> None:
        if belief not in ("mean", "last"):
            raise ValueError(f"belief must be 'mean' or 'last', got {belief!r}")
        self.env = env
        self.i = int(player_id)
        self.n = int(env.n)
        self.belief_mode = belief
        if not (0 <= self.i < self.n):
            raise ValueError(
                f"player_id must be in [0, {self.n}), got {self.i}"
            )

        space = env.action_space[self.i]
        if isinstance(space, Discrete):
            # ---- discrete matrix game (e.g. repeated-PD stage game) ----
            self.mode = "discrete"
            self.K = int(space.n)
            self.grid = np.arange(self.K)          # identity: native == index
            self._payoff_matrix = self._discover_discrete_payoff(env, payoff)
        else:
            # ---- continuous / aggregative game (Cournot, Bertrand-diff, VCM) ----
            self.mode = "continuous"
            g = grid
            if g is None:
                for attr in _GRID_ATTRS:
                    cand = getattr(env, attr, None)
                    if cand is not None:
                        cand = np.asarray(cand, dtype=float).reshape(-1)
                        if cand.size >= 2:
                            g = cand
                            break
            if g is None:
                raise ValueError(
                    "FictitiousPlay: no discrete action grid found on env "
                    f"(looked for {_GRID_ATTRS}); pass grid=... explicitly."
                )
            self.grid = np.asarray(g, dtype=float).reshape(-1)
            self.K = int(self.grid.size)

            # best-response map (exact & cheap) if the env exposes one ...
            br = getattr(env, "best_response", None)
            self._br = br if callable(br) and payoff is None else None
            # ... else fall back to argmax of a pure payoff oracle over the grid.
            pf = payoff
            if pf is None:
                for m in _PAYOFF_METHODS:
                    cand = getattr(env, m, None)
                    if callable(cand):
                        pf = cand
                        break
            if pf is None and self._br is None:
                raise ValueError(
                    "FictitiousPlay: env exposes neither a best_response map nor a "
                    f"payoff oracle {_PAYOFF_METHODS}; pass payoff=... explicitly."
                )
            self._payoff_vec = pf

        self._init_belief()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _discover_discrete_payoff(self, env, payoff):
        """Build the ``K x K`` own-payoff matrix ``A[a, o] = payoff(a, o)``.

        Uses an explicit ``payoff(own, opp)`` callable, else the env's
        ``stage_payoff``. The discrete branch handles two-player matrix games (the
        only discrete env here is the 2-player PD); a discrete game with ``n > 2``
        would need an ``n``-way payoff tensor and is rejected with a clear error.
        """
        if self.n != 2:
            raise NotImplementedError(
                "FictitiousPlay discrete branch supports 2-player matrix games "
                f"(got n={self.n}); pass a payoff tensor via a continuous grid, "
                "or extend _discover_discrete_payoff for n > 2."
            )
        fn = payoff
        if fn is None:
            fn = getattr(env, "stage_payoff", None)
        if not callable(fn):
            raise ValueError(
                "FictitiousPlay: discrete env exposes no stage_payoff(own, opp); "
                "pass payoff=... explicitly."
            )
        A = np.empty((self.K, self.K), dtype=float)
        for a in range(self.K):
            for o in range(self.K):
                A[a, o] = float(fn(a, o))
        return A

    def _init_belief(self) -> None:
        """Zero the belief accumulators (called by ``__init__`` and ``reset``)."""
        if self.mode == "continuous":
            self._opp_sum = np.zeros(self.n - 1, dtype=float)   # sum of opp actions
            self._opp_last = np.zeros(self.n - 1, dtype=float)  # most recent opp actions
        else:
            self._opp_counts = np.zeros(self.K, dtype=float)    # opp action histogram
            self._opp_last = 0
        self._count = 0
        self._greedy = None

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------
    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        """Reinitialise the belief. Consumes NO ``rng`` (like ``MeanBased``); the
        best response is deterministic, so the shared stream is untouched."""
        self._init_belief()

    def _fold(self, obs) -> None:
        """Fold one memory-1 observation of the opponents' last actions into the
        belief. ``obs`` is the opponents' native-action vector (continuous envs)
        or the encoded previous joint profile ``2*own + opp`` (discrete envs)."""
        if self.mode == "continuous":
            o = np.asarray(obs, dtype=float).reshape(-1)
            if o.size != self.n - 1:
                raise ValueError(
                    f"expected an opponent vector of length {self.n - 1}, got {o.size}"
                )
            self._opp_sum += o
            self._opp_last = o
        else:
            # memory-1 discrete encoding: opponent action is the low base-K digit.
            opp = int(obs) % self.K
            self._opp_counts[opp] += 1.0
            self._opp_last = opp
        self._count += 1

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        """Fold the latest observation into the belief and return the best-response
        action **index**. Deterministic (first-max) tie-break -> no ``rng`` draw."""
        self._fold(obs)
        a = self._best_response_index()
        self._greedy = a
        return int(a)

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        """No-op: the belief is the whole learning state and is folded from the
        memory-1 observation inside :meth:`act` (one opponent observation per
        period). Kept for the :class:`Agent` interface."""
        return None

    # ------------------------------------------------------------------
    # best response to the current belief
    # ------------------------------------------------------------------
    def _best_response_index(self) -> int:
        if self.mode == "discrete":
            if self.belief_mode == "mean":
                # best response to the empirical MIXED belief over opponent actions
                belief = self._opp_counts / max(self._count, 1)
                expected = self._payoff_matrix @ belief      # length-K
                return int(np.argmax(expected))
            # best response to the opponent's LAST action
            return int(np.argmax(self._payoff_matrix[:, self._opp_last]))

        # ---- continuous / aggregative ----
        if self.belief_mode == "mean":
            opp = self._opp_sum / max(self._count, 1)
        else:
            opp = self._opp_last
        if self._br is not None:
            # exact continuous best response; entry i uses only opponents, so the
            # own slot value is irrelevant -> snap the continuous BR to the grid.
            profile = self._assemble(0.0, opp)
            br_i = float(np.asarray(self._br(profile), dtype=float)[self.i])
            return int(np.argmin(np.abs(self.grid - br_i)))
        # argmax of the payoff oracle over the grid, opponents fixed at the belief.
        best_val = -np.inf
        best_idx = 0
        for k in range(self.K):
            profile = self._assemble(self.grid[k], opp)
            val = float(np.asarray(self._payoff_vec(profile), dtype=float)[self.i])
            if val > best_val:
                best_val = val
                best_idx = k
        return best_idx

    def _assemble(self, own_value, opp_vector) -> np.ndarray:
        """Build a length-``n`` joint native profile: own seat = ``own_value``, the
        other seats = ``opp_vector`` in ascending player order (skipping self) --
        the same ordering the continuous envs use to build their observations."""
        profile = np.empty(self.n, dtype=float)
        profile[self.i] = own_value
        k = 0
        opp_vector = np.asarray(opp_vector, dtype=float).reshape(-1)
        for j in range(self.n):
            if j != self.i:
                profile[j] = opp_vector[k]
                k += 1
        return profile

    # ------------------------------------------------------------------
    # helpers / introspection
    # ------------------------------------------------------------------
    def native(self, index):
        """Map an action index to its native action value (``grid[index]``)."""
        return self.grid[int(index)]

    def belief_mean(self):
        """Current belief about the opponents.

        Continuous mode: the running mean opponent-action vector (length ``n-1``).
        Discrete mode: the empirical opponent-action probability vector (length
        ``K``). ``None`` before the first observation.
        """
        if self._count == 0:
            return None
        if self.mode == "continuous":
            return self._opp_sum / self._count
        return self._opp_counts / self._count

    def greedy_action(self):
        """Best-response index to the CURRENT belief without folding a new
        observation (a pure read of the learned policy)."""
        if self._count == 0:
            return None
        return int(self._best_response_index())

    # -- convergence hooks (Agent interface) --
    def snapshot_policy(self) -> None:
        self._pol_snap = None if self._count == 0 else int(self._best_response_index())

    def policy_stability(self) -> float:
        snap = getattr(self, "_pol_snap", None)
        if snap is None or self._count == 0:
            return 1.0
        return float(snap == int(self._best_response_index()))


class BestResponse(FictitiousPlay):
    """Myopic best-response (Cournot best-reply) dynamics: fictitious play with a
    point belief on the opponents' **most recent** action rather than their whole
    empirical history. Equivalent to ``FictitiousPlay(env, i, belief="last")``.

    Without the ``1/t`` averaging of fictitious play this is only convergent where
    the *undamped* simultaneous best-response map is itself a contraction (e.g.
    Cournot ``n = 2``); for Cournot ``n >= 3`` it cycles/diverges, exactly as
    documented for the raw best-response map. Use :class:`FictitiousPlay` (the
    averaged rule) for guaranteed convergence in potential games.
    """

    def __init__(self, env, player_id: int = 0, grid=None, payoff=None) -> None:
        super().__init__(env, player_id=player_id, belief="last",
                         grid=grid, payoff=payoff)


__all__ = ["FictitiousPlay", "BestResponse"]
