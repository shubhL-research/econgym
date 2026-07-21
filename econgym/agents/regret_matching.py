"""Hart--Mas-Colell regret matching (external-regret / Hannan-consistent variant).

This implements *regret matching* as described in Hart & Mas-Colell,
"A Simple Adaptive Procedure Leading to Correlated Equilibrium" (Econometrica,
2000) and its Blackwell-approachability analysis. The agent maintains a vector of
cumulative (external) regrets -- one per own action -- and, each period, plays a
mixed strategy **proportional to the positive part of the regret vector**
(uniform when no regret is positive):

    R_t(k) = sum_{s <= t} [ u(k, a^{-i}_s) - u(a^i_s, a^{-i}_s) ]      (cumulative regret)

    p_{t+1}(k) = R_t^+(k) / sum_j R_t^+(j)      if sum_j R_t^+(j) > 0
               = 1 / K                          otherwise                  (uniform)

**Full information is intrinsic to the procedure.** Regret matching needs, each
period, the counterfactual payoff ``u(k, a^{-i})`` of *every* own action ``k``
against the opponent's realized action -- not only the payoff of the action it
actually played. This is what makes it a *no-external-regret* rule; the
partial-information (bandit) setting requires a different algorithm (e.g. Exp3)
and does not enjoy the same guarantee. The agent therefore resolves an exact
payoff oracle (see below) and raises if none is available, rather than silently
degrading to a heuristic without the theoretical backing.

**Theory this validates against.**

* *Hannan consistency (regret bound).* Regret matching is a Blackwell-approachability
  strategy for the negative orthant, so the cumulative positive regret grows
  sub-linearly: ``sum_k (R_T^+(k))^2 <= K * M^2 * T`` with ``M`` the per-period
  payoff range, hence the **average** regret is bounded by a closed form that
  vanishes:

      max_k R_T^+(k) / T  <=  M * sqrt(K / T)   -> 0.

* *Convergence to (coarse) correlated equilibrium.* When every player uses a
  Hannan-consistent rule, the **empirical distribution of play** converges to the
  set of coarse correlated equilibria (CCE) of the stage game. (The
  internal-regret refinement of the same procedure converges to the smaller set of
  correlated equilibria; for games solvable by strict dominance the two sets
  coincide with the Nash set, as in the Prisoner's Dilemma where CCE = {(D, D)}.)

**How it works through the shared EconEnv interface (not hardcoded to one env).**

The agent is a general 2-player, discrete-action *matrix-game* learner. It reads
everything it needs from the environment / API rather than assuming a specific
game:

* the number of own actions ``K`` from ``env.action_space[player_id].n``;
* the number of opponent actions ``K_opp`` from ``env.action_space[opp_id].n``;
* the per-period counterfactual payoffs ``u(k, a^{-i})`` for every own action
  ``k`` from a payoff oracle, resolved (in priority order) from an explicitly
  supplied ``payoff`` matrix/callable, else the env's ``stage_payoff(own, opp)``
  hook (exposed by the symmetric EconGym matrix envs, e.g. ``RepeatedPDEnv``);
* the opponent's realized action, decoded from the memory-1 observation using the
  EconGym joint-action encoding ``obs = own * K_opp + opp`` (so
  ``opp = next_obs % K_opp``), overridable via ``decode_opp``.

It draws from the shared episode ``rng`` only in ``act`` (one ``rng.random()`` per
step for the inverse-CDF sample), never owns a Generator, and consumes no ``rng``
in ``reset`` (the procedure starts from the uniform strategy).
"""
from __future__ import annotations

import numpy as np

from .base import Agent


class RegretMatching(Agent):
    """Hart--Mas-Colell (external) regret-matching learner for 2-player matrix games.

    Parameters
    ----------
    env : EconEnv
        The environment. Used only to read the discrete action counts
        (``env.action_space[...].n``) and, when no explicit ``payoff`` is given,
        the symmetric ``stage_payoff`` counterfactual oracle. The agent is **not**
        specialised to any single env.
    player_id : int
        This agent's index (0 or 1). The single opponent is ``1 - player_id``.
    payoff : (K x K_opp) array-like | callable | None
        Own-payoff oracle **from this player's perspective**: ``payoff[k, o]`` (or
        ``payoff(k, o)``) is the payoff of playing own action ``k`` against
        opponent action ``o``. Pass this for asymmetric games (each player gets its
        own matrix). If ``None`` (default) the oracle is taken from
        ``env.stage_payoff`` (valid for player-symmetric games such as the PD).
        A ``ValueError`` is raised when neither is available.
    decode_opp : callable | None
        Optional ``decode_opp(next_obs, own_action) -> opp_action``. Defaults to
        the EconGym memory-1 convention ``opp = int(next_obs) % K_opp``.

    Notes
    -----
    ``act`` ignores the observation: external regret matching plays an
    *unconditional* distribution over own actions (this is what yields convergence
    of the empirical play to the coarse-correlated-equilibrium set).
    """

    def __init__(self, env, player_id: int = 0, payoff=None, decode_opp=None):
        self.player_id = int(player_id)
        self.opp_id = 1 - self.player_id

        self.K = int(env.action_space[self.player_id].n)
        # opponent action count (symmetric fallback if the opponent space is absent)
        try:
            self.K_opp = int(env.action_space[self.opp_id].n)
        except (IndexError, AttributeError, TypeError):
            self.K_opp = self.K

        # --- resolve the exact counterfactual payoff matrix (K x K_opp) ---
        self._M = self._resolve_payoff_matrix(env, payoff)

        # opponent-action decoder (EconGym joint-action encoding by default)
        if decode_opp is not None:
            self._decode_opp = decode_opp
        else:
            k_opp = self.K_opp
            self._decode_opp = lambda next_obs, own: int(next_obs) % k_opp

        # learning state (allocated in reset)
        self.regret = None          # cumulative external regret, length K
        self.strategy_sum = None    # sum of mixed strategies played (avg strategy)
        self.action_counts = None   # realized-action histogram (empirical marginal)
        self.t_updates = 0          # number of update() calls (regret horizon)
        self._last_p = None
        self._pol_snap = None

    # ------------------------------------------------------------------
    # payoff-oracle resolution
    # ------------------------------------------------------------------
    def _resolve_payoff_matrix(self, env, payoff):
        """Return the exact (K x K_opp) own-payoff matrix, or raise if unavailable."""
        if payoff is not None:
            if callable(payoff):
                M = np.array([[float(payoff(k, o)) for o in range(self.K_opp)]
                              for k in range(self.K)], dtype=np.float64)
            else:
                M = np.asarray(payoff, dtype=np.float64)
                if M.shape != (self.K, self.K_opp):
                    raise ValueError(
                        f"payoff matrix shape {M.shape} != (K, K_opp)="
                        f"{(self.K, self.K_opp)}"
                    )
            return M
        # env-provided stage-payoff oracle (exposed by the symmetric EconGym matrix
        # envs; valid because those games are player-symmetric).
        oracle = getattr(env, "stage_payoff", None)
        if callable(oracle):
            return np.array([[float(oracle(k, o)) for o in range(self.K_opp)]
                             for k in range(self.K)], dtype=np.float64)
        raise ValueError(
            "RegretMatching needs full-information counterfactual payoffs, but the "
            "environment exposes no `stage_payoff(own, opp)` oracle and no explicit "
            "`payoff` matrix was supplied. Pass `payoff=` (own-perspective K x K_opp "
            "matrix or callable) for this env. (Regret matching is a full-information "
            "no-external-regret rule; the bandit setting requires a different "
            "algorithm.)"
        )

    def _counterfactuals(self, opp: int) -> np.ndarray:
        """Own payoff of every action ``k`` against the realized opponent action
        (the exact column ``u(., opp)`` of the payoff matrix)."""
        return self._M[:, opp]

    # ------------------------------------------------------------------
    # strategy
    # ------------------------------------------------------------------
    def _distribution(self) -> np.ndarray:
        """Regret-matching mixed strategy: proportional to positive regret,
        uniform when no regret is positive."""
        pos = np.maximum(self.regret, 0.0)
        s = pos.sum()
        if s > 0.0:
            return pos / s
        return np.full(self.K, 1.0 / self.K)

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------
    def reset(self, rng: np.random.Generator, track_conv: bool = False) -> None:
        # Consumes NO rng: regret matching starts from the uniform strategy
        # (empty regret vector), so no initial draw is needed.
        self.regret = np.zeros(self.K, dtype=np.float64)
        self.strategy_sum = np.zeros(self.K, dtype=np.float64)
        self.action_counts = np.zeros(self.K, dtype=np.int64)
        self.t_updates = 0
        self._last_p = None
        self._pol_snap = None

    def act(self, obs, t: int, rng: np.random.Generator) -> int:
        """Sample an action from the current regret-matching distribution.

        Draws exactly one ``rng.random()`` and maps it through the inverse CDF, so
        the reproducible stream advances by one draw per step (like Thompson).
        """
        p = self._distribution()
        self._last_p = p
        u = rng.random()
        a = int(np.searchsorted(np.cumsum(p), u, side="right"))
        if a >= self.K:              # guard against fp rounding at the top of the CDF
            a = self.K - 1
        return a

    def update(self, obs, action: int, reward: float, next_obs) -> None:
        """Accumulate external regret from the realized transition.

        The opponent's action is decoded from ``next_obs`` (the memory-1
        observation of the just-played joint profile); the counterfactual payoff
        of every own action against it feeds the regret update
        ``R[k] += u(k, opp) - u(action, opp)``.
        """
        action = int(action)
        opp = int(self._decode_opp(next_obs, action))
        cf = self._counterfactuals(opp)
        self.regret += cf - float(reward)
        # diagnostics: average strategy and realized-action histogram
        if self._last_p is not None:
            self.strategy_sum += self._last_p
        self.action_counts[action] += 1
        self.t_updates += 1

    # ------------------------------------------------------------------
    # analytics used by the validation test (closed-form regret benchmark)
    # ------------------------------------------------------------------
    def cumulative_regret(self) -> np.ndarray:
        """Cumulative external regret vector ``R_T`` (length ``K``)."""
        return self.regret.copy()

    def average_positive_regret(self) -> float:
        """``max_k R_T^+(k) / T`` -- the quantity Hannan consistency drives to 0."""
        if self.t_updates == 0:
            return 0.0
        return float(np.maximum(self.regret, 0.0).max() / self.t_updates)

    def regret_bound(self) -> float:
        """Closed-form Blackwell/regret-matching bound on the average positive regret:
        ``M * sqrt(K / T)`` with ``M`` the per-period payoff range. A guaranteed
        upper bound on :meth:`average_positive_regret`."""
        if self.t_updates == 0:
            return float("inf")
        return float(self.payoff_range() * np.sqrt(self.K / self.t_updates))

    def payoff_range(self) -> float:
        """Per-period payoff range ``M = max u - min u`` of this player's matrix."""
        return float(self._M.max() - self._M.min())

    def average_strategy(self) -> np.ndarray:
        """Time-average of the mixed strategies actually played (length ``K``).
        Converges into the CCE-consistent marginal set."""
        if self.strategy_sum is None or self.strategy_sum.sum() == 0.0:
            return np.full(self.K, 1.0 / self.K)
        return self.strategy_sum / self.strategy_sum.sum()

    def empirical_frequencies(self) -> np.ndarray:
        """Empirical frequency of each realized own action (length ``K``)."""
        total = self.action_counts.sum()
        if total == 0:
            return np.full(self.K, 1.0 / self.K)
        return self.action_counts / total

    # ------------------------------------------------------------------
    # optional convergence hooks (runner track_conv path)
    # ------------------------------------------------------------------
    def snapshot_policy(self) -> None:
        self._pol_snap = int(np.argmax(self._distribution()))

    def policy_stability(self) -> float:
        if self._pol_snap is None:
            return 1.0
        return float(self._pol_snap == int(np.argmax(self._distribution())))

    def cells_visited(self):
        return int((self.action_counts > 0).sum())

    def total_cells(self):
        return int(self.K)

    def min_visit(self):
        nz = self.action_counts[self.action_counts > 0]
        return int(nz.min()) if nz.size else 0
