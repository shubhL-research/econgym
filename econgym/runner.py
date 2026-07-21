"""Episode driver: wire an environment to a list of agents and run T steps.

``run_episode`` is n-general (never hard-codes 2) yet reproduces n=2 exactly by
preserving the original RNG call order (see CONTRACT.md Section 7):

    rng = np.random.default_rng(seed)
    for agent in agents:        # agent order 0,1,...  (QLearner draws Q here)
        agent.reset(rng, track_conv=track_conv)
    obs = env.reset(rng)        # ONE rng.integers(0,K,size=n) draw
    for t in range(T):
        actions = [agents[i].act(obs[i], t, rng) for i in range(n)]  # order 0,1,...
        next_obs, rewards, info = env.step(actions)
        for i in range(n):
            agents[i].update(obs[i], actions[i], rewards[i], next_obs[i])
        ...
        obs = next_obs

The list comprehension resolves agent 0 fully (its ``rng.random()`` then optional
``rng.integers``) before agent 1 -- exactly the original's
``a0 = ...; a1 = ...`` -- which is why n=2 byte-matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .core import Box


@dataclass
class Result:
    prices: np.ndarray            # shape (T, n); the EXECUTED-action trace (alias: actions).
                                  #   int64 action INDICES for Discrete envs (Bertrand);
                                  #   native float actions for continuous Box envs (e.g. Cournot q).
    profits: np.ndarray           # shape (T, n) float the per-agent REWARD trace
    n: int
    T: int
    seed: int
    # grid-specific fields: optional, populated only when the env exposes them
    # (Bertrand/discretized envs); None for envs without a price grid.
    K: int | None = None          # env.K if present, else None
    grid: np.ndarray | None = None  # env.grid if present, else None
    c: float | None = None        # env.c if present, else None
    # convergence diagnostics (populated only when track_conv=True, else None):
    pol_stable: list | None = None
    converged: bool | None = None
    cells_visited: list | None = None
    total_cells: int | None = None
    min_visit: list | None = None
    # generic extras:
    infos: list | None = None     # per-step info dicts (kept only when track_conv=True)

    @property
    def actions(self) -> np.ndarray:
        """Generic alias; ``prices`` stays the primary name for Bertrand."""
        return self.prices


def run_episode(env, agents, T, seed, track_conv=False) -> Result:
    """Run one episode of ``T`` steps and return a :class:`Result`.

    Parameters
    ----------
    env : EconEnv
        Any EconGym environment (``env.n`` agents). Its ``reset`` is called with
        the shared ``rng``; its ``step`` returns the 5-tuple ``(obs, rewards,
        terminated, truncated, info)``. Bertrand's execution path is byte-identical
        to v0 (``terminated``/``truncated`` are always ``False``).
    agents : sequence of Agent
        Length must equal ``env.n``.
    T : int
        Number of periods (horizon cap; envs may terminate earlier).
    seed : int
        Seed for the single shared ``np.random.default_rng``.
    track_conv : bool
        If True, snapshot greedy policies at ``floor(0.9*T)`` and populate the
        convergence diagnostics on the returned ``Result``.
    """
    n = env.n
    if len(agents) != n:
        raise ValueError(f"expected {n} agents for n={n}, got {len(agents)}")

    # --- action-space bridge -------------------------------------------------
    # Agents return an action INDEX (Agent.act contract). For a Discrete action
    # space the index IS the native action (Bertrand: the price index), so
    # nothing changes and the trace stays int64/byte-exact. For a continuous
    # (Box) action space the env's ``step`` consumes NATIVE values, so the index
    # is mapped through the agent's ``native(index)`` hook before stepping and
    # the trace holds native floats.
    spaces = getattr(env, "action_space", None)
    to_native = []
    continuous = False
    for i, ag in enumerate(agents):
        is_box = spaces is not None and isinstance(spaces[i], Box)
        native = getattr(ag, "native", None)
        if is_box and native is None:
            raise ValueError(
                f"agent {i} ({type(ag).__name__}) returns action indices but env "
                f"{type(env).__name__} has a continuous (Box) action space and the "
                f"agent exposes no native(index) mapping; use a native-action agent "
                f"(e.g. FictitiousPlay / BestResponse) or add a native() method."
            )
        continuous = continuous or is_box
        to_native.append(native)
    needs_map = any(f is not None for f in to_native)

    rng = np.random.default_rng(seed)                 # 1) the single shared Generator

    # 2) agent inits, in agent order (QLearner draws its Q table here). We pass
    #    track_conv so agents can skip allocating diagnostic-only state (e.g.
    #    QLearner's visit counter) when no diagnostics will be read. reset must
    #    not touch the rng based on this flag, so the byte-exact stream is intact.
    for agent in agents:
        agent.reset(rng, track_conv=track_conv)

    # 3) env draws the initial state from the SAME rng (Bertrand: one integers call).
    obs = env.reset(rng=rng)

    prices = np.empty((T, n), dtype=np.float64 if continuous else np.int64)
    profits = np.empty((T, n), dtype=np.float64)
    infos = [] if track_conv else None
    t_snap = int(0.9 * T)

    # local bindings keep the hot loop allocation-light
    acts = [a.act for a in agents]
    upds = [a.update for a in agents]
    step = env.step
    rng_local = rng

    t_final = T
    for t in range(T):
        actions = [acts[i](obs[i], t, rng_local) for i in range(n)]   # raw INDICES
        # map index -> native action for continuous envs; identity for Discrete.
        step_actions = actions if not needs_map else [
            to_native[i](actions[i]) if to_native[i] is not None else actions[i]
            for i in range(n)
        ]
        next_obs, rewards, terminated, truncated, info = step(step_actions)
        for i in range(n):
            upds[i](obs[i], actions[i], rewards[i], next_obs[i])       # agent keeps its index
        prices[t] = step_actions
        profits[t] = rewards
        if infos is not None:
            infos.append(info)
        if track_conv and t == t_snap:
            for a in agents:
                a.snapshot_policy()
        obs = next_obs
        if terminated or truncated:      # Bertrand: always False -> full T, never trims
            t_final = t + 1
            break

    if t_final != T:                     # only trims for envs that terminate early
        prices, profits = prices[:t_final], profits[:t_final]

    result = Result(
        prices=prices, profits=profits, n=n, T=t_final, seed=seed,
        K=getattr(env, "K", None), grid=getattr(env, "grid", None),
        c=getattr(env, "c", None), infos=infos,
    )
    if track_conv:
        result.pol_stable = [a.policy_stability() for a in agents]
        result.converged = metrics.is_converged(result)
        result.cells_visited = [a.cells_visited() for a in agents]
        result.total_cells = agents[0].total_cells() if agents else None
        result.min_visit = [a.min_visit() for a in agents]
    return result
