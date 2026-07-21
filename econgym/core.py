"""Shared multi-agent interface for EconGym (dependency-free).

This module provides the three primitives every environment builds on:

  * :class:`Discrete` / :class:`Box` -- lightweight spaces that MIRROR the
    ``gymnasium.spaces`` API surface (``n``/``low``/``high``/``shape``/``dtype``,
    ``sample``/``contains``) WITHOUT importing gymnasium. Runtime deps stay
    ``numpy`` only.
  * :class:`EconEnv` -- a PettingZoo-parallel-style, simultaneous-move,
    multi-agent base. All agents act at once; ``step`` takes a list of one
    action per agent and returns per-agent observations and rewards.

Seeding contract (the single source of episode randomness)
----------------------------------------------------------
There is exactly ONE ``np.random.Generator`` per episode. The runner creates it
once (``np.random.default_rng(seed)``) and threads it EVERYWHERE: agents draw
from it in agent order, then the env draws its initial state from the SAME
stream. Spaces and envs *consume* this Generator; they never own one. This is
what makes v0 Q-learning byte-exact: agent Q-tables are drawn before the env's
initial price profile, in a single unbroken stream.

``EconEnv.reset(seed=None, *, rng=None)`` honors that contract:
  * ``rng`` given  -> ``self.rng = rng``                 (SHARED stream; runner path)
  * else           -> ``self.rng = np.random.default_rng(seed)``  (standalone path)
"""
from __future__ import annotations

import numpy as np


class Discrete:
    """A finite set ``{start, ..., start + n - 1}``. Mirrors ``gymnasium.spaces.Discrete``."""

    def __init__(self, n: int, start: int = 0):
        self.n = int(n)
        self.start = int(start)
        self.shape = ()          # scalar
        self.dtype = np.int64

    def sample(self, rng: np.random.Generator) -> int:
        """Draw a uniform member using the SHARED episode ``rng`` (never own one)."""
        return self.start + int(rng.integers(self.n))

    def contains(self, x) -> bool:
        return self.start <= int(x) < self.start + self.n

    def __repr__(self):
        return f"Discrete({self.n}, start={self.start})"


class Box:
    """A closed box in ``R^d``. Mirrors ``gymnasium.spaces.Box`` (continuous)."""

    def __init__(self, low, high, shape=None, dtype=np.float64):
        self.low = np.broadcast_to(
            np.asarray(low, dtype=dtype),
            shape if shape is not None else np.shape(low),
        ).copy()
        self.high = np.broadcast_to(
            np.asarray(high, dtype=dtype),
            shape if shape is not None else np.shape(high),
        ).copy()
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw uniformly on ``[low, high]`` using the SHARED episode ``rng``."""
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self):
        return f"Box(low={self.low.min()}, high={self.high.max()}, shape={self.shape})"


class EconEnv:
    """Abstract base for all EconGym environments.

    Contract (every subclass honors this):
      * attribute  ``n`` : int                          number of agents
      * attribute  ``action_space`` : list[Space]       length n; per-agent action space
      * attribute  ``observation_space`` : list[Space]  length n; per-agent observation space
      * attribute  ``rng`` : np.random.Generator        the shared episode Generator (set by reset)

      * ``reset(seed=None, *, rng=None) -> obs``         obs is a list, one entry per agent
      * ``step(actions: list) -> (obs, rewards, terminated, truncated, info)``
      * ``equilibrium() -> dict``                        CLOSED-FORM benchmark for this env
      * ``benchmark = equilibrium``                      alias

    Subclasses implement only three hooks: ``_reset``, ``step``, ``equilibrium``.
    """

    metadata = {"name": "EconEnv", "simultaneous": True}

    # --- seeding contract (single shared Generator threaded through the episode) ---
    def reset(self, seed: int | None = None, *, rng: np.random.Generator | None = None):
        """Resolve the episode RNG, then delegate to the subclass ``_reset``.

        Exactly ONE of the two paths is taken:
          * ``rng`` given -> ``self.rng = rng``  (SHARED stream; used by run_episode so
            agents that already drew from ``rng`` keep byte-exact ordering).
          * else          -> ``self.rng = np.random.default_rng(seed)`` (standalone,
            gymnasium-style). ``np.random.default_rng`` passes an existing Generator
            through unchanged, so ``reset(some_generator)`` also works positionally.

        Returns the per-agent observation list from ``_reset``.
        """
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        return self._reset()

    # --- subclass hooks (the only three methods an env must implement) ---
    def _reset(self) -> list:
        """Draw initial state from ``self.rng``; return the per-agent obs list."""
        raise NotImplementedError

    def step(self, actions: list):
        """Advance one period. Return ``(obs, rewards, terminated, truncated, info)``."""
        raise NotImplementedError

    def equilibrium(self) -> dict:
        """Closed-form benchmark used by this env's validation test."""
        raise NotImplementedError

    def benchmark(self) -> dict:
        """Alias for :meth:`equilibrium`."""
        return self.equilibrium()
