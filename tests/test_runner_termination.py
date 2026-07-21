"""Cover ``run_episode``'s early-termination + array-trim path.

The shipped envs that go through the raw runner (Bertrand, RepeatedPD) never set
``terminated=True``, so the runner's early-stop trimming was previously exercised
by no test. These tests drive a trivial dummy ``EconEnv`` that terminates after
``k`` steps (``k < T``) to lock down the advertised behavior: the returned
``Result`` reflects the early stop in ``T`` and in the trimmed trace shapes, and
the ``track_conv`` diagnostics behave correctly when the snapshot step is never
reached.
"""
import numpy as np
import pytest

from econgym import run_episode
from econgym.core import EconEnv, Discrete


class _CountdownEnv(EconEnv):
    """Two-agent env that terminates after exactly ``k`` steps.

    Native action == index (Discrete), so it composes with the raw runner. Reward
    is a constant per agent; the initial obs is drawn from the shared rng (one
    ``integers`` call) to honor the seeding contract.
    """

    def __init__(self, n=2, k=5, K=3):
        self.n = int(n)
        self.k = int(k)
        self.K = int(K)
        self._t = 0
        self.action_space = [Discrete(self.K) for _ in range(self.n)]
        self.observation_space = [Discrete(self.K) for _ in range(self.n)]

    def _reset(self):
        self._t = 0
        obs = self.rng.integers(0, self.K, size=self.n)
        return [int(o) for o in obs]

    def step(self, actions):
        self._t += 1
        obs = [int(a) for a in actions]
        rewards = np.full(self.n, 1.0, dtype=float)
        terminated = self._t >= self.k
        return obs, rewards, terminated, False, {"t": self._t}

    def equilibrium(self):
        return {"note": "dummy env, no economic content"}


class _ConstAgent:
    """Trivial constant-action agent (draws nothing from rng)."""

    def __init__(self, action=0):
        self._a = int(action)

    def reset(self, rng, track_conv=False):
        return None

    def act(self, obs, t, rng):
        return self._a

    def update(self, obs, action, reward, next_obs):
        return None

    # convergence hooks so track_conv=True works
    def snapshot_policy(self):
        self._snapped = True

    def policy_stability(self):
        return 1.0

    def cells_visited(self):
        return 0

    def total_cells(self):
        return self.K if hasattr(self, "K") else 0

    def min_visit(self):
        return 0


def test_early_termination_trims_result():
    """T reflects the early stop; traces are trimmed to (k, n)."""
    k, T, n = 5, 100, 2
    env = _CountdownEnv(n=n, k=k)
    agents = [_ConstAgent() for _ in range(n)]
    res = run_episode(env, agents, T=T, seed=0)
    assert res.T == k
    assert res.prices.shape == (k, n)
    assert res.profits.shape == (k, n)
    # every recorded action is the constant 0, rewards are the constant 1.0
    assert np.array_equal(res.prices, np.zeros((k, n), dtype=np.int64))
    assert np.array_equal(res.profits, np.ones((k, n), dtype=float))


def test_no_early_termination_uses_full_T():
    """When k >= T the env never terminates early: full horizon, no trim."""
    T, n = 8, 2
    env = _CountdownEnv(n=n, k=1000)   # never reached within T
    agents = [_ConstAgent() for _ in range(n)]
    res = run_episode(env, agents, T=T, seed=1)
    assert res.T == T
    assert res.prices.shape == (T, n)


def test_track_conv_with_early_terminate_before_snapshot():
    """If the env terminates BEFORE floor(0.9*T), the snapshot never fires; the
    diagnostics still populate without error (policy_stability default -> stable)."""
    k, T, n = 3, 100, 2   # snapshot would be at t=90, never reached
    env = _CountdownEnv(n=n, k=k)
    agents = [_ConstAgent() for _ in range(n)]
    res = run_episode(env, agents, T=T, seed=0, track_conv=True)
    assert res.T == k
    assert res.pol_stable == [1.0, 1.0]
    assert res.converged is True          # documented default when no snapshot
    assert res.infos is not None and len(res.infos) == k
