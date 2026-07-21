"""Runner regression: ``run_episode`` must map an agent's action INDEX to the
env's NATIVE action for continuous (Box) action spaces -- otherwise index-based
agents get stepped with raw grid indices on continuous envs (silent wrong
numbers). Discrete envs (Bertrand/repeated-PD) are unaffected: index == native
there, so the trace stays int64 and byte-exact.
"""
import numpy as np
import pytest

from econgym import Agent, CournotEnv, FictitiousPlay, run_episode


class _IndexOnlyAgent(Agent):
    """Minimal agent that returns a bare action index and exposes NO ``native``
    mapping -- the kind that cannot be driven on a continuous action space."""

    def act(self, obs, t, rng):
        return 0

    def update(self, obs, action, reward, next_obs):
        return None


def test_run_episode_rejects_index_only_agent_on_continuous_env():
    """An index-only agent (no native mapping) on a Box-action env must raise a
    clear error instead of silently stepping raw indices as native actions."""
    env = CournotEnv()
    agents = [_IndexOnlyAgent() for _ in range(env.n)]
    with pytest.raises(ValueError, match="native"):
        run_episode(env, agents, T=10, seed=0)


def test_run_episode_maps_index_to_native_on_continuous_env():
    """FictitiousPlay driven through the shared run_episode on Cournot must play
    NATIVE quantities converging to the closed-form Nash q*, not grid indices."""
    env = CournotEnv()                    # n=3, a=100, b=1, c=10 -> q* = 22.5 (a grid point)
    agents = [FictitiousPlay(env, i) for i in range(env.n)]
    res = run_episode(env, agents, T=5000, seed=0)
    q_star = env.equilibrium()["q_i"]     # 22.5

    # continuous env -> the recorded action trace is NATIVE quantities (floats),
    # not int64 grid indices
    assert res.prices.dtype == np.float64
    # ... and fictitious play converges to the Nash quantity (within one grid step)
    np.testing.assert_allclose(res.prices[-1], q_star, atol=1.0)
