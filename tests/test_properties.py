"""Property-based tests (Hypothesis): invariants that must hold across RANDOM
parameters, not just the fixed configs the per-env tests pin down.

These fuzz:
  * Cournot physics vs closed-form theory over the whole parameter space,
  * run_episode determinism under the shared-seed contract,
  * the space contract (sampled native actions are valid, step output is
    well-formed) for EVERY registered environment.
"""
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from econgym import BertrandEnv, CournotEnv, QLearner, list_envs, make, run_episode


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=6),
    a=st.floats(min_value=20.0, max_value=200.0),
    b=st.floats(min_value=0.5, max_value=5.0),
    c=st.floats(min_value=0.0, max_value=15.0),
)
def test_cournot_step_at_nash_realizes_closed_form(n, a, b, c):
    """Feeding the closed-form Nash quantities into the env's OWN step must
    realize the closed-form price P* and per-firm profit pi*, for any valid
    parameters. This ties the simulator to theory across the whole space."""
    assume(a - c > 5.0)                       # comfortably interior equilibrium
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_star = env.equilibrium()["q_i"]
    env.reset(seed=0)
    _obs, rewards, _term, _trunc, info = env.step([q_star] * n)

    P_star = (a + n * c) / (n + 1)
    pi_star = (a - c) ** 2 / (b * (n + 1) ** 2)
    assert info["P"] == pytest.approx(P_star, rel=1e-9, abs=1e-9)
    assert np.allclose(rewards, pi_star, rtol=1e-9, atol=1e-9)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000))
def test_run_episode_is_deterministic_given_the_seed(seed):
    """Same seed -> identical trajectory (the single-shared-rng contract)."""
    def once():
        env = BertrandEnv(n=2, K=7, c=1.0, p_max=10.0)
        agents = [QLearner(env, eps_decay=3e-4) for _ in range(2)]
        return run_episode(env, agents, T=200, seed=seed)

    r1, r2 = once(), once()
    assert np.array_equal(r1.prices, r2.prices)
    assert np.array_equal(r1.profits, r2.profits)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000))
def test_every_env_has_a_well_formed_space_and_step(seed):
    """For every registered env: sampled native actions live in their space, and
    one step returns per-agent obs/rewards of the right length with finite,
    real rewards and boolean termination flags."""
    for env_id in list_envs():
        env = make(env_id)
        obs = env.reset(seed=seed)
        assert len(obs) == env.n

        actions = []
        for space in env.action_space:
            a = space.sample(env.rng)
            assert space.contains(a), f"{env_id}: sampled {a!r} not in {space!r}"
            actions.append(a)

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        assert len(next_obs) == env.n
        assert len(rewards) == env.n
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert np.all(np.isfinite(np.asarray(rewards, dtype=float)))
