"""Unit tests for Bertrand market physics."""
import numpy as np
import pytest

from econgym import BertrandEnv, QLearner, run_episode
from econgym.envs.bertrand import make_grid, step_profits


def test_grid():
    env = BertrandEnv(n=2, K=21, c=1.0, p_min=0.0, p_max=10.0)
    assert env.grid.shape == (21,)
    assert env.grid[0] == 0.0
    assert env.grid[-1] == 10.0
    assert np.allclose(env.grid, np.linspace(0.0, 10.0, 21))


def test_undercut_wins_all():
    # lowest price serves the whole unit of demand; higher price earns 0.
    grid = make_grid(21, 0.0, 10.0)
    c = 1.0
    # indices 5 (price 2.5) and 10 (price 5.0): 5 undercuts.
    pr = step_profits(np.array([5, 10]), grid, c)
    assert pr[1] == 0.0
    assert pr[0] == pytest.approx((grid[5] - c) * 1.0)


def test_tie_splits_equally():
    grid = make_grid(21, 0.0, 10.0)
    c = 1.0
    pr = step_profits(np.array([8, 8]), grid, c)
    assert pr[0] == pr[1]
    assert pr[0] == pytest.approx((grid[8] - c) * 0.5)


def test_nash_and_monopoly_k21():
    env = BertrandEnv(n=2, K=21, c=1.0, p_min=0.0, p_max=10.0)
    # K=21 grid includes p=1.0=c -> p_comp = c -> nash = 0.
    assert env.nash_profit() == 0.0
    assert env.monopoly_profit() == 4.5


def test_nash_k7_positive():
    # K=7 grid does NOT include p=1.0 exactly; lowest grid price >= c is > c.
    env = BertrandEnv(n=2, K=7, c=1.0, p_min=0.0, p_max=10.0)
    above = env.grid[env.grid >= 1.0]
    assert env.nash_profit() == pytest.approx((above.min() - 1.0) / 2)
    assert env.nash_profit() > 0.0


def test_reset_draws_and_obs_n2():
    env = BertrandEnv(n=2, K=7)
    rng = np.random.default_rng(0)
    # reset draws exactly one integers(0,K,size=n); replicate to compare.
    obs = env.reset(np.random.default_rng(0))
    expected_prev = np.random.default_rng(0).integers(0, 7, size=2)
    # n=2: obs = [prev[1], prev[0]]  (opponent's index)
    assert obs == [int(expected_prev[1]), int(expected_prev[0])]


def test_step_obs_is_opponent_index_n2():
    env = BertrandEnv(n=2, K=7)
    env.reset(np.random.default_rng(0))
    obs, rewards, terminated, truncated, info = env.step([3, 5])
    assert obs == [5, 3]                     # opponent indices, swapped
    assert list(info["prices"]) == [3, 5]
    assert rewards.shape == (2,)


def test_n_general_obs_tuple():
    env = BertrandEnv(n=3, K=5)          # grid = [0, 2.5, 5, 7.5, 10]
    env.reset(np.random.default_rng(1))
    obs, rewards, terminated, truncated, info = env.step([1, 2, 3])   # lowest = index 1 (price 2.5 > c)
    # agent i sees the other two indices, in agent order skipping self.
    assert obs[0] == (2, 3)
    assert obs[1] == (1, 3)
    assert obs[2] == (1, 2)
    # 3-way undercut: index 1 has the lowest (above-cost) price -> serves all.
    assert rewards[0] > 0 and rewards[1] == 0 and rewards[2] == 0
    assert rewards[0] == pytest.approx((env.grid[1] - env.c) * 1.0)


def test_n3_qlearner_full_episode_smoke():
    """End-to-end n>2 learning run: three QLearners through run_episode over a
    short horizon. Exercises the base-K joint-opponent state encoder for real
    (n_states = K**(n-1) states, tuple observations -> flat index), not just the
    env's observation encoding. Asserts shapes, finiteness, valid index ranges,
    and that the (n>2) encoder actually reached multiple states."""
    n, K, T = 3, 4, 200
    env = BertrandEnv(n=n, K=K, c=1.0, p_min=0.0, p_max=10.0)
    agents = [QLearner(env, alpha=0.1, gamma=0.95, epsilon=0.2, eps_decay=3e-4)
              for _ in range(n)]

    # base-K encoding: n-1 opponents, each in [0, K) -> K**(n-1) flat states.
    assert agents[0].n_states == K ** (n - 1) == 16

    res = run_episode(env, agents, T=T, seed=1, track_conv=True)

    assert res.prices.shape == (T, n)
    assert res.profits.shape == (T, n)
    assert res.prices.min() >= 0 and res.prices.max() < K
    assert np.isfinite(res.profits).all()
    # convergence diagnostics are populated for every firm
    assert res.converged in (True, False)
    assert res.pol_stable is not None and len(res.pol_stable) == n
    # the base-K encoder must have driven the agents through several states
    assert res.cells_visited is not None
    assert agents[0].cells_visited() > 1
    # sanity: some learning happened (Q table moved off its ~0 init somewhere)
    assert np.any(agents[0].Q != 0.0)
