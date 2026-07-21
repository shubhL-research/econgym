"""Core interface tests: dependency-free spaces, the EconEnv seeding contract,
and the Bertrand ``equilibrium()`` benchmark hook validated against its
closed-form (discrete-grid) values.

These lock the shared interface v1's environment builders reuse:
  * spaces mirror the gymnasium surface but consume the SHARED episode rng;
  * ``EconEnv.reset(rng=...)`` threads one Generator (byte-exact ordering) while
    ``reset(seed=...)`` / ``reset(generator)`` also work standalone;
  * ``BertrandEnv.equilibrium()`` equals the hand-derived closed form EXACTLY.
"""
import numpy as np
import pytest

from econgym import BertrandEnv, Box, Discrete, EconEnv


# ----------------------------------------------------------------------
# Spaces: no gymnasium import, correct surface, consume the shared rng.
# ----------------------------------------------------------------------
def test_no_gymnasium_dependency():
    import econgym.core as core
    import sys
    assert "gymnasium" not in sys.modules
    # the module never imports it
    assert not hasattr(core, "gymnasium")


def test_discrete_surface_and_sampling():
    d = Discrete(7)
    assert d.n == 7 and d.start == 0 and d.shape == () and d.dtype == np.int64
    assert d.contains(0) and d.contains(6) and not d.contains(7) and not d.contains(-1)
    # sample consumes the SHARED rng -> reproducible from a seed, matching a twin draw
    a = d.sample(np.random.default_rng(0))
    b = int(np.random.default_rng(0).integers(7))
    assert a == b
    d2 = Discrete(4, start=10)
    assert d2.contains(10) and d2.contains(13) and not d2.contains(9) and not d2.contains(14)


def test_box_surface_and_sampling():
    b = Box(low=0.0, high=1.0, shape=(3,))
    assert b.shape == (3,) and b.dtype == np.float64
    assert b.contains([0.0, 0.5, 1.0]) and not b.contains([0.0, 1.5, 0.0])
    x = b.sample(np.random.default_rng(1))
    y = np.random.default_rng(1).uniform(np.zeros(3), np.ones(3))
    assert np.allclose(x, y)
    # broadcast low/high
    b2 = Box(low=[0.0, -1.0], high=[2.0, 3.0])
    assert b2.shape == (2,)
    assert b2.low.tolist() == [0.0, -1.0] and b2.high.tolist() == [2.0, 3.0]


# ----------------------------------------------------------------------
# EconEnv seeding contract.
# ----------------------------------------------------------------------
def test_reset_rng_uses_shared_stream():
    """When ``rng`` is passed, the env consumes THAT generator (no fresh stream):
    the env's draw is exactly the next draw of the shared stream."""
    env = BertrandEnv(n=2, K=7)
    shared = np.random.default_rng(42)
    env.reset(rng=shared)                       # draws integers(0,7,size=2) from shared
    twin = np.random.default_rng(42)
    twin.integers(0, 7, size=2)                 # replicate the one env draw
    # both streams must now be at the same position
    assert shared.random() == twin.random()
    assert env.rng is shared


def test_reset_seed_path_is_standalone():
    """``reset(seed=k)`` builds a fresh default_rng(k); ``reset(generator)`` passes
    a Generator straight through (default_rng returns it unchanged)."""
    e1 = BertrandEnv(n=2, K=7)
    obs1 = e1.reset(seed=5)
    e2 = BertrandEnv(n=2, K=7)
    obs2 = e2.reset(np.random.default_rng(5))    # positional Generator == seed path
    assert obs1 == obs2
    expected = np.random.default_rng(5).integers(0, 7, size=2)
    assert obs1 == [int(expected[1]), int(expected[0])]


def test_step_returns_5_tuple():
    env = BertrandEnv(n=2, K=7)
    env.reset(seed=0)
    out = env.step([3, 5])
    assert len(out) == 5
    obs, rewards, terminated, truncated, info = out
    assert terminated is False and truncated is False
    assert isinstance(info, dict) and "prices" in info


def test_base_hooks_raise_not_implemented():
    class Empty(EconEnv):
        pass
    e = Empty()
    with pytest.raises(NotImplementedError):
        e.reset(seed=0)                          # reset -> _reset (not implemented)
    with pytest.raises(NotImplementedError):
        e.step([])
    with pytest.raises(NotImplementedError):
        e.equilibrium()


# ----------------------------------------------------------------------
# Bertrand equilibrium() hook validated against the closed form EXACTLY.
# ----------------------------------------------------------------------
def test_bertrand_equilibrium_matches_closed_form_k21():
    env = BertrandEnv(n=2, K=21, c=1.0, p_min=0.0, p_max=10.0)
    eq = env.equilibrium()
    # K=21 grid includes p=1.0=c -> competitive price = c, per-firm Nash profit = 0.
    assert eq["nash_price"] == 1.0
    assert eq["nash_profit_per_firm"] == 0.0 == env.nash_profit()
    assert eq["monopoly_price"] == 10.0
    assert eq["monopoly_profit_per_firm"] == 4.5 == env.monopoly_profit()
    # benchmark alias returns the identical dict
    assert env.benchmark() == eq


def test_bertrand_equilibrium_matches_closed_form_k7():
    env = BertrandEnv(n=2, K=7, c=1.0, p_min=0.0, p_max=10.0)
    eq = env.equilibrium()
    above = env.grid[env.grid >= env.c]
    p_comp = float(above.min())
    assert eq["nash_price"] == p_comp
    assert eq["nash_profit_per_firm"] == pytest.approx((p_comp - env.c) / env.n)
    assert eq["nash_profit_per_firm"] > 0.0     # K=7 grid skips p=c exactly
