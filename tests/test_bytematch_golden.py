"""Byte-for-byte faithfulness lock against a VENDORED golden trace.

The central scientific claim of EconGym v0 is that its Bertrand env + agents +
runner reproduce Paper 1's original ``simulation.py`` *byte for byte*. The
cross-check in ``test_metrics.py`` verifies that directly, but only when the
original source tree happens to be checked out (it skips otherwise). To keep the
faithfulness guarantee binding EVERYWHERE -- CI, a fresh clone, any machine, after
the original repo moves -- a small golden trace RECORDED FROM THE ORIGINAL CODE is
vendored into ``tests/data/bytematch_golden.npz`` (see the manifest alongside it
for provenance). These tests replay econgym against that recorded original output
with ZERO external dependency.

If any of these fail, the reproduction has genuinely drifted -- fix the RNG order
or the agent update, never the golden file.
"""
import json
import os

import numpy as np
import pytest

from econgym import BertrandEnv, QLearner, MeanBased, run_episode, metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "data", "bytematch_golden.npz")
_MANIFEST = os.path.join(_HERE, "data", "bytematch_golden_manifest.json")


@pytest.fixture(scope="module")
def golden():
    assert os.path.exists(_GOLDEN), f"vendored golden trace missing: {_GOLDEN}"
    with open(_MANIFEST) as f:
        man = json.load(f)
    return np.load(_GOLDEN), man


def _cfg(man):
    return (man["K"], man["T"], man["alpha"], man["gamma"], man["epsilon"],
            man["eps_decay"], man["c"], man["p_min"], man["p_max"])


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_qlearning_matches_vendored_original(golden, seed):
    """econgym Q-learning == the recorded ORIGINAL trace, byte for byte."""
    data, man = golden
    K, T, alpha, gamma, eps, decay, c, pmin, pmax = _cfg(man)

    env = BertrandEnv(n=2, K=K, c=c, p_min=pmin, p_max=pmax)
    agents = [QLearner(env, alpha=alpha, gamma=gamma, epsilon=eps,
                       eps_decay=decay) for _ in range(2)]
    res = run_episode(env, agents, T, seed, track_conv=True)

    assert np.array_equal(res.prices, data[f"q_price_{seed}"])
    assert np.allclose(res.profits, data[f"q_profit_{seed}"], atol=0, rtol=0)

    # metrics computed on the two traces therefore agree exactly
    d = metrics.delta_index(res.prices, res.profits, env.grid, env.c, env.n, 2000)
    d_g = metrics.delta_index(data[f"q_price_{seed}"], data[f"q_profit_{seed}"],
                              env.grid, env.c, env.n, 2000)
    assert d == pytest.approx(d_g)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_meanbased_matches_vendored_original(golden, seed):
    """econgym mean-based == the recorded ORIGINAL trace (post one-draw offset),
    byte for byte. The offset is already baked into the vendored golden file."""
    data, man = golden
    K, T, alpha, gamma, eps, decay, c, pmin, pmax = _cfg(man)

    env = BertrandEnv(n=2, K=K, c=c, p_min=pmin, p_max=pmax)
    agents = [MeanBased(env, epsilon=eps, eps_decay=decay) for _ in range(2)]
    res = run_episode(env, agents, T, seed, track_conv=True)

    assert np.array_equal(res.prices, data[f"m_price_{seed}"])
    assert np.allclose(res.profits, data[f"m_profit_{seed}"], atol=0, rtol=0)
