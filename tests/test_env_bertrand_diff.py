"""Validation tests for the differentiated-good Bertrand environment.

Per CONTRACT_v1 section 3.2. Validates the env's physics against its closed-form
symmetric interior Nash benchmark, the single non-negotiable quality bar:

  1. static      -- ``equilibrium()`` equals the hand-derived FOC formula
                    ``p* = (alpha + beta c)/(2 beta - gamma(n-1))`` (defaults hit
                    ``p*=4, q*=6, pi*=18``), cross-checked against
                    ``solvers.closed_form.bertrand_diff_nash``;
  2. convergence -- best-response iteration from a generic start reaches ``[p*]*n``;
  3. deviation   -- no unilateral price move beats ``pi_i*`` (rivals fixed at ``p*``);
  4. interior    -- ``q_i* > 0`` (demand not clamped);
plus a direct ``env.step`` / ``env.profit`` physics check that the simulator
reproduces ``pi_i*`` at ``p*`` exactly, and constructor-guard checks.
"""
import numpy as np
import pytest

from econgym import BertrandDiffEnv
from econgym.solvers.best_response import best_response_iteration
from econgym.solvers.closed_form import bertrand_diff_nash

DEFAULTS = dict(alpha=10.0, beta=2.0, gamma=1.0, c=1.0)


def _p_star(alpha, beta, gamma, c, n):
    """Independent recomputation of the closed-form symmetric Nash price."""
    return (alpha + beta * c) / (2.0 * beta - gamma * (n - 1))


# ----------------------------------------------------------------------
# 1. Static benchmark matches the solved FOC exactly.
# ----------------------------------------------------------------------
def test_static_matches_solved_foc():
    env = BertrandDiffEnv(n=2, **DEFAULTS)
    eq = env.equilibrium()
    # Numeric defaults hit p*=4, q*=6, pi*=18 exactly (contract appendix A).
    assert eq["p"] == pytest.approx(4.0, abs=1e-12)
    assert eq["q_i"] == pytest.approx(6.0, abs=1e-12)
    assert eq["profit_i"] == pytest.approx(18.0, abs=1e-12)
    # Hand-derived formula, recomputed independently of the solver.
    assert eq["p"] == pytest.approx(_p_star(n=2, **DEFAULTS), rel=1e-12)
    # Cross-check against the closed-form solver (single source of truth).
    assert eq == bertrand_diff_nash(n=2, **DEFAULTS)
    # benchmark alias returns the identical dict.
    assert env.benchmark() == eq


@pytest.mark.parametrize("n", [2, 3, 4])
def test_static_matches_solved_foc_general_n(n):
    # gamma small enough that 2*beta - gamma*(n-1) > 0 for every n tested.
    alpha, beta, gamma, c = 12.0, 3.0, 1.0, 2.0
    env = BertrandDiffEnv(n=n, alpha=alpha, beta=beta, gamma=gamma, c=c)
    eq = env.equilibrium()
    p_star = _p_star(alpha, beta, gamma, c, n)
    q_star = alpha - beta * p_star + gamma * (n - 1) * p_star
    pi_star = (p_star - c) * q_star
    assert eq["p"] == pytest.approx(p_star, rel=1e-12)
    assert eq["q_i"] == pytest.approx(q_star, rel=1e-12)
    assert eq["profit_i"] == pytest.approx(pi_star, rel=1e-12)
    assert eq == bertrand_diff_nash(alpha, beta, gamma, c, n)


# ----------------------------------------------------------------------
# 2. Best-response iteration converges to [p*]*n.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
def test_best_response_converges(n):
    alpha, beta, gamma, c = 12.0, 3.0, 1.0, 2.0
    env = BertrandDiffEnv(n=n, alpha=alpha, beta=beta, gamma=gamma, c=c)
    p_star = _p_star(alpha, beta, gamma, c, n)
    # Guard: the best-response map is a contraction (stable fixed point).
    assert env.equilibrium()["contraction_factor"] < 1.0
    # From c*ones.
    fp = best_response_iteration(env.best_response, np.full(n, c), tol=1e-12)
    assert np.allclose(fp, p_star, atol=1e-6)
    # From a random x0 >= c.
    rng = np.random.default_rng(0)
    x0 = c + rng.uniform(0.0, 5.0, size=n)
    fpr = best_response_iteration(env.best_response, x0, tol=1e-12)
    assert np.allclose(fpr, p_star, atol=1e-6)


# ----------------------------------------------------------------------
# 3. No profitable unilateral deviation from p* (rivals fixed at p*).
# ----------------------------------------------------------------------
def test_no_profitable_deviation():
    env = BertrandDiffEnv(n=2, **DEFAULTS)
    eq = env.equilibrium()
    p_star, pi_star = eq["p"], eq["profit_i"]
    # Fine grid of firm-0 deviations spanning [c, p_cap]; firm 1 stays at p*.
    grid = np.linspace(env.c, env.p_cap, 4001)
    best_dev = max(env.profit(np.array([p0, p_star]))[0] for p0 in grid)
    assert best_dev <= pi_star + 1e-9
    # The equilibrium price is itself the best response (max on the continuum).
    assert env.profit(np.array([p_star, p_star]))[0] == pytest.approx(pi_star, rel=1e-12)
    assert env.best_response(np.array([p_star, p_star]))[0] == pytest.approx(p_star, rel=1e-12)


# ----------------------------------------------------------------------
# 4. Interior equilibrium: demand strictly positive (not clamped).
# ----------------------------------------------------------------------
def test_interior_positive_demand():
    env = BertrandDiffEnv(n=2, **DEFAULTS)
    eq = env.equilibrium()
    q = env.demand(np.full(2, eq["p"]))
    assert np.all(q > 0.0)
    assert q[0] == pytest.approx(eq["q_i"], rel=1e-12)


# ----------------------------------------------------------------------
# 5. Env.step / env.profit physics reproduce the benchmark exactly at p*.
# ----------------------------------------------------------------------
def test_env_step_physics_matches_benchmark():
    env = BertrandDiffEnv(n=2, **DEFAULTS)
    eq = env.equilibrium()
    p_star, q_star, pi_star = eq["p"], eq["q_i"], eq["profit_i"]
    env.reset(seed=0)
    obs, rewards, terminated, truncated, info = env.step([p_star, p_star])
    assert terminated is False and truncated is False
    assert isinstance(info, dict) and "prices" in info and "demand" in info
    assert rewards[0] == pytest.approx(pi_star, rel=1e-12)
    assert rewards[1] == pytest.approx(pi_star, rel=1e-12)
    assert info["demand"][0] == pytest.approx(q_star, rel=1e-12)
    # profit() pure function agrees with the step rewards.
    assert np.allclose(env.profit([p_star, p_star]), rewards)
    # Observation shape: each firm sees the (n-1) rival prices.
    assert len(obs) == 2 and obs[0].shape == (1,) and obs[0][0] == pytest.approx(p_star)


def test_seeding_contract_shared_stream():
    """reset(rng=...) consumes the shared stream (one uniform(c,p_cap,size=n) draw)."""
    env = BertrandDiffEnv(n=2, **DEFAULTS)
    shared = np.random.default_rng(7)
    env.reset(rng=shared)
    twin = np.random.default_rng(7)
    twin.uniform(env.c, env.p_cap, size=2)     # replicate the one env draw
    assert shared.random() == twin.random()    # streams aligned -> no extra draws
    assert env.rng is shared


# ----------------------------------------------------------------------
# Constructor guards (interiority / substitutes regime).
# ----------------------------------------------------------------------
def test_constructor_rejects_bad_params():
    with pytest.raises(ValueError):
        BertrandDiffEnv(n=2, alpha=10.0, beta=1.0, gamma=1.5, c=1.0)   # gamma > beta
    with pytest.raises(ValueError):
        BertrandDiffEnv(n=4, alpha=10.0, beta=2.0, gamma=1.9, c=1.0)   # 2b - g(n-1) < 0
    with pytest.raises(ValueError):
        BertrandDiffEnv(n=1, **DEFAULTS)                                # n < 2
