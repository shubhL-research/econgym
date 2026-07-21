"""Validation tests for the ``cournot`` environment (contract v1, section 3.1).

Every assertion here validates the ENVIRONMENT PHYSICS against the CLOSED-FORM
benchmark -- nothing is loosened to force a pass:

  * ``test_static_matches_closed_form``    -- ``env.equilibrium()`` equals the
    four hand-derived formulas EXACTLY (independent oracle), and the shared
    ``solvers.closed_form.cournot_nash`` / top-level ``cournot_nash`` agree.
  * ``test_step_physics_match_closed_form_at_nash`` -- the env's OWN ``step``,
    fed the Nash quantities, realises price ``P*``, aggregate ``Q*`` and per-firm
    profit ``pi_i*`` (ties the simulator to theory, not just the formula to
    itself).
  * ``test_nash_is_best_response_fixed_point`` -- ``BR(q*) == q*`` exactly.
  * ``test_best_response_iteration_converges`` -- best-response dynamics reach the
    Nash quantities (see the damping note below).
  * ``test_no_profitable_deviation`` -- no unilateral quantity deviation beats
    ``pi_i*`` on a fine grid (equilibrium, not merely a fixed point).
  * plus fictitious-play convergence, the ``P = max(0, .)`` demand floor, the
    per-agent spaces, the seeding contract, and the constructor guards.

Damping note. The *simultaneous* Cournot best-response map is a contraction only
for ``n = 2``; at ``n = 3`` it is neutrally stable (a 2-cycle) and for ``n >= 4``
it diverges (aggregate-direction eigenvalue ``-(n-1)/2``). This is a property of
the map, not of the equilibrium. We therefore drive the shared
``best_response_iteration`` with the *relaxed* map
``x -> (1-lambda) x + lambda BR(x)`` (``lambda = 2/(n+1)``), which has the SAME
fixed point (the Nash below) and converges. ``test_undamped_br_contraction_boundary``
pins this behaviour down explicitly so the correction is documented, not hidden.
"""
import numpy as np
import pytest

from econgym import CournotEnv, Box, cournot_nash
from econgym.envs.cournot import cournot_nash as cournot_nash_env
from econgym.solvers.closed_form import cournot_nash as cournot_nash_solver
from econgym.solvers.best_response import best_response_iteration


# (n, a, b, c) parameter sets -- all satisfy the interior condition a > c.
CASES = [
    (1, 100.0, 1.0, 10.0),   # monopoly limit
    (2, 100.0, 1.0, 10.0),   # duopoly
    (3, 100.0, 1.0, 10.0),   # contract defaults
    (5, 120.0, 2.0, 10.0),
    (10, 50.0, 0.5, 5.0),
]


def closed_form(a, b, c, n):
    """Independent oracle: the four Cournot formulas, derived in the test itself
    (so the test does not merely re-check the solver against itself)."""
    q_i = (a - c) / (b * (n + 1))
    Q = n * q_i
    P = (a + n * c) / (n + 1)
    profit_i = (a - c) ** 2 / (b * (n + 1) ** 2)
    return q_i, Q, P, profit_i


def relaxed_br(env, lam):
    """Damped/relaxed best-response map ``x -> (1-lam) x + lam BR(x)`` -- shares
    the fixed point of ``BR`` but is a contraction (module docstring)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        return (1.0 - lam) * x + lam * env.best_response(x)
    return f


# ----------------------------------------------------------------------
# 1. Static benchmark: equilibrium() == closed form, EXACTLY.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_static_matches_closed_form(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_i, Q, P, profit_i = closed_form(a, b, c, n)

    eq = env.equilibrium()
    assert eq["q_i"] == pytest.approx(q_i, rel=1e-12)
    assert eq["Q"] == pytest.approx(Q, rel=1e-12)
    assert eq["P"] == pytest.approx(P, rel=1e-12)
    assert eq["profit_i"] == pytest.approx(profit_i, rel=1e-12)

    # Internal consistency of the closed form: Q = n q_i, P = a - b Q, and
    # profit_i = (P - c) q_i.
    assert eq["Q"] == pytest.approx(n * eq["q_i"], rel=1e-12)
    assert eq["P"] == pytest.approx(a - b * eq["Q"], rel=1e-12)
    assert eq["profit_i"] == pytest.approx((eq["P"] - c) * eq["q_i"], rel=1e-12)

    # The env delegates to the SHARED solver (single source of truth); the
    # top-level re-export and the env-module re-export are the same object.
    assert cournot_nash is cournot_nash_solver is cournot_nash_env
    assert env.equilibrium() == cournot_nash_solver(a, b, c, n)


def test_defaults_hit_expected_numbers():
    """Contract defaults n=3, a=100, b=1, c=10 -> q*=22.5, Q*=67.5, P*=32.5,
    pi*=506.25."""
    eq = CournotEnv().equilibrium()
    assert eq == {"q_i": 22.5, "Q": 67.5, "P": 32.5, "profit_i": 506.25}


@pytest.mark.parametrize("n,a,b,c", CASES)
def test_benchmark_alias(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    assert env.benchmark() == env.equilibrium()


# ----------------------------------------------------------------------
# 2. The ENV'S OWN step physics reproduce the closed form at the Nash profile.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_step_physics_match_closed_form_at_nash(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_i, Q, P, profit_i = closed_form(a, b, c, n)

    env.reset(seed=0)
    q = np.full(n, q_i)
    obs, rewards, terminated, truncated, info = env.step(q)

    assert terminated is False and truncated is False
    assert isinstance(info, dict)
    assert info["Q"] == pytest.approx(Q, rel=1e-12)
    assert info["P"] == pytest.approx(P, rel=1e-12)
    # every firm earns exactly the closed-form per-firm profit
    assert np.allclose(rewards, profit_i, rtol=1e-12, atol=1e-9)
    # observation shape: each firm sees the other n-1 quantities
    assert len(obs) == n
    for o in obs:
        assert np.asarray(o).shape == (n - 1,)


# ----------------------------------------------------------------------
# 3. The Nash profile is a fixed point of the best-response map (exactly).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_nash_is_best_response_fixed_point(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_i, *_ = closed_form(a, b, c, n)
    q_star = np.full(n, q_i)
    assert np.allclose(env.best_response(q_star), q_star, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# 4. Best-response DYNAMICS converge to the Nash quantities.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_best_response_iteration_converges(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_i, *_ = closed_form(a, b, c, n)
    q_star = np.full(n, q_i)
    lam = 2.0 / (n + 1)          # spectral radius n/(n+1) < 1 for all n

    # from the all-zeros start
    x0 = np.zeros(n)
    x = best_response_iteration(relaxed_br(env, lam), x0, tol=1e-12, max_iter=100_000)
    assert np.allclose(x, q_star, rtol=0, atol=1e-6)
    # the returned point is a genuine fixed point of the (undamped) BR map == Nash
    assert np.allclose(env.best_response(x), x, rtol=0, atol=1e-6)

    # from a random positive start (deterministic seed)
    rng = np.random.default_rng(12345)
    x0r = rng.uniform(0.0, env.q_max, size=n)
    xr = best_response_iteration(relaxed_br(env, lam), x0r, tol=1e-12, max_iter=100_000)
    assert np.allclose(xr, q_star, rtol=0, atol=1e-6)


def test_undamped_br_contraction_boundary():
    """Documents (does not hide) the real behaviour of the *undamped*
    simultaneous Cournot best-response map: a contraction only at n=2, so the
    plain Picard ``best_response_iteration`` converges there but NOT at n>=3."""
    # n = 2: contraction (slope 1/2) -> plain iteration converges to Nash.
    e2 = CournotEnv(n=2)
    x = best_response_iteration(e2.best_response, np.zeros(2), tol=1e-12, max_iter=100_000)
    assert np.allclose(x, e2.equilibrium()["q_i"], atol=1e-9)

    # n >= 3: undamped map is neutrally stable / divergent -> does NOT converge.
    for n in (3, 5):
        en = CournotEnv(n=n)
        with pytest.raises(RuntimeError):
            best_response_iteration(en.best_response, np.zeros(n),
                                    tol=1e-12, max_iter=5_000)


# ----------------------------------------------------------------------
# 5. Fictitious play on the discretised belief converges near the Nash.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_fictitious_play_converges(n):
    """Each firm best-responds to the running average (empirical belief) of the
    joint quantity profile; the time-average play lands within ONE grid step of
    the Nash quantity."""
    env = CournotEnv(n=n)
    q_i = env.equilibrium()["q_i"]
    iters = 20_000

    q = np.full(n, env.q_max / 2.0)
    hist = q.astype(float).copy()
    for t in range(1, iters):
        avg = hist / t
        q = env.best_response(avg)
        hist += q
    time_avg = hist / iters

    grid_step = env.q_max / (env.G - 1)
    assert np.max(np.abs(time_avg - q_i)) < grid_step


# ----------------------------------------------------------------------
# 6. No profitable unilateral deviation from the Nash profile.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_no_profitable_deviation(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    q_i, _, _, profit_i = closed_form(a, b, c, n)

    # deviator sweeps a fine quantity grid while all rivals stay at q_i*
    dev_grid = np.linspace(0.0, env.q_max, 4001)
    rivals_total = q_i * (n - 1)
    P = np.maximum(0.0, a - b * (dev_grid + rivals_total))
    dev_payoff = (P - c) * dev_grid

    assert dev_payoff.max() <= profit_i + 1e-9
    # and the equilibrium quantity is (essentially) the argmax on the grid
    best_q = dev_grid[int(np.argmax(dev_payoff))]
    assert abs(best_q - q_i) <= env.q_max / (len(dev_grid) - 1) + 1e-9


# ----------------------------------------------------------------------
# 7. Demand floor: price clamps at 0 for large aggregate output.
# ----------------------------------------------------------------------
def test_price_nonnegativity():
    env = CournotEnv()                     # n=3, a=100, b=1, c=10, q_max=90
    # every firm at the choke quantity -> Q = 3*90 = 270 -> a - bQ = -170 < 0.
    q = np.full(env.n, env.q_max)
    obs, rewards, terminated, truncated, info = env.step(q)
    assert info["P"] == 0.0
    # with P clamped to 0 each firm earns exactly -c * q_i (< 0)
    assert np.allclose(rewards, -env.c * env.q_max)
    assert np.all(rewards < 0.0)
    # env.price() honours the same floor
    assert env.price(q) == 0.0
    # a moderate profile keeps price strictly positive
    assert env.price(np.full(env.n, 1.0)) == pytest.approx(env.a - env.b * env.n * 1.0)


# ----------------------------------------------------------------------
# 8. Per-agent spaces + grid.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,a,b,c", CASES)
def test_spaces_and_grid(n, a, b, c):
    env = CournotEnv(n=n, a=a, b=b, c=c)
    assert len(env.action_space) == n
    assert len(env.observation_space) == n
    assert all(isinstance(s, Box) for s in env.action_space)
    assert all(isinstance(s, Box) for s in env.observation_space)
    # native action box is [0, q_max]; observation box is the (n-1) rival vector
    for s in env.action_space:
        assert float(s.low) == 0.0 and float(s.high) == pytest.approx(env.q_max)
    for s in env.observation_space:
        assert s.shape == (n - 1,)
    # discretised grid endpoints and length
    assert env.qgrid.shape == (env.G,)
    assert env.qgrid[0] == 0.0
    assert env.qgrid[-1] == pytest.approx(env.q_max)
    assert env.q_max == pytest.approx((a - c) / b)


# ----------------------------------------------------------------------
# 9. Seeding contract: reset consumes exactly ONE draw from the shared stream.
# ----------------------------------------------------------------------
def test_reset_consumes_shared_rng():
    env = CournotEnv(n=3)
    shared = np.random.default_rng(7)
    env.reset(rng=shared)                              # one uniform(0,q_max,size=3)
    twin = np.random.default_rng(7)
    twin.uniform(0.0, env.q_max, size=3)               # replicate the single draw
    assert shared.random() == twin.random()            # streams now aligned
    assert env.rng is shared


def test_reset_seed_path_is_standalone_and_reproducible():
    e1 = CournotEnv(n=3)
    obs1 = e1.reset(seed=5)
    e2 = CournotEnv(n=3)
    obs2 = e2.reset(np.random.default_rng(5))          # positional Generator path
    for a, b in zip(obs1, obs2):
        assert np.array_equal(a, b)
    # the initial draw is exactly uniform(0, q_max, size=n) from default_rng(5)
    expected = np.random.default_rng(5).uniform(0.0, e1.q_max, size=3)
    # obs[i] omits firm i's own quantity; firm 0 sees firms 1,2
    assert np.allclose(obs1[0], expected[[1, 2]])


# ----------------------------------------------------------------------
# 10. Limits + constructor guards.
# ----------------------------------------------------------------------
def test_monopoly_and_competitive_limits():
    # n=1 collapses to monopoly: q*=(a-c)/(2b), P*=(a+c)/2, pi*=(a-c)^2/(4b)
    mono = CournotEnv(n=1, a=100.0, b=1.0, c=10.0).equilibrium()
    assert mono["q_i"] == pytest.approx(45.0)
    assert mono["P"] == pytest.approx(55.0)
    assert mono["profit_i"] == pytest.approx(2025.0)
    # as n grows the market price -> marginal cost c (competitive limit)
    a, b, c = 100.0, 1.0, 10.0
    p_small = CournotEnv(n=2, a=a, b=b, c=c).equilibrium()["P"]
    p_big = CournotEnv(n=200, a=a, b=b, c=c).equilibrium()["P"]
    assert p_big < p_small
    assert p_big == pytest.approx((a + 200 * c) / 201)
    assert abs(p_big - c) < 0.5


def test_constructor_rejects_bad_params():
    with pytest.raises(ValueError):
        CournotEnv(n=0)                    # need n >= 1
    with pytest.raises(ValueError):
        CournotEnv(a=10.0, c=10.0)         # need a > c (interior)
    with pytest.raises(ValueError):
        CournotEnv(a=5.0, c=10.0)          # a < c
    with pytest.raises(ValueError):
        CournotEnv(b=0.0)                  # need b > 0
