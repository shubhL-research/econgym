"""End-to-end INTEGRATION tests for the EconGym v1 suite.

Where the per-component suites (``test_env_*``, ``test_agent_*``, ``test_solvers``)
each validate one piece against a closed form, this file validates that the pieces
actually compose. It checks three cross-cutting contracts:

  1. **Public surface.** Every symbol advertised in ``econgym.__all__`` (and in the
     ``envs`` / ``agents`` / ``solvers`` sub-package ``__all__``s) is importable,
     resolves to a real object, and is reachable via ``from econgym import *``.

  2. **Interface generality of the shared runner.** For every *compatible*
     ``(agent, env)`` pair a short episode is driven through the ONE shared
     :func:`econgym.run_episode` and must return a well-formed :class:`Result`
     (right shape, finite rewards, in-range discrete actions, bit-reproducible).
     "Compatible" is used in the strict, semantically-valid sense: the pair runs
     through ``run_episode`` with the agent's ACTION INDEX equal to the env's
     NATIVE action, i.e. the discrete-action envs (Bertrand's price index,
     RepeatedPD's C/D index). See the module note below on why the index-based
     agents are *not* wired to the continuous (native-action) envs through the raw
     runner. The test also asserts the interface *correctly declines* the
     structurally-incompatible pairs (a bandit handed a continuous ``Box`` env, a
     Bertrand-only learner handed a game with no price grid, ...), which is the
     other half of "the interface generalizes": it fails loudly instead of
     silently producing garbage.

  3. **Env / solver agreement.** Every environment's ``equilibrium()`` must agree
     with the stand-alone solver it delegates to (the "single source of truth"
     invariant), re-checked here at the package level across the whole roster.

Note on ``run_episode`` and the continuous (native-action) envs
--------------------------------------------------------------
``run_episode`` passes each agent's ``act`` output straight into ``env.step`` and
records it into an ``int64`` trace. That is exactly right for the discrete-index
envs (Bertrand, RepeatedPD) whose native action IS the index. The continuous envs
(Cournot, differentiated Bertrand, public goods, the auctions) instead consume
*native* float actions, and the only index-based learner that fits them
(``FictitiousPlay``) returns a grid INDEX that must be mapped through
``agent.native(idx)`` before ``step`` -- which the runner does not do. So those
pairs are intentionally excluded from the ``run_episode`` matrix here (the
``FictitiousPlay`` suite drives them with its own native-mapping driver). This is
a documented generality boundary of the shared runner, not a defect in any single
component.
"""
import numpy as np
import pytest

import econgym
from econgym import (
    # runner + core
    run_episode, Result, EconEnv, Discrete, Box,
    # envs
    BertrandEnv, BertrandDiffEnv, CournotEnv, PublicGoodsEnv,
    SecondPriceEnv, RepeatedPDEnv, FirstPriceAuctionEnv, RubinsteinEnv,
    # agents
    QLearner, MeanBased, Thompson, UCB1, RegretMatching,
    FictitiousPlay, BestResponse,
    # solvers (value objects)
    cournot_nash, bertrand_diff_nash, first_price_bne, second_price_bne,
    public_goods_nash, repeated_pd_threshold, rubinstein_split,
)
from econgym import solvers as solvers_pkg


# ======================================================================
# 1. Public surface: every advertised symbol is importable and real.
# ======================================================================
def test_every_public_symbol_resolves():
    """Every name in ``econgym.__all__`` is a real, non-None attribute."""
    assert econgym.__all__, "econgym.__all__ is empty"
    unresolved = [name for name in econgym.__all__
                  if getattr(econgym, name, None) is None]
    assert unresolved == [], f"unresolved public symbols: {unresolved}"


def test_star_import_exposes_all():
    """``from econgym import *`` binds exactly the ``__all__`` surface."""
    ns = {}
    exec("from econgym import *", ns)
    missing = [name for name in econgym.__all__ if name not in ns]
    assert missing == [], f"names in __all__ missing from star-import: {missing}"


def test_all_has_no_duplicates():
    """``__all__`` is a clean list (a duplicate usually means a merge slip)."""
    dupes = sorted({n for n in econgym.__all__
                    if econgym.__all__.count(n) > 1})
    assert dupes == [], f"duplicate names in econgym.__all__: {dupes}"


def test_subpackage_surfaces_resolve():
    """The ``envs`` / ``agents`` / ``solvers`` sub-package public surfaces all
    resolve too (the package is coherent from the top down)."""
    for pkg in (econgym.envs, econgym.agents, econgym.solvers):
        unresolved = [n for n in pkg.__all__ if getattr(pkg, n, None) is None]
        assert unresolved == [], f"{pkg.__name__}: unresolved {unresolved}"


def test_key_symbols_have_expected_kinds():
    """Spot-check that the headline symbols are the right kind of object, so a
    star-import genuinely gives a usable env/agent/solver API."""
    for env_cls in (BertrandEnv, BertrandDiffEnv, CournotEnv, PublicGoodsEnv,
                    SecondPriceEnv, RepeatedPDEnv, FirstPriceAuctionEnv,
                    RubinsteinEnv):
        assert isinstance(env_cls, type) and issubclass(env_cls, EconEnv)
    for agent_cls in (QLearner, MeanBased, Thompson, UCB1, RegretMatching,
                      FictitiousPlay, BestResponse):
        assert isinstance(agent_cls, type)
    for solver_fn in (cournot_nash, bertrand_diff_nash, first_price_bne,
                      second_price_bne, public_goods_nash,
                      repeated_pd_threshold, rubinstein_split):
        assert callable(solver_fn)
    assert callable(run_episode)
    assert isinstance(Discrete, type) and isinstance(Box, type)


# ======================================================================
# 2. Interface generality: compatible (agent x env) pairs through run_episode.
# ======================================================================
# A pair is COMPATIBLE (semantically valid through the shared runner) iff the
# agent constructs from the env AND its action index equals the env's native
# action -- i.e. the discrete-action envs. Each entry: (label, env_factory,
# agents_factory). agents_factory(env) -> list[Agent] of length env.n.
_BERTRAND = lambda: BertrandEnv(n=2, K=7, c=1.0)
_PD = lambda: RepeatedPDEnv()

COMPAT = [
    # Bertrand (Discrete(K) price index == native action)
    ("bertrand+qlearner",  _BERTRAND, lambda e: [QLearner(e) for _ in range(e.n)]),
    ("bertrand+meanbased", _BERTRAND, lambda e: [MeanBased(e) for _ in range(e.n)]),
    ("bertrand+thompson",  _BERTRAND, lambda e: [Thompson(e) for _ in range(e.n)]),
    ("bertrand+ucb1",      _BERTRAND, lambda e: [UCB1(e, agent_id=i) for i in range(e.n)]),
    # Repeated PD (Discrete(2) C/D index == native action)
    ("pd+thompson",  _PD, lambda e: [Thompson(e) for _ in range(e.n)]),
    ("pd+ucb1",      _PD, lambda e: [UCB1(e, agent_id=i) for i in range(e.n)]),
    ("pd+regret",    _PD, lambda e: [RegretMatching(e, player_id=i) for i in range(e.n)]),
    ("pd+fp",        _PD, lambda e: [FictitiousPlay(e, player_id=i) for i in range(e.n)]),
    ("pd+bestresp",  _PD, lambda e: [BestResponse(e, player_id=i) for i in range(e.n)]),
]

_T = 40


@pytest.mark.parametrize("label,env_factory,agents_factory", COMPAT,
                         ids=[c[0] for c in COMPAT])
def test_run_episode_drives_compatible_pair(label, env_factory, agents_factory):
    """Each compatible pair runs through the SHARED run_episode and yields a
    well-formed Result: right shapes, finite rewards, in-range discrete actions."""
    env = env_factory()
    agents = agents_factory(env)
    assert len(agents) == env.n

    res = run_episode(env, agents, T=_T, seed=0)

    # a proper Result of the advertised shape
    assert isinstance(res, Result)
    assert res.n == env.n and res.T == _T and res.seed == 0
    assert res.prices.shape == (_T, env.n)
    assert res.profits.shape == (_T, env.n)
    # the generic alias mirrors the primary trace
    assert np.array_equal(res.actions, res.prices)
    # rewards are always finite
    assert np.isfinite(res.profits).all()
    # every recorded action is a valid index of that agent's Discrete action set
    for i in range(env.n):
        n_actions = env.action_space[i].n
        col = res.prices[:, i]
        assert col.min() >= 0 and col.max() < n_actions


@pytest.mark.parametrize("label,env_factory,agents_factory", COMPAT,
                         ids=[c[0] for c in COMPAT])
def test_run_episode_is_reproducible(label, env_factory, agents_factory):
    """Same seed -> bit-identical price and profit traces (the seeding contract
    holds through the whole env+agents+runner stack)."""
    e1, e2 = env_factory(), env_factory()
    r1 = run_episode(e1, agents_factory(e1), T=_T, seed=20260722)
    r2 = run_episode(e2, agents_factory(e2), T=_T, seed=20260722)
    assert np.array_equal(r1.prices, r2.prices)
    assert np.array_equal(r1.profits, r2.profits)


def test_run_episode_rejects_wrong_agent_count():
    """The runner enforces ``len(agents) == env.n`` (a wiring guardrail)."""
    env = _BERTRAND()
    with pytest.raises(ValueError):
        run_episode(env, [QLearner(env)], T=5, seed=0)   # only 1 agent for n=2


# --- the other half of generality: the interface DECLINES incompatible pairs ---
# Each: (label, env_factory, agent_builder) where agent_builder(env) must raise on
# construction (the agent cannot be served by that env's action interface).
INCOMPAT = [
    # QLearner / MeanBased need a discrete price grid (env.K); a continuous env
    # has none -> construction fails rather than mis-running.
    ("qlearner-on-cournot",  CournotEnv, lambda e: QLearner(e)),
    ("meanbased-on-cournot", CournotEnv, lambda e: MeanBased(e)),
    # Bandits need a Discrete action set; a Box action space is refused.
    ("thompson-on-cournot",  CournotEnv, lambda e: Thompson(e)),
    ("thompson-on-firstprice", FirstPriceAuctionEnv, lambda e: Thompson(e)),
    ("ucb1-on-publicgoods",  PublicGoodsEnv, lambda e: UCB1(e)),
    # RegretMatching needs a full-information counterfactual oracle; Bertrand
    # exposes none (and no explicit payoff given) -> refused.
    ("regret-on-bertrand",   _BERTRAND, lambda e: RegretMatching(e, player_id=0)),
    # FictitiousPlay needs a discrete action grid; the auctions expose none of the
    # recognised grid attributes -> refused.
    ("fp-on-secondprice",    SecondPriceEnv, lambda e: FictitiousPlay(e, player_id=0)),
]


@pytest.mark.parametrize("label,env_factory,agent_builder", INCOMPAT,
                         ids=[c[0] for c in INCOMPAT])
def test_interface_declines_incompatible_pairs(label, env_factory, agent_builder):
    """A structurally-incompatible (agent, env) pair fails LOUDLY at construction
    -- the interface never silently mis-runs an agent it cannot serve."""
    env = env_factory()
    with pytest.raises((TypeError, ValueError, AttributeError, IndexError)):
        agent_builder(env)


# ======================================================================
# 3. Env / solver agreement: equilibrium() == the matching solver.
# ======================================================================
def _num_agree(a, b):
    """Scalar/array-tolerant equality for equilibrium quantities."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (tuple, list, str)):
        return a == b
    return np.allclose(np.asarray(a, float), np.asarray(b, float))


def test_cournot_equilibrium_matches_solver():
    for (n, a, b, c) in [(2, 100.0, 1.0, 10.0), (3, 100.0, 1.0, 10.0),
                         (5, 120.0, 2.0, 10.0)]:
        env = CournotEnv(n=n, a=a, b=b, c=c)
        assert env.equilibrium() == cournot_nash(a, b, c, n)


def test_bertrand_diff_equilibrium_matches_solver():
    for (n, alpha, beta, gamma, c) in [(2, 10.0, 2.0, 1.0, 1.0),
                                       (3, 12.0, 3.0, 1.0, 2.0)]:
        env = BertrandDiffEnv(n=n, alpha=alpha, beta=beta, gamma=gamma, c=c)
        assert env.equilibrium() == bertrand_diff_nash(alpha, beta, gamma, c, n)


def test_public_goods_equilibrium_matches_solver():
    env = PublicGoodsEnv()
    assert env.equilibrium() == public_goods_nash(env.w, env.r, env.n)


def test_rubinstein_equilibrium_matches_solver():
    for delta in (0.5, 0.8, 0.9, 0.95):
        env = RubinsteinEnv(delta=delta)
        assert env.equilibrium() == rubinstein_split(delta)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_first_price_equilibrium_matches_solver(n):
    """First-price env equilibrium agrees with ``first_price_bne`` on the scalar
    quantities and the (callable) bid function at sample values."""
    env = FirstPriceAuctionEnv(n=n)
    eq, sol = env.equilibrium(), first_price_bne(n)
    assert _num_agree(eq["bid_slope"], sol["bid_slope"])
    assert _num_agree(eq["expected_revenue"], sol["expected_revenue"])
    vs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose([eq["bid_fn"](v) for v in vs], [sol["bid_fn"](v) for v in vs])


@pytest.mark.parametrize("n", [2, 3, 5])
def test_second_price_equilibrium_matches_solver(n):
    """Second-price env equilibrium agrees with ``second_price_bne`` (dominant
    truthful bidding: identity bid function, slope 1, revenue (n-1)/(n+1))."""
    env = SecondPriceEnv(n=n)
    eq, sol = env.equilibrium(), second_price_bne(n)
    assert _num_agree(eq["dominant"], sol["dominant"])
    assert _num_agree(eq["bid_slope"], sol["bid_slope"])
    assert _num_agree(eq["expected_revenue"], sol["expected_revenue"])
    vs = np.array([0.0, 0.3, 0.7, 1.0])
    assert np.allclose([eq["bid_fn"](v) for v in vs], vs)          # truthful b(v)=v


def test_repeated_pd_equilibrium_matches_solver():
    env = RepeatedPDEnv()
    cf = repeated_pd_threshold(env.T, env.R, env.P, env.S)
    eq = env.equilibrium()
    assert _num_agree(eq["grim_threshold"], cf["grim_threshold"])
    assert eq["one_shot_nash"] == cf["one_shot_nash"] == ("D", "D")


def test_bertrand_equilibrium_is_self_consistent():
    """Homogeneous Bertrand has no separate closed-form solver (its discrete-grid
    benchmark IS the source of truth), so we verify the env's reported dict is
    internally consistent: per-firm profit == (price - c)/n at both the Nash and
    monopoly benchmark prices."""
    env = BertrandEnv(n=2, K=21, c=1.0)
    eq = env.equilibrium()
    assert _num_agree(eq["nash_profit_per_firm"], (eq["nash_price"] - env.c) / env.n)
    assert _num_agree(eq["monopoly_profit_per_firm"],
                      (eq["monopoly_price"] - env.c) / env.n)
    assert eq["nash_price"] >= env.c        # competitive price never below cost


def test_solver_single_source_of_truth_identity():
    """The env modules and the ``solvers`` package expose the SAME solver objects
    (delegation, not copies), so env and solver can never drift apart."""
    assert first_price_bne is solvers_pkg.closed_form.first_price_bne
    assert cournot_nash is solvers_pkg.closed_form.cournot_nash
    assert rubinstein_split is solvers_pkg.closed_form.rubinstein_split
    assert econgym.closed_form is solvers_pkg.closed_form
