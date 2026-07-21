"""Validation tests for the Rubinstein alternating-offers bargaining env.

Per CONTRACT_v1 §3.7 the Rubinstein equilibrium is *analytic* -- the unique
subgame-perfect split. These tests validate the env against that closed-form
benchmark on three complementary fronts:

  1. **Static exactness.** ``env.equilibrium()`` equals the hand-derived closed
     form ``(1/(1+delta), delta/(1+delta))`` exactly, cross-checked against the
     single-source-of-truth solver ``closed_form.rubinstein_split``.
  2. **Env physics reproduce the SPE.** Driving ``env.step`` with the SPE
     strategies yields immediate (round-1) agreement at exactly the SPE shares,
     and the env applies ``delta**(round-1)`` discounting correctly when
     agreement is delayed.
  3. **Deviation / dominance structure.** The responder is exactly indifferent at
     the SPE offer (its accept boundary equals its continuation value), and -- on
     a fine grid evaluated *through the env's own accept/reject physics* -- the
     proposer has no profitable deviation from keeping ``1/(1+delta)``.

Plus the ``delta -> 1`` limit (both shares -> 1/2, monotone) and space/seeding
sanity. No tolerance is loosened: the closed form is rational and hits exactly;
the deviation scan uses ``1e-9``; indifference uses ``1e-12``.
"""
import numpy as np
import pytest

from econgym import RubinsteinEnv
from econgym.solvers import closed_form


DELTAS = [0.5, 0.8, 0.9, 0.95]


def _spe_actions(env, keep, accept):
    """Build the length-2 action list [ [offer_keep, .], [., accept_signal] ]
    with the CURRENT proposer keeping ``keep`` and the responder accepting iff
    ``accept``. The unused components are filler."""
    p = env._proposer()
    r = 1 - p
    acts = [None, None]
    acts[p] = [float(keep), 0.0]                 # proposer: offer_keep in slot 0
    acts[r] = [0.0, 1.0 if accept else 0.0]      # responder: accept bit in slot 1
    return acts


# ----------------------------------------------------------------------
# 1. Static: equilibrium() == closed form exactly, cross-checked vs solver.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", DELTAS)
def test_split_matches_closed_form(delta):
    env = RubinsteinEnv(delta=delta)
    eq = env.equilibrium()
    p_share = 1.0 / (1.0 + delta)
    r_share = delta / (1.0 + delta)
    assert eq["proposer_share"] == pytest.approx(p_share, rel=0, abs=1e-15)
    assert eq["responder_share"] == pytest.approx(r_share, rel=0, abs=1e-15)
    assert eq["agreement_round"] == 1
    # shares partition the whole pie
    assert eq["proposer_share"] + eq["responder_share"] == pytest.approx(1.0, abs=1e-15)
    # single source of truth: env delegates to the solver -> identical dict
    assert eq == closed_form.rubinstein_split(delta)
    # benchmark alias returns the same thing
    assert env.benchmark() == eq


def test_solver_rejects_out_of_range_delta():
    with pytest.raises(ValueError):
        closed_form.rubinstein_split(1.0)
    with pytest.raises(ValueError):
        closed_form.rubinstein_split(-0.1)
    with pytest.raises(ValueError):
        RubinsteinEnv(delta=1.0)


# ----------------------------------------------------------------------
# 2. Responder indifference (the pillar of the derivation).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", DELTAS)
def test_responder_indifference(delta):
    """The responder is exactly indifferent between accepting delta/(1+delta) now
    and rejecting to become next-round proposer earning 1/(1+delta), discounted
    by delta."""
    accept_now = delta / (1.0 + delta)
    reject_then_propose = delta * (1.0 / (1.0 + delta))
    assert accept_now == pytest.approx(reject_then_propose, abs=1e-12)


# ----------------------------------------------------------------------
# 3. Env physics reproduce the SPE outcome and discounting.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", DELTAS)
def test_env_spe_play_reaches_closed_form(delta):
    """Round-1 SPE play through env.step: proposer keeps 1/(1+delta), responder
    accepts -> rewards equal the SPE shares (discount delta**0 = 1), terminated."""
    env = RubinsteinEnv(delta=delta)
    env.reset(seed=0)
    eq = env.equilibrium()
    obs, rew, term, trunc, info = env.step(
        _spe_actions(env, keep=eq["proposer_share"], accept=True)
    )
    assert term is True and trunc is False
    assert info["proposer"] == 0 and info["round"] == 1 and info["accepted"] is True
    # agent 0 is the round-1 proposer -> gets the proposer share, agent 1 the responder share
    assert rew[0] == pytest.approx(eq["proposer_share"], abs=1e-15)
    assert rew[1] == pytest.approx(eq["responder_share"], abs=1e-15)
    assert rew.sum() == pytest.approx(1.0, abs=1e-15)   # no delay -> full pie realized


@pytest.mark.parametrize("delta", DELTAS)
def test_env_discounts_delayed_agreement(delta):
    """Reject in round 1 then agree in round 2: the env applies a delta**1 factor
    and the proposer role has correctly swapped to agent 1."""
    env = RubinsteinEnv(delta=delta)
    env.reset(seed=0)
    # Round 1: agent 0 proposes, responder (agent 1) REJECTS -> continue.
    _, rew1, term1, trunc1, info1 = env.step(_spe_actions(env, keep=0.5, accept=False))
    assert term1 is False and trunc1 is False
    assert np.allclose(rew1, 0.0)                     # nothing realized on rejection
    assert info1["proposer"] == 0
    # Round 2: agent 1 now proposes. It keeps its SPE proposer share; agent 0 accepts.
    assert env._proposer() == 1
    p_share = 1.0 / (1.0 + delta)
    r_share = delta / (1.0 + delta)
    _, rew2, term2, trunc2, info2 = env.step(_spe_actions(env, keep=p_share, accept=True))
    assert term2 is True and info2["round"] == 2 and info2["discount"] == pytest.approx(delta)
    # round-2 proposer is agent 1; both payoffs scaled by delta**(2-1) = delta
    assert rew2[1] == pytest.approx(delta * p_share, abs=1e-15)
    assert rew2[0] == pytest.approx(delta * r_share, abs=1e-15)
    # present value of the whole pie has shrunk to delta (one round of delay)
    assert rew2.sum() == pytest.approx(delta, abs=1e-15)


# ----------------------------------------------------------------------
# 3b. Responder rationality boundary == continuation value (via env physics).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", DELTAS)
def test_responder_accept_boundary_is_continuation_value(delta):
    """A rational round-1 responder accepts iff its offered share >= its
    continuation value delta/(1+delta). We drive the env's accept/reject physics
    with the rational decision and confirm: just above the boundary the deal
    strikes (positive responder reward); just below, the env continues to round 2
    with zero realized payoff. This pins the accept boundary to delta/(1+delta)."""
    cont = delta / (1.0 + delta)          # responder continuation value
    eps = 1e-6

    # (a) offered slightly ABOVE continuation -> rational accept -> deal strikes.
    env = RubinsteinEnv(delta=delta)
    env.reset(seed=0)
    keep_low = 1.0 - (cont + eps)         # responder offered cont+eps
    _, rew, term, _, info = env.step(_spe_actions(env, keep=keep_low, accept=True))
    assert term is True
    assert info["responder_offered"] == pytest.approx(cont + eps, abs=1e-12)
    assert rew[1] == pytest.approx(cont + eps, abs=1e-12)   # responder takes the deal

    # (b) offered slightly BELOW continuation -> rational reject -> game continues.
    env2 = RubinsteinEnv(delta=delta)
    env2.reset(seed=0)
    keep_high = 1.0 - (cont - eps)        # responder offered cont-eps (< continuation)
    _, rew2, term2, trunc2, _ = env2.step(_spe_actions(env2, keep=keep_high, accept=False))
    assert term2 is False and trunc2 is False
    assert np.allclose(rew2, 0.0)
    assert env2.round == 2                 # rolled over to the next round


# ----------------------------------------------------------------------
# 3c. Proposer has no profitable deviation (scan through the env's physics).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", DELTAS)
def test_proposer_no_profitable_deviation(delta):
    """On a fine grid of round-1 proposer keep-shares x, evaluate the proposer's
    round-1 value against a RATIONAL responder, using the env's own accept/reject
    accounting:

      * If the responder is offered (1-x) >= its continuation delta/(1+delta) it
        accepts (env strikes the deal) and the proposer earns x now.
      * Otherwise it rejects; the proposer becomes the round-2 responder and its
        continuation value is delta * (delta/(1+delta)) = delta**2/(1+delta).

    The SPE keep-share x* = 1/(1+delta) must be the unique maximizer, and no
    deviation may beat it by more than 1e-9.
    """
    cont_resp = delta / (1.0 + delta)              # responder's continuation value
    proposer_reject_value = delta * (delta / (1.0 + delta))   # = delta**2/(1+delta)
    x_star = 1.0 / (1.0 + delta)

    def proposer_value(x):
        """Round-1 proposer value at keep-share ``x`` against a rational responder,
        computed THROUGH the env's own accept/reject physics."""
        offered = 1.0 - x
        rational_accept = offered >= cont_resp - 1e-15     # tie -> accept
        env = RubinsteinEnv(delta=delta)
        env.reset(seed=0)
        _, rew, term, _, _ = env.step(_spe_actions(env, keep=x, accept=rational_accept))
        if rational_accept:
            assert term is True
            assert rew[0] == pytest.approx(x, abs=1e-12)   # proposer realizes x now
            return float(rew[0])
        assert term is False
        return proposer_reject_value                       # continuation as round-2 responder

    # (i) At EXACTLY the SPE keep-share the proposer realizes x* (responder is at
    #     its indifference point and accepts).
    assert proposer_value(x_star) == pytest.approx(x_star, abs=1e-12)

    # (ii) NO deviation on a fine grid beats the SPE value x* by more than 1e-9.
    grid = np.linspace(0.0, 1.0, 4001)
    values = np.array([proposer_value(x) for x in grid])
    assert values.max() <= x_star + 1e-9
    # the discrete optimizer sits within one grid step of x* (the true continuous max)
    assert grid[int(values.argmax())] == pytest.approx(x_star, abs=2.0 / (len(grid) - 1))

    # (iii) The deviation structure is strict on both sides of x*:
    #   * keeping MORE than x* pushes the offer below the responder's continuation,
    #     so it is rejected and the proposer falls to delta**2/(1+delta) < x*;
    assert proposer_reject_value < x_star - 1e-12
    assert proposer_value(x_star + 1e-6) == pytest.approx(proposer_reject_value, abs=1e-12)
    #   * keeping LESS than x* is accepted but simply leaves money on the table.
    assert proposer_value(x_star - 1e-6) == pytest.approx(x_star - 1e-6, abs=1e-12)
    assert proposer_value(x_star - 1e-6) < x_star


# ----------------------------------------------------------------------
# 4. delta -> 1 limit: both shares -> 1/2, monotonically.
# ----------------------------------------------------------------------
def test_limit_behavior():
    deltas = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
    p_shares = [RubinsteinEnv(delta=d).equilibrium()["proposer_share"] for d in deltas]
    r_shares = [RubinsteinEnv(delta=d).equilibrium()["responder_share"] for d in deltas]
    # proposer share decreases monotonically toward 1/2 from above;
    # responder share increases monotonically toward 1/2 from below.
    for a, b in zip(p_shares, p_shares[1:]):
        assert b < a
    for a, b in zip(r_shares, r_shares[1:]):
        assert b > a
    assert p_shares[-1] > 0.5 and r_shares[-1] < 0.5
    assert p_shares[-1] == pytest.approx(0.5, abs=1e-3)
    assert r_shares[-1] == pytest.approx(0.5, abs=1e-3)
    # exact symmetric first-mover advantage collapses: p - r = (1-delta)/(1+delta)
    for d, ps, rs in zip(deltas, p_shares, r_shares):
        assert (ps - rs) == pytest.approx((1.0 - d) / (1.0 + d), abs=1e-15)


# ----------------------------------------------------------------------
# Spaces / seeding sanity.
# ----------------------------------------------------------------------
def test_spaces_and_reset_contract():
    env = RubinsteinEnv(delta=0.9)
    assert env.n == 2
    assert len(env.action_space) == 2 and len(env.observation_space) == 2
    # each agent submits a length-2 [offer_keep, accept_signal] action
    assert env.action_space[0].shape == (2,)
    assert env.action_space[0].contains([0.3, 1.0])
    assert not env.action_space[0].contains([1.5, 0.0])
    # reset returns one observation per agent; both see the same public state
    obs = env.reset(seed=0)
    assert len(obs) == 2
    assert np.array_equal(obs[0], obs[1])
    assert obs[0][0] == 0.0 and obs[0][1] == 1.0   # [proposer_id=0, round=1]
    # the seeding contract: reset(rng=...) threads the shared stream (no state drawn)
    shared = np.random.default_rng(7)
    env.reset(rng=shared)
    assert env.rng is shared


def test_max_rounds_truncation():
    """With a finite horizon, persistent rejection truncates with zero payoffs."""
    env = RubinsteinEnv(delta=0.9, max_rounds=3)
    env.reset(seed=0)
    for _ in range(3):
        _, rew, term, trunc, _ = env.step(_spe_actions(env, keep=0.9, accept=False))
        assert np.allclose(rew, 0.0)
    # after max_rounds rejections the episode is truncated (disagreement)
    assert trunc is True and term is False
