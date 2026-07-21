"""Reproduce Paper 1's Table-1 (K=21 baseline) aggregate statistics through the
`econgym` v0 Bertrand environment, and assert every check in ``targets.json``.

This exercises the REAL econgym env + agents + runner + metrics (it never
imports the original ``simulation.py``). EVERY target, tolerance, and acceptance
bound is LOADED from ``targets.json`` (the single source of truth) -- none are
hard-coded a second time here. If a statistic drifts out of tolerance the fix is
the RNG order, never the threshold.

The two headline signs are asserted explicitly:
  * INTENSITY  (delta_index): Q-learning colludes HARDER -> delta z << 0.
  * FREQUENCY  (regime share): mean-based colludes MORE OFTEN -> share z >> 0.
"""
import json
import math
import os
import re

import numpy as np
import pytest

from econgym import BertrandEnv, QLearner, MeanBased, run_episode, metrics


# ----------------------------------------------------------------------
# Load the binding targets. ``targets.json`` lives at the repo ROOT and is
# resolved RELATIVE TO THE SOURCE TREE (``tests/../targets.json``); it is the
# single source of truth for the reproduction acceptance. It is NOT declared as
# package-data and is deliberately not shipped inside the wheel -- the reproduce
# suite always runs from the source checkout. An override path may be supplied
# via the ``ECONGYM_TARGETS`` environment variable.
# ----------------------------------------------------------------------
def _load_targets():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("ECONGYM_TARGETS"),
        os.path.join(here, os.pardir, "targets.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("targets.json not found in " + repr(candidates))


TARGETS = _load_targets()
CFG = TARGETS["config"]
TGT = TARGETS["targets"]
CHECKS = TARGETS["acceptance_summary"]["checks"]

K = CFG["K"]
T = CFG["T"]
N = CFG["n"]
ALPHA = CFG["alpha"]
GAMMA = CFG["gamma"]
EPS = CFG["epsilon"]
DECAY = CFG["eps_decay"]
C = CFG["c"]
PMIN = CFG["p_min"]
PMAX = CFG["p_max"]
T0 = CFG["T0"]
N_SEEDS = CFG["n_seeds"]
H_STAR = CFG["H_star"]
P_STAR_FACTOR = CFG["p_star_factor"]
SEEDS = list(range(N_SEEDS))


# ----------------------------------------------------------------------
# Acceptance bounds, read STRAIGHT out of targets.json -> acceptance_summary
# -> checks (never independently retyped here). This keeps targets.json the
# single source of truth for the z-thresholds and one-sided bounds.
# ----------------------------------------------------------------------
def _bound(lhs_token: str, op: str) -> float:
    """Numeric bound of an acceptance-check line, e.g.
    ``_bound('delta_z_mean_minus_q', '<=') -> -6.0``. Matches the check whose
    (stripped) text starts with ``lhs_token`` and contains ``op``, then returns
    the first number after ``op``."""
    for line in CHECKS:
        stripped = line.strip()
        if stripped.startswith(lhs_token) and op in stripped:
            after = stripped.split(op, 1)[1]
            m = re.search(r"-?\d+\.?\d*", after)
            if m:
                return float(m.group())
    raise KeyError(f"no acceptance check for '{lhs_token} {op} ...'")


# z-thresholds and one-sided bounds (loaded, not retyped)
DELTA_Z_MAX = _bound("delta_z_mean_minus_q", "<=")     # -6.0
SHARE_Z_MIN = _bound("share_z_mean_minus_q", ">=")     # 5.0
MEAN_ENTROPY_MAX = _bound("mean_entropy_mean", "<=")   # 0.15
Q_ENTROPY_MIN = _bound("q_entropy_mean", ">=")         # 0.35
MEAN_SHARE_MIN = _bound("mean_collusive_share", ">=")  # 0.80
Q_SHARE_MAX = _bound("q_collusive_share", "<=")        # 0.15

# numeric targets / tolerances (loaded from the targets sections)
INT = TGT["collusion_intensity_delta_index"]
CONV = TGT["convergence_fraction"]
TOL_DELTA = INT["tol_delta_mean"]                      # 0.05
TOL_CONV = CONV["tol_conv_frac"]                       # 0.08


# ----------------------------------------------------------------------
# Aggregate statistics helpers (identical formulas to the ground-truth harness).
# ----------------------------------------------------------------------
def _two_sample_z(a, b):
    """(a - b) independent two-sample (Welch) z. a=mean-based, b=q-learning."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    return (a.mean() - b.mean()) / se if se > 0 else math.inf


def _two_prop_z(ka, na, kb, nb):
    """(pa - pb) two-proportion z. a=mean-based, b=q-learning."""
    pa, pb = ka / na, kb / nb
    p = (ka + kb) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return (pa - pb) / se if se > 0 else math.inf


def _run_cell(kind):
    """Run all seeds for one learner kind; return per-seed arrays."""
    deltas, hnorms, convs, regimes = [], [], [], []
    for seed in SEEDS:
        env = BertrandEnv(n=N, K=K, c=C, p_min=PMIN, p_max=PMAX)
        if kind == "q":
            agents = [QLearner(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPS,
                               eps_decay=DECAY) for _ in range(N)]
        else:
            agents = [MeanBased(env, epsilon=EPS, eps_decay=DECAY)
                      for _ in range(N)]
        res = run_episode(env, agents, T, seed, track_conv=True)
        d = metrics.delta_index(res.prices, res.profits, env.grid, env.c,
                                env.n, T0)
        hn = metrics.mean_entropy(res.prices, env.K, T0)[1]
        p_bar = float(env.grid[res.prices[-T0:]].mean())
        deltas.append(d)
        hnorms.append(hn)
        convs.append(1 if res.converged else 0)
        regimes.append(metrics.regime(hn, p_bar, env.c, H_STAR, P_STAR_FACTOR))
    return dict(
        delta=np.array(deltas),
        hnorm=np.array(hnorms),
        conv=np.array(convs),
        regime=regimes,
        collusive=sum(r == "Collusive" for r in regimes),
    )


# ----------------------------------------------------------------------
# Compute the whole reproduction ONCE and share across assertions.
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def repro():
    q = _run_cell("q")
    m = _run_cell("mean")
    n = N_SEEDS
    return dict(
        q_delta_mean=float(q["delta"].mean()),
        mean_delta_mean=float(m["delta"].mean()),
        delta_z=_two_sample_z(m["delta"], q["delta"]),         # mean - q
        q_conv_frac=float(q["conv"].mean()),
        mean_conv_frac=float(m["conv"].mean()),
        q_entropy_mean=float(q["hnorm"].mean()),
        mean_entropy_mean=float(m["hnorm"].mean()),
        q_collusive_share=q["collusive"] / n,
        mean_collusive_share=m["collusive"] / n,
        share_z=_two_prop_z(m["collusive"], n, q["collusive"], n),
    )


# ----------------------------------------------------------------------
# Deterministic market benchmarks (exact, K=21 grid).
# ----------------------------------------------------------------------
def test_market_benchmarks():
    env = BertrandEnv(n=N, K=K, c=C, p_min=PMIN, p_max=PMAX)
    mb = TARGETS["market_benchmarks"]
    assert env.nash_profit() == mb["nash_profit_per_firm"] == 0.0
    assert env.monopoly_profit() == mb["monopoly_profit_per_firm"] == 4.5


# ----------------------------------------------------------------------
# The binding aggregate reproduction. Targets/tolerances come from targets.json.
# ----------------------------------------------------------------------
def test_collusion_intensity(repro):
    assert abs(repro["q_delta_mean"] - INT["q_delta_mean"]) <= TOL_DELTA
    assert abs(repro["mean_delta_mean"] - INT["mean_delta_mean"]) <= TOL_DELTA


def test_intensity_sign_and_z(repro):
    # Q colludes HARDER -> (mean - q) delta z strongly NEGATIVE.
    assert repro["delta_z"] <= DELTA_Z_MAX          # loaded from targets.json
    assert repro["mean_delta_mean"] < repro["q_delta_mean"]


def test_convergence(repro):
    assert abs(repro["q_conv_frac"] - CONV["q_conv_frac"]) <= TOL_CONV
    assert abs(repro["mean_conv_frac"] - CONV["mean_conv_frac"]) <= TOL_CONV


def test_entropy_separation(repro):
    # acceptance (loaded): mean_entropy_mean <= 0.15 AND q_entropy_mean >= 0.35
    assert repro["mean_entropy_mean"] <= MEAN_ENTROPY_MAX
    assert repro["q_entropy_mean"] >= Q_ENTROPY_MIN


def test_collusion_frequency(repro):
    # acceptance (loaded): mean_collusive_share >= 0.80 AND q_collusive_share <= 0.15
    assert repro["mean_collusive_share"] >= MEAN_SHARE_MIN
    assert repro["q_collusive_share"] <= Q_SHARE_MAX


def test_frequency_sign_and_z(repro):
    # Mean-based colludes MORE OFTEN -> share z strongly POSITIVE.
    assert repro["share_z"] >= SHARE_Z_MIN          # loaded from targets.json
    assert repro["mean_collusive_share"] > repro["q_collusive_share"]


def test_both_headline_faces_hold(repro):
    """The paper's 'two faces': mean colludes more OFTEN, Q colludes HARDER."""
    # frequency: mean > Q, z > 0
    assert repro["mean_collusive_share"] > repro["q_collusive_share"]
    assert repro["share_z"] > 0
    # intensity: Q > mean, z < 0
    assert repro["q_delta_mean"] > repro["mean_delta_mean"]
    assert repro["delta_z"] < 0


def test_full_acceptance_summary(repro):
    """Every check listed in targets.json -> acceptance_summary.checks. All
    targets, tolerances, and bounds are LOADED from targets.json (nothing here
    is retyped)."""
    checks = {
        "q_delta_mean": abs(repro["q_delta_mean"] - INT["q_delta_mean"]) <= TOL_DELTA,
        "mean_delta_mean": abs(repro["mean_delta_mean"] - INT["mean_delta_mean"]) <= TOL_DELTA,
        "delta_z<=-6": repro["delta_z"] <= DELTA_Z_MAX,
        "q_conv_frac": abs(repro["q_conv_frac"] - CONV["q_conv_frac"]) <= TOL_CONV,
        "mean_conv_frac": abs(repro["mean_conv_frac"] - CONV["mean_conv_frac"]) <= TOL_CONV,
        "mean_entropy<=0.15": repro["mean_entropy_mean"] <= MEAN_ENTROPY_MAX,
        "q_entropy>=0.35": repro["q_entropy_mean"] >= Q_ENTROPY_MIN,
        "mean_collusive>=0.80": repro["mean_collusive_share"] >= MEAN_SHARE_MIN,
        "q_collusive<=0.15": repro["q_collusive_share"] <= Q_SHARE_MAX,
        "share_z>=5": repro["share_z"] >= SHARE_Z_MIN,
    }
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, f"failed checks: {failed}  |  repro={repro}"
