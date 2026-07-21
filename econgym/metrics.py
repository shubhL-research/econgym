"""Diagnostics for Bertrand RL-collusion episodes.

Moved *verbatim* (semantics-preserving) from Paper-1's `simulation.py`:

  * ``entropy_bits``  -- K-normalised Shannon entropy (bits) with optional
    Miller-Madow bias correction;
  * ``mean_entropy``  -- mean (raw, normalised) entropy over the final T0 window;
  * ``delta_index``   -- collusion index in [0, 1] (discrete Bertrand-Nash = 0,
    joint monopoly = 1);
  * ``is_converged``  -- non-circular convergence (greedy-policy stability over
    the final 10%), NOT inferred from entropy;
  * ``regime``        -- Chaotic / Competitive / Collusive classifier.

These functions are pure; unit tests check them against the original.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Static / discrete market benchmarks (grid-based; verbatim formulas).
# Duplicated here (independent of the env) so `delta_index` needs only the grid.
# ----------------------------------------------------------------------
def nash_profit(grid, c, n):
    """Per-firm profit at the DISCRETE competitive (Bertrand-Nash) benchmark.

    The static competitive equilibrium sits at the lowest grid price weakly
    above marginal cost; that price earns a small positive per-firm profit
    (split by ``n``). Returns 0 only if a grid point equals ``c``.
    """
    above = grid[grid >= c]
    p_comp = above.min() if above.size else grid.max()
    return (p_comp - c) / n


def monopoly_profit(grid, c, n):
    """Per-firm profit at symmetric joint monopoly (top price, split by n)."""
    return (grid.max() - c) / n


# ----------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------
def entropy_bits(col, K, miller_madow=True) -> tuple[float, float]:
    """Shannon entropy (bits) of an action-index sequence, with optional
    Miller-Madow bias correction.

    Returns ``(raw_bits, K_normalised)`` where the normaliser is ``log2(K)``,
    so the second element lies in [0, 1].
    """
    counts = np.bincount(col, minlength=K).astype(float)
    nobs = counts.sum()
    f = counts[counts > 0] / nobs
    H = float(-(f * np.log2(f)).sum())
    if miller_madow:
        # bias correction (in bits): (nonzero_bins - 1) / (2 * nobs) / ln(2)
        H += (np.count_nonzero(counts) - 1) / (2.0 * nobs) / np.log(2)
    return H, H / np.log2(K)


def mean_entropy(price_hist, K, T0, miller_madow=True) -> tuple[float, float]:
    """Mean (raw, K-normalised) entropy over the final ``T0`` window, averaged
    across the ``n`` firm columns."""
    w = price_hist[-T0:]
    vals = [entropy_bits(w[:, i], K, miller_madow) for i in range(w.shape[1])]
    Hbar = float(np.mean([v[0] for v in vals]))
    Hnorm = float(np.mean([v[1] for v in vals]))
    return Hbar, Hnorm


# ----------------------------------------------------------------------
# Collusion index
# ----------------------------------------------------------------------
def delta_index(price_hist, profit_hist, grid, c, n, T0) -> float:
    """Collusion index normalised between the DISCRETE Bertrand-Nash (0) and
    symmetric joint monopoly (1).

    ``pi`` is mean per-firm profit over the final ``T0`` periods.
    """
    pi = float(profit_hist[-T0:].mean())
    piN = nash_profit(grid, c, n)
    piM = monopoly_profit(grid, c, n)
    return (pi - piN) / (piM - piN)


# ----------------------------------------------------------------------
# Convergence + regime
# ----------------------------------------------------------------------
def is_converged(res, tol=0.99) -> bool:
    """Non-circular convergence: greedy policy unchanged for >= ``tol`` of the
    visited states over the final 10% of the run, for EVERY firm.

    Accepts a ``Result`` (uses ``.pol_stable``), a dict with key ``'pol_stable'``
    (matching the original ``simulation.py`` output), or a raw list of per-agent
    stability fractions.
    """
    ps = getattr(res, "pol_stable", None)
    if ps is None:
        if isinstance(res, dict):
            ps = res.get("pol_stable")
        else:
            ps = res
    return bool(ps) and min(ps) >= tol


def regime(Hnorm, p_bar, c, H_star, p_star_factor=2.0) -> str:
    """'Chaotic' if ``Hnorm >= H_star``; else 'Competitive' if
    ``p_bar < p_star_factor * c`` else 'Collusive'."""
    if Hnorm >= H_star:
        return "Chaotic"
    return "Competitive" if p_bar < p_star_factor * c else "Collusive"
