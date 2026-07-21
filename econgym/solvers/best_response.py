"""Best-response dynamics solvers (agent-free, deterministic).

* ``best_response_iteration(br, x0, ...)`` -- continuous fixed-point iteration
  ``x <- br(x)`` for games whose best-response map is a contraction (e.g.
  Cournot, or differentiated Bertrand with ``gamma*(n-1)/(2*beta) < 1``); it
  converges to the unique Nash equilibrium from any starting point.
* ``fictitious_play(A, B, ...)`` -- discrete fictitious play for a 2-player
  normal-form game: each player myopically best-responds to the empirical
  frequency of the opponent's past play; the returned time-averaged strategies
  converge to a Nash equilibrium in games with the fictitious-play property
  (zero-sum, 2xn, potential/dominance-solvable games).
"""
from __future__ import annotations

import numpy as np


def best_response_iteration(br, x0, tol=1e-10, max_iter=10_000):
    """Iterate the best-response map ``br`` from ``x0`` to its fixed point.

    Parameters
    ----------
    br : callable
        Maps a joint strategy vector ``x`` (length ``n``) to the vector of
        best responses ``br(x)`` (length ``n``).
    x0 : array-like
        Starting joint strategy vector.
    tol : float
        Convergence tolerance on the max-norm of the update ``||x_{k+1} - x_k||_inf``.
    max_iter : int
        Maximum iterations before giving up.

    Returns
    -------
    np.ndarray
        The fixed point ``x*`` with ``br(x*) == x*`` (within ``tol``).

    Raises
    ------
    RuntimeError
        If the map does not converge within ``max_iter`` iterations.
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(max_iter):
        x_new = np.asarray(br(x), dtype=float)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        x = x_new
    raise RuntimeError(
        f"best_response_iteration did not converge within {max_iter} iterations "
        f"(last max-step={np.max(np.abs(x_new - x))})"
    )


def fictitious_play(A, B, iters: int = 20_000, x0: int = 0, y0: int = 0):
    """Discrete fictitious play on the 2-player normal-form game ``(A, B)``.

    Each round both players best-respond to the *empirical* mixed strategy of the
    opponent (the running frequency of the opponent's past pure actions), and the
    counts are updated. The time-averaged play converges to a Nash equilibrium in
    every game with the fictitious-play property -- zero-sum games, ``2 x n``
    games, potential games, and (as here) dominance-solvable games such as the
    Prisoner's Dilemma.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Row player's payoff matrix (``A[i, j]`` = row payoff at profile ``(i, j)``).
    B : array-like, shape (m, n)
        Column player's payoff matrix.
    iters : int
        Number of fictitious-play rounds.
    x0, y0 : int
        Initial pure actions used to seed the empirical beliefs (the counts start
        as a single observation of each, so beliefs are well defined from round
        one). Deterministic: no RNG is consumed.

    Returns
    -------
    (x_avg, y_avg) : tuple[np.ndarray, np.ndarray]
        Time-averaged (empirical) mixed strategies of the row and column players,
        each a probability vector (lengths ``m`` and ``n``).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError(
            f"A and B must be 2-D payoff matrices of equal shape; got "
            f"{A.shape} and {B.shape}."
        )
    m, n = A.shape

    # Empirical action counts, seeded with one observation each so the first
    # best response is well defined.
    row_counts = np.zeros(m, dtype=float)
    col_counts = np.zeros(n, dtype=float)
    row_counts[int(x0)] += 1.0
    col_counts[int(y0)] += 1.0

    for _ in range(int(iters)):
        # Each player best-responds to the opponent's empirical distribution.
        opp_col = col_counts / col_counts.sum()          # belief about column
        opp_row = row_counts / row_counts.sum()          # belief about row
        br_row = int(np.argmax(A @ opp_col))             # row's best response
        br_col = int(np.argmax(opp_row @ B))             # column's best response
        row_counts[br_row] += 1.0
        col_counts[br_col] += 1.0

    x_avg = row_counts / row_counts.sum()
    y_avg = col_counts / col_counts.sum()
    return x_avg, y_avg
