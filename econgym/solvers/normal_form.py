"""General 2-player normal-form Nash equilibria via support enumeration.

``support_enumeration(A, B)`` returns every Nash equilibrium of a finite
two-player game with row-player payoff matrix ``A`` and column-player payoff
matrix ``B`` (both shape ``(m, n)``: ``A[i, j]`` / ``B[i, j]`` is the payoff to
the row / column player when the row plays pure action ``i`` and the column plays
pure action ``j``).

Algorithm (the classical support-enumeration method)
----------------------------------------------------
An equilibrium is a mixed profile ``(x, y)`` (probability vectors of length ``m``
/ ``n``) in which every action played with positive probability is a best
response. Two facts drive the enumeration:

1. *Equal support size.* In a nondegenerate game the two players' supports have
   the same cardinality ``k`` (the indifference / best-response conditions form a
   square system). So we enumerate, for each ``k = 1 .. min(m, n)``, every pair of
   supports ``(I, J)`` with ``|I| = |J| = k``.

2. *Indifference on the support.* If the row player mixes over exactly ``I`` then,
   in equilibrium, it must be indifferent among those rows against the column
   strategy ``y``; that pins ``y`` (supported on ``J``) via the linear system

       sum_{j in J} A[i, j] * y_j = u   for all i in I,   sum_{j in J} y_j = 1,

   and symmetrically ``x`` (supported on ``I``) from the column player's
   indifference over ``J`` using ``B``. Each is a ``(k+1) x (k+1)`` linear solve.

A candidate survives iff (a) the solved probabilities are strictly positive on
their declared support (so the equilibrium's support is exactly ``(I, J)``), and
(b) it is *best-response consistent*: no pure action off the support earns more
than the on-support payoff. Surviving profiles are de-duplicated.

This recovers all pure and mixed equilibria of nondegenerate games (and the
isolated equilibria of degenerate ones). Textbook checks: the Prisoner's Dilemma
returns the unique ``(D, D)``; matching pennies returns the unique
``((1/2, 1/2), (1/2, 1/2))``; a coordination game returns its two pure and one
mixed equilibrium.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = ["support_enumeration"]


def _solve_indifference(payoff, support_own, support_opp):
    """Opponent mixed strategy (on ``support_opp``) that makes the *own* player
    indifferent across ``support_own``.

    Solves the ``(k+1) x (k+1)`` system

        sum_{j in support_opp} payoff[i, j] * y_j = u   for i in support_own,
        sum_{j in support_opp} y_j = 1,

    for the ``k`` probabilities ``y_j`` (and the common payoff ``u``). ``payoff``
    is the *own* player's payoff matrix, indexed so that ``payoff[i, j]`` is the
    own payoff when the own player uses row-index ``i`` and the opponent uses
    column-index ``j``.

    Returns
    -------
    (probs, u) : (np.ndarray, float) or (None, None)
        ``probs`` has length ``len(support_opp)`` (the mixed strategy restricted
        to the opponent's support), ``u`` the induced own payoff. Returns
        ``(None, None)`` if the linear system is singular.
    """
    k = len(support_own)
    # Unknowns: [y over support_opp (k values), u]. Rows: k indifference eqns
    # (coeff of u is -1) + 1 normalization eqn.
    M = np.zeros((k + 1, k + 1), dtype=float)
    rhs = np.zeros(k + 1, dtype=float)
    for r, i in enumerate(support_own):
        for cidx, j in enumerate(support_opp):
            M[r, cidx] = payoff[i, j]
        M[r, k] = -1.0
    M[k, :k] = 1.0          # sum of probabilities ...
    rhs[k] = 1.0            # ... equals 1
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None, None
    return sol[:k], float(sol[k])


def support_enumeration(A, B, tol: float = 1e-9):
    """Return all Nash equilibria of the 2-player game ``(A, B)``.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Row player's payoff matrix (``A[i, j]`` = row payoff at profile ``(i, j)``).
    B : array-like, shape (m, n)
        Column player's payoff matrix (``B[i, j]`` = column payoff at ``(i, j)``).
    tol : float
        Numerical tolerance: probabilities on a support must exceed ``tol`` to be
        counted, and a pure action off the support may exceed the on-support
        payoff by at most ``tol`` before the candidate is rejected as a
        best-response violation.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Each element is ``(sigma_row, sigma_col)`` -- full-length probability
        vectors (lengths ``m`` and ``n``). Sorted deterministically; every
        returned profile is a genuine Nash equilibrium (no profitable unilateral
        pure deviation, within ``tol``).

    Notes
    -----
    Support enumeration finds all equilibria of nondegenerate games and the
    isolated equilibria of degenerate ones; a degenerate game with a *continuum*
    of equilibria will have that continuum represented only by its extreme
    supports.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
        raise ValueError(
            f"A and B must be 2-D payoff matrices of equal shape; got "
            f"{A.shape} and {B.shape}."
        )
    m, n = A.shape
    equilibria = []
    seen = set()

    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                # y (on J) makes the ROW player indifferent across I, using A.
                y_sup, u = _solve_indifference(A, I, J)
                if y_sup is None or np.any(y_sup <= tol):
                    continue
                # x (on I) makes the COLUMN player indifferent across J. The
                # column player's payoff for using column j against row-mix x is
                # sum_i x_i B[i, j]; that is the own-indifference system for the
                # column player with own-index = j, opponent-index = i, i.e. use
                # B transposed so that B.T[j, i] = B[i, j].
                x_sup, w = _solve_indifference(B.T, J, I)
                if x_sup is None or np.any(x_sup <= tol):
                    continue

                x = np.zeros(m)
                y = np.zeros(n)
                for idx, i in enumerate(I):
                    x[i] = x_sup[idx]
                for idx, j in enumerate(J):
                    y[j] = y_sup[idx]

                # Best-response consistency: no off-support pure action beats the
                # on-support payoff. Row payoffs against y: A @ y; column payoffs
                # against x: x @ B.
                row_payoffs = A @ y
                col_payoffs = x @ B
                if row_payoffs.max() > u + 1e-7:
                    continue
                if col_payoffs.max() > w + 1e-7:
                    continue

                key = (tuple(np.round(x, 9)), tuple(np.round(y, 9)))
                if key in seen:
                    continue
                seen.add(key)
                equilibria.append((x, y))

    equilibria.sort(key=lambda xy: (xy[0].tolist(), xy[1].tolist()))
    return equilibria
