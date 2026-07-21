"""Solvers package -- pure, agent-free, deterministic equilibrium routines.

These operate on *model parameters / payoff callables*, never on ``Agent``
objects, and are the primary oracle in the environment validation tests. Each
environment's ``equilibrium()`` delegates to the matching ``closed_form.*``
function so the env and the solver can never disagree (single source of truth).

Public surface
--------------
* :mod:`~econgym.solvers.closed_form` -- analytic Nash / BNE / SPE dicts:
  ``cournot_nash``, ``bertrand_diff_nash``, ``first_price_bne``,
  ``second_price_bne``, ``public_goods_nash``, ``repeated_pd_threshold``,
  ``rubinstein_split``.
* :mod:`~econgym.solvers.normal_form` -- ``support_enumeration(A, B)``: all Nash
  equilibria of a 2-player normal-form game via support enumeration.
* :mod:`~econgym.solvers.best_response` -- ``best_response_iteration`` (continuous
  fixed-point iteration) and ``fictitious_play`` (discrete belief dynamics).
"""
from . import best_response, closed_form, normal_form
from .best_response import best_response_iteration, fictitious_play
from .closed_form import (
    bertrand_diff_nash,
    cournot_nash,
    first_price_bne,
    public_goods_nash,
    repeated_pd_threshold,
    rubinstein_split,
    second_price_bne,
)
from .normal_form import support_enumeration

__all__ = [
    # modules
    "closed_form",
    "normal_form",
    "best_response",
    # closed-form equilibria
    "cournot_nash",
    "bertrand_diff_nash",
    "first_price_bne",
    "second_price_bne",
    "public_goods_nash",
    "repeated_pd_threshold",
    "rubinstein_split",
    # normal-form solver
    "support_enumeration",
    # best-response dynamics
    "best_response_iteration",
    "fictitious_play",
]
