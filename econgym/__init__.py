"""econgym -- a Gymnasium-style suite of economics reinforcement-learning
environments.

**v0** ships the first environment: a homogeneous-good Bertrand market that
faithfully reproduces the RL-collusion dynamics of Paper 1 ("The Two Faces of
Algorithmic Collusion"). The design cleanly separates the three concepts fused
inside the original simulation loops so that further environments can be added
under ``econgym/envs/`` later:

  * environment (market physics)   -> :class:`BertrandEnv`  (``econgym.envs``)
  * agents (learning rules)        -> :class:`QLearner`, :class:`MeanBased`
  * runner + metrics (diagnostics) -> :func:`run_episode`, :mod:`metrics`

The science is preserved exactly: single-seed Q-learning traces are byte-for-byte
identical to the original; mean-based is reproduced at the aggregate level (one
intended, documented one-draw RNG offset).
"""
from . import metrics
from .config import BASE, H_STAR, H_STAR_DEFAULT, P_STAR_FACTOR
from .core import Box, Discrete, EconEnv
from .envs.bertrand import BertrandEnv
from .envs.bertrand_diff import BertrandDiffEnv
from .envs.cournot import CournotEnv, cournot_nash
from .envs.public_goods import PublicGoodsEnv, public_goods_nash
from .envs.second_price import SecondPriceEnv
from .envs.repeated_pd import RepeatedPDEnv
from .envs.first_price import FirstPriceAuctionEnv, first_price_bne
from .envs.rubinstein import RubinsteinEnv
from .agents.base import Agent
from .agents.qlearning import QLearner
from .agents.meanbased import MeanBased
from .agents.thompson import Thompson
from .agents.ucb1 import UCB1
from .agents.regret_matching import RegretMatching
from .agents.fictitious_play import FictitiousPlay, BestResponse
from .runner import Result, run_episode
from . import solvers
from .solvers import (
    best_response,
    closed_form,
    normal_form,
    best_response_iteration,
    bertrand_diff_nash,
    fictitious_play,
    repeated_pd_threshold,
    rubinstein_split,
    second_price_bne,
    support_enumeration,
)

__version__ = "0.2.0"

__all__ = [
    "EconEnv",
    "Discrete",
    "Box",
    "BertrandEnv",
    "BertrandDiffEnv",
    "CournotEnv",
    "cournot_nash",
    "PublicGoodsEnv",
    "public_goods_nash",
    "SecondPriceEnv",
    "RepeatedPDEnv",
    "FirstPriceAuctionEnv",
    "first_price_bne",
    "RubinsteinEnv",
    "QLearner",
    "MeanBased",
    "Thompson",
    "UCB1",
    "RegretMatching",
    "FictitiousPlay",
    "BestResponse",
    "Agent",
    "run_episode",
    "Result",
    "metrics",
    "BASE",
    "H_STAR",
    "H_STAR_DEFAULT",
    "P_STAR_FACTOR",
    # --- solvers (learned-vs-optimal backbone) ---
    "solvers",
    "closed_form",
    "normal_form",
    "best_response",
    "bertrand_diff_nash",
    "second_price_bne",
    "repeated_pd_threshold",
    "rubinstein_split",
    "support_enumeration",
    "best_response_iteration",
    "fictitious_play",
]
