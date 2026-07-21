"""Learning agents."""
from .base import Agent
from .qlearning import QLearner
from .meanbased import MeanBased
from .thompson import Thompson
from .ucb1 import UCB1
from .regret_matching import RegretMatching
from .fictitious_play import FictitiousPlay, BestResponse

__all__ = [
    "Agent", "QLearner", "MeanBased", "Thompson", "UCB1", "RegretMatching",
    "FictitiousPlay", "BestResponse",
]
