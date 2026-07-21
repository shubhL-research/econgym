"""A tiny Gym-style environment registry.

    import econgym
    env = econgym.make("Cournot-v0", n=3)

``make(env_id, **kwargs)`` constructs the environment registered under
``env_id``, forwarding ``kwargs`` to its constructor. ``list_envs()`` returns
the sorted list of available ids.
"""
from __future__ import annotations

from .envs.bertrand import BertrandEnv
from .envs.bertrand_diff import BertrandDiffEnv
from .envs.cournot import CournotEnv
from .envs.first_price import FirstPriceAuctionEnv
from .envs.public_goods import PublicGoodsEnv
from .envs.repeated_pd import RepeatedPDEnv
from .envs.rubinstein import RubinsteinEnv
from .envs.second_price import SecondPriceEnv

_REGISTRY = {
    "Bertrand-v0": BertrandEnv,
    "BertrandDiff-v0": BertrandDiffEnv,
    "Cournot-v0": CournotEnv,
    "FirstPrice-v0": FirstPriceAuctionEnv,
    "PublicGoods-v0": PublicGoodsEnv,
    "RepeatedPD-v0": RepeatedPDEnv,
    "Rubinstein-v0": RubinsteinEnv,
    "SecondPrice-v0": SecondPriceEnv,
}


def make(env_id: str, **kwargs):
    """Construct the environment registered under ``env_id``.

    Keyword arguments are forwarded to the environment constructor, so
    ``make("Cournot-v0", n=5)`` is exactly ``CournotEnv(n=5)``.
    """
    try:
        cls = _REGISTRY[env_id]
    except KeyError:
        raise ValueError(
            f"unknown environment id {env_id!r}; available ids: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def list_envs() -> list:
    """Return the sorted list of registered environment ids."""
    return sorted(_REGISTRY)


__all__ = ["make", "list_envs"]
