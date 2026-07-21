"""Gym-style environment registry: ``econgym.make("Cournot-v0")``."""
import pytest

import econgym
from econgym import BertrandEnv, CournotEnv, list_envs, make


def test_make_returns_the_right_env_instance():
    assert isinstance(make("Cournot-v0"), CournotEnv)
    assert isinstance(make("Bertrand-v0"), BertrandEnv)


def test_make_passes_constructor_kwargs_through():
    env = make("Cournot-v0", n=5)
    assert env.n == 5


def test_make_unknown_id_raises_with_a_helpful_message():
    with pytest.raises(ValueError, match="NoSuchEnv-v0"):
        make("NoSuchEnv-v0")


def test_list_envs_returns_all_eight_ids_sorted():
    ids = list_envs()
    assert len(ids) == 8
    assert ids == sorted(ids)
    for expected in ("Bertrand-v0", "Cournot-v0", "FirstPrice-v0", "Rubinstein-v0"):
        assert expected in ids


def test_make_and_list_are_exported_at_top_level():
    assert hasattr(econgym, "make")
    assert hasattr(econgym, "list_envs")
