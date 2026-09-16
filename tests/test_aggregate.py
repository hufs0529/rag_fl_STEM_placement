import numpy as np
import pytest

from src.aggregate import fedavg


def test_uniform_average_of_two_clients():
    out = fedavg([{"w": np.ones((2, 2))}, {"w": np.zeros((2, 2))}])
    assert np.allclose(out["w"], 0.5)


def test_weighted_average_respects_the_weights():
    out = fedavg([{"w": np.ones(3)}, {"w": np.zeros(3)}], weights=[3, 1])
    assert np.allclose(out["w"], 0.75)


def test_mismatched_keys_are_rejected_rather_than_silently_averaged():
    with pytest.raises(ValueError):
        fedavg([{"a": np.ones(2)}, {"b": np.ones(2)}])


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        fedavg([])
