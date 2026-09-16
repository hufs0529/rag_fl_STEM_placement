"""스냅샷이 모델에서 분리되는지 검증한다.

이 등가성이 깨지면 divergence 가 **언제나 0** 으로 나온다. 여러 클라이언트의
파라미터를 모아 비교할 때 모든 항목이 마지막 학습 상태를 가리키기 때문이다.
실측으로 게이트 (iii) 가 cos +1.000 / divergence 0.0000 을 냈다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.communication import (
    get_trainable_state_dict, state_dict_to_ndarrays, state_dict_to_numpy,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def test_state_dict_snapshot_does_not_follow_later_training():
    model = Tiny()
    before = get_trainable_state_dict(model)
    with torch.no_grad():
        model.w.add_(1.0)                       # "학습" 1 스텝
    torch.testing.assert_close(before["w"], torch.zeros(3))


def test_numpy_snapshot_does_not_follow_later_training():
    model = Tiny()
    snap = state_dict_to_numpy(get_trainable_state_dict(model))
    with torch.no_grad():
        model.w.add_(2.0)
    np.testing.assert_allclose(snap["w"], np.zeros(3))


def test_ndarrays_snapshot_does_not_follow_later_training():
    model = Tiny()
    arrays = state_dict_to_ndarrays(get_trainable_state_dict(model))
    with torch.no_grad():
        model.w.add_(3.0)
    np.testing.assert_allclose(arrays[0], np.zeros(3))


def test_two_clients_sharing_one_model_keep_distinct_snapshots():
    """게이트 (iii) 의 실제 사용 방식: 모델 하나를 돌려쓰며 클라이언트별로 찍는다."""
    model = Tiny()
    snapshots = []
    for delta in (1.0, 5.0):
        with torch.no_grad():
            model.w.fill_(0.0)                  # global 파라미터로 되돌림
            model.w.add_(delta)                 # 그 클라이언트의 로컬 학습
        snapshots.append(state_dict_to_numpy(get_trainable_state_dict(model)))
    np.testing.assert_allclose(snapshots[0]["w"], np.full(3, 1.0))
    np.testing.assert_allclose(snapshots[1]["w"], np.full(3, 5.0))
    assert not np.allclose(snapshots[0]["w"], snapshots[1]["w"])


def test_frozen_parameters_are_not_communicated():
    model = Tiny()
    assert list(get_trainable_state_dict(model)) == ["w"]
