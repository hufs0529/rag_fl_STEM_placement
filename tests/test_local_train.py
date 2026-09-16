"""grad_accumulation 이 '메모리만 줄이고 수학은 그대로'인지 검증한다.

이 등가성이 깨지면 batch 를 줄인 것이 실험 조건을 바꾼 것이 되므로, 게이트가
고른 K 와 본실험의 K 가 다른 세팅에서 측정된 셈이 된다.
"""
import pytest

torch = pytest.importorskip("torch")

from src.local_train import LocalTrainer


class TinyModel(torch.nn.Module):
    """입력 합에 선형 변환을 걸고 MSE 를 내는 최소 모델. loss 를 out.loss 로 노출."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(0.5)

    def forward(self, x, y):
        pred = self.lin(x).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        return type("Out", (), {"loss": loss})()


def make_batches(bs):
    x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.5, 0.5], [2.0, 2.0],
                      [1.0, 0.0], [0.0, 1.0], [1.5, 2.5], [2.5, 1.5]])
    y = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.8, 0.2, 2.2, 1.8])
    return [{"x": x[i:i + bs], "y": y[i:i + bs]} for i in range(0, len(x), bs)]


def run(bs, accum, lr=0.1):
    torch.manual_seed(0)
    model = TinyModel()
    cfg = {"learning_rate": lr, "weight_decay": 0.0, "batch_size": bs,
           "grad_accumulation": accum, "max_grad_norm": None}
    trainer = LocalTrainer(model, make_batches(bs), cfg, device="cpu")
    trainer.train_steps(1)
    return model.lin.weight.detach().clone()


def test_accumulation_reproduces_the_full_batch_update():
    """batch 8 x 누적 1 과 batch 2 x 누적 4 가 **같은 가중치**를 내야 한다."""
    full = run(bs=8, accum=1)
    split = run(bs=2, accum=4)
    torch.testing.assert_close(full, split, rtol=1e-5, atol=1e-6)


def test_accumulation_of_two_also_matches():
    torch.testing.assert_close(run(bs=8, accum=1), run(bs=4, accum=2), rtol=1e-5, atol=1e-6)


def test_a_step_is_one_update_regardless_of_accumulation():
    """K 의 의미는 '갱신 횟수'로 고정된다 - 누적을 늘려도 스텝 수는 그대로."""
    torch.manual_seed(0)
    model = TinyModel()
    cfg = {"learning_rate": 0.1, "weight_decay": 0.0, "batch_size": 2,
           "grad_accumulation": 4, "max_grad_norm": None}
    trainer = LocalTrainer(model, make_batches(2), cfg, device="cpu")
    out = trainer.train_steps(3)
    assert out["steps"] == 3 and trainer.steps_done == 3
    assert out["grad_accumulation"] == 4
    assert out["examples_per_step"] == 8          # 2 x 4


def test_missing_or_zero_accumulation_falls_back_to_one():
    for value in (None, 0, 1):
        torch.manual_seed(0)
        model = TinyModel()
        cfg = {"learning_rate": 0.1, "weight_decay": 0.0, "batch_size": 8,
               "grad_accumulation": value, "max_grad_norm": None}
        trainer = LocalTrainer(model, make_batches(8), cfg, device="cpu")
        assert trainer.train_steps(1)["grad_accumulation"] == 1
