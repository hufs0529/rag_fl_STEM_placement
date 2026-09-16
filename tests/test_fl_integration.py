"""federated 루프의 배선 통합 테스트.

네트워크 없이 돌도록 작은 GPT-2 를 설정에서 직접 만들고(체크포인트 다운로드 없음),
토크나이저는 conftest 의 가짜 토크나이저를 쓴다. 확인하는 것은 정확도가 아니라
**경로**다: 예제 조립 -> 마스킹 -> K 스텝 로컬 학습 -> FedAvg -> drift/바이트 기록.

torch/peft 가 없는 환경에서는 통째로 skip 된다.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from src.aggregate import fedavg
from src.communication import (
    get_trainable_state_dict, numpy_to_state_dict,
    set_trainable_state_dict, state_dict_to_numpy,
)
from src.config import Config
from src.data import PassageStore, build_dataloader, build_example
from src.divergence import client_deltas, summarise
from src.fl_server import RoundRecorder
from src.local_train import LocalTrainer
from src.models import payload_bytes

pytestmark = pytest.mark.slow

TRAIN_CFG = {"learning_rate": 1e-3, "weight_decay": 0.0, "max_grad_norm": 1.0}


def _tiny_lora_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    config = GPT2Config(n_embd=32, n_layer=2, n_head=2, n_positions=128, vocab_size=30100)
    model = GPT2LMHeadModel(config)
    return get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["c_attn"],
                   bias="none", task_type="CAUSAL_LM"),
    )


def _client_examples(client_id, depth, n=8):
    store = PassageStore(
        [{"pid": f"c{client_id}_p{i}", "title": "T", "text": f"client {client_id} passage {i}"}
         for i in range(12)]
    )
    ids = [f"c{client_id}_p{i}" for i in range(12)]
    scores = {pid: -i for i, pid in enumerate(ids)}
    scores[ids[9]] = 50.0                      # 깊게 재채점해야만 올라오는 passage
    examples = []
    for q in range(n):
        question = {"qid": f"c{client_id}_q{q}", "question": f"question {q}?", "answers": ["yes"]}
        examples.append(build_example(question, ids, scores, store, depth, top_k=3))
    return examples


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [abs(hash(tok)) % 30000 + 10 for tok in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _FakeLogger:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def _round(model, clients, global_params, steps):
    client_params, metrics = [], []
    for trainer in clients:
        set_trainable_state_dict(model, numpy_to_state_dict(global_params))
        metrics.append(trainer.train_steps(steps))
        client_params.append(state_dict_to_numpy(get_trainable_state_dict(model)))
    return client_params, metrics


def test_two_rounds_of_fedavg_move_the_adapter_and_log_the_budget():
    torch.manual_seed(0)
    model = _tiny_lora_model()
    tokenizer = _FakeTokenizer()
    cfg = Config({"train": TRAIN_CFG})

    clients = [
        LocalTrainer(
            model,
            build_dataloader(_client_examples(cid, depth=50), tokenizer, batch_size=2, max_length=128),
            TRAIN_CFG,
        )
        for cid in range(2)
    ]

    global_params = state_dict_to_numpy(get_trainable_state_dict(model))
    initial = {k: v.copy() for k, v in global_params.items()}
    n_trainable = sum(v.size for v in initial.values())

    logger = _FakeLogger()
    recorder = RoundRecorder(logger, len(clients), payload_bytes(n_trainable), {})

    for rnd in (1, 2):
        client_params, metrics = _round(model, clients, global_params, steps=2)
        record = recorder.record(rnd, client_params, global_params, metrics)
        global_params = dict(fedavg(client_params))
        assert record["cumulative_bytes"] == rnd * 2 * payload_bytes(n_trainable) * 2
        assert np.isfinite(record["train_loss"])

    # 어댑터가 실제로 움직였고 drift 가 기록되었다
    moved = max(
        float(np.abs(global_params[k] - initial[k]).max()) for k in initial
    )
    assert moved > 0
    assert len(logger.records) == 2
    assert logger.records[-1]["drift_divergence"] >= 0


def test_optimiser_state_survives_across_rounds():
    # K 가 작을 때 매 라운드 AdamW 를 새로 만들면 warm-up 에 갇힌다 - 그래서
    # LocalTrainer 는 라운드를 넘어 살아 있어야 한다
    torch.manual_seed(0)
    model = _tiny_lora_model()
    trainer = LocalTrainer(
        model,
        build_dataloader(_client_examples(0, depth=0), _FakeTokenizer(), batch_size=2, max_length=128),
        TRAIN_CFG,
    )
    trainer.train_steps(3)
    assert trainer.steps_done == 3
    assert trainer.optimizer.state_dict()["state"]
    trainer.train_steps(2)
    assert trainer.steps_done == 5


def test_only_lora_parameters_are_communicated():
    model = _tiny_lora_model()
    state = get_trainable_state_dict(model)
    assert state
    assert all("lora" in key for key in state)


def test_identical_clients_produce_no_drift():
    torch.manual_seed(0)
    model = _tiny_lora_model()
    params = state_dict_to_numpy(get_trainable_state_dict(model))
    stats = summarise(client_deltas(params, [params, params]))
    assert stats["divergence"] == pytest.approx(0.0)
