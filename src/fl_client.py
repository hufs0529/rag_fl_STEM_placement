"""Flower 클라이언트 (Subtask 1.2 / 1.3).

클라이언트는 전역 어댑터를 받아 자기 사설 데이터로 K 스텝 학습한 뒤 어댑터를
돌려준다. 통신되는 것은 LoRA A, B 뿐이다.

두 가지가 구조적으로 중요하다:

1) **optimiser state 는 라운드 간 유지된다.** Flower 는 라운드마다 클라이언트
   객체를 새로 만들 수 있으므로, 상태를 클라이언트 객체가 아니라 팩토리에
   보관한다. 그렇지 않으면 K 가 작을 때 AdamW 가 매 라운드 warm-up 으로
   되돌아가, K 의 효과와 옵티마이저 재시작 효과가 섞인다.

2) **로컬 학습 루프는 gate 3 와 동일한 코드(LocalTrainer)** 다. 게이트가 고른 K
   가 다른 루프에서 측정된 값이면 캘리브레이션이 무의미해진다.

Flower client. Only the LoRA A and B matrices are communicated. Optimiser state
lives in the factory, not the client object, because Flower may recreate the
client each round; and the local loop is the same code gate 3 calibrated on.
"""

from typing import Dict, List

from src.communication import (
    get_trainable_state_dict, ndarrays_to_state_dict,
    set_trainable_state_dict, state_dict_to_ndarrays,
)
from src.data import build_dataloader, selection_diagnostics
from src.local_train import LocalTrainer


class ClientState:
    """라운드를 넘어 살아남아야 하는 것들: 모델 핸들, 데이터로더, 옵티마이저."""

    def __init__(self, client_id: int, examples, tokenizer, model, cfg, device=None):
        self.client_id = client_id
        self.examples = examples
        self.model = model
        self.cfg = cfg
        self.diagnostics = selection_diagnostics(examples)
        self.dataloader = build_dataloader(
            examples,
            tokenizer,
            cfg.train.batch_size,
            cfg.model.max_seq_length,
            seed=1000 + client_id,
        )
        self.trainer = LocalTrainer(model, self.dataloader, cfg.train, device)

    def num_examples(self) -> int:
        return len(self.examples)


class RagFlowerClient:
    """flwr.client.NumPyClient 인터페이스. flwr 없이도 임포트되도록 상속은 런타임에."""

    def __init__(self, state: ClientState, keys: List[str], local_steps: int):
        self.state = state
        self.keys = keys
        self.local_steps = local_steps

    def get_parameters(self, config=None):
        return state_dict_to_ndarrays(get_trainable_state_dict(self.state.model))

    def fit(self, parameters, config=None):
        set_trainable_state_dict(
            self.state.model, ndarrays_to_state_dict(self.keys, parameters)
        )
        metrics = self.state.trainer.train_steps(self.local_steps)
        return (
            self.get_parameters(),
            self.state.num_examples(),
            {
                "loss": metrics["loss"],
                "client_id": self.state.client_id,
                "rank_change_rate": self.state.diagnostics["rank_change_rate"],
                "promotion_rate": self.state.diagnostics["promotion_rate"],
            },
        )

    def evaluate(self, parameters, config=None):
        # 평가는 서버에서 전역 모델로만 수행한다 (Subtask 1.3): 클라이언트별
        # 평가는 평가셋을 쪼개게 되어 조건 간 비교 가능성을 해친다.
        return 0.0, self.state.num_examples(), {}


def build_flower_client(state: ClientState, keys: List[str], local_steps: int):
    """flwr 의 NumPyClient 로 감싼다 (임포트는 이 함수 안에서)."""
    import flwr as fl

    class _Client(fl.client.NumPyClient, RagFlowerClient):
        pass

    client = _Client(state, keys, local_steps)
    return client.to_client() if hasattr(client, "to_client") else client
