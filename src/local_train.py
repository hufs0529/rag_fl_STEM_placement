"""클라이언트 로컬 학습 K 스텝 (Subtask 1.2 (iii), 1.3).

Week 1 의 gate 3(K 캘리브레이션)과 Week 3 의 Flower 클라이언트가 **같은**
로컬 학습 루프를 쓴다. 게이트에서 고른 K 가 본실험의 K 와 다른 루프에서
측정된 것이면 캘리브레이션이 무의미해지기 때문이다.

optimiser state 는 라운드 간 유지된다: K 가 작을 때 매 라운드 AdamW 를 새로
만들면 모멘트가 계속 warm-up 상태에 머물러, K 의 효과와 옵티마이저 재시작의
효과가 섞인다 (계획서 Subtask 1.2).

The local training loop shared by gate 3 and the Week-3 Flower client, so the K
chosen by calibration is measured on the same loop the main runs use. Optimiser
state is carried across rounds so small K does not leave AdamW in warm-up.
"""

from itertools import cycle
from typing import Dict, Iterable, Iterator, List


class LocalTrainer:
    """한 클라이언트의 로컬 학습기. 라운드 간 optimiser state 와 데이터 위치를 유지한다."""

    def __init__(self, model, dataloader, train_cfg, device=None):
        import torch

        self.model = model
        self.device = device or next(model.parameters()).device
        self.cfg = train_cfg
        self.dataloader = dataloader
        self._iterator: Iterator = iter(cycle(dataloader))
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.steps_done = 0

    def state(self) -> dict:
        """라운드 간 이월되는 상태 (persist_optimizer_state=True 일 때)."""
        return {"optimizer": self.optimizer.state_dict(), "steps_done": self.steps_done}

    def load_state(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps_done = state.get("steps_done", 0)

    def train_steps(self, num_steps: int) -> Dict[str, float]:
        """정확히 num_steps(=K) 만큼 학습. 데이터가 모자라면 순환한다.

        **한 스텝 = grad_accumulation 개의 미니배치 + 갱신 1 회.** 즉 한 스텝이
        소비하는 예제 수는 batch_size x grad_accumulation 이고, K 의 의미는
        누적 설정과 무관하게 "갱신 횟수"로 고정된다.

        누적이 필요한 이유는 메모리다. CPU 에서는 bfloat16 학습이 불가능해
        float32 로 돌아가고, logits 텐서가 batch x seq x vocab x 4 바이트가 된다:
        batch 8 x seq 1024 x vocab 49,152 => 1.61 GB, 역전파까지 약 4.8 GB.
        실측으로 프로세스가 7.2 GB 까지 올라가 OOM 킬러에 죽었다. batch 2 와
        누적 4 는 **같은 유효 배치 8** 을 유지하면서 그 피크를 1.2 GB 로 낮춘다.
        """
        import torch

        accum = max(1, int(self.cfg.get("grad_accumulation", 1) or 1))
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        losses: List[float] = []
        for _ in range(num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum):
                batch = next(self._iterator)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                # 누적 분할: 미니배치 손실을 accum 으로 나눠 더하면, 기울기 합이
                # 유효 배치 하나를 그대로 돌린 것과 같아진다.
                loss = out.loss / accum
                loss.backward()
                step_loss += float(loss.detach())
                del out, loss, batch          # 다음 미니배치 전에 그래프를 놓아준다
            if self.cfg.get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(trainable, float(self.cfg["max_grad_norm"]))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(step_loss)
            self.steps_done += 1
        return {
            "loss": sum(losses) / len(losses) if losses else float("nan"),
            "steps": num_steps,
            "grad_accumulation": accum,
            "examples_per_step": accum * int(self.cfg.get("batch_size", 1) or 1),
            "steps_done": self.steps_done,
        }
