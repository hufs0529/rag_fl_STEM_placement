"""FedAvg 서버 + 라운드별 측정 기록 (Subtask 1.3).

서버가 매 라운드 기록하는 것들(계획서 "Also recorded"):
  - 누적 통신 바이트 (R x N x |AB| x 2)
  - client-update divergence (기전 확인용)
  - rank-change rate / promotion rate (처리가 실제로 걸렸는지)
  - 예약된 라운드에서의 F1 / EM (곡선)
  - wall-clock

평가는 **전역 모델에 대해 중앙에서만** 수행한다.

FedAvg server with per-round measurement logging. Evaluation happens centrally
on the global model only.
"""

import time
from typing import Callable, Dict, List, Optional

from src.aggregate import fedavg
from src.communication import cumulative_bytes
from src.divergence import client_deltas, summarise


class RoundRecorder:
    """라운드 기록을 만들어 JSONL 로 넘긴다. 분석은 이 기록만 읽는다."""

    def __init__(self, logger, num_clients: int, payload_bytes: int, run_meta: Dict):
        self.logger = logger
        self.num_clients = num_clients
        self.payload_bytes = payload_bytes
        self.run_meta = run_meta
        self.started = time.time()

    def record(
        self,
        round_number: int,
        client_params: List[Dict],
        global_params: Dict,
        client_metrics: List[Dict],
        eval_metrics: Optional[Dict] = None,
    ) -> Dict:
        drift = summarise(client_deltas(global_params, client_params))
        record = {
            "event": "round",
            "round": round_number,
            "cumulative_bytes": cumulative_bytes(
                round_number, self.num_clients, self.payload_bytes
            ),
            "payload_bytes_per_client_per_direction": self.payload_bytes,
            "train_loss": _mean(m.get("loss") for m in client_metrics),
            "rank_change_rate": _mean(m.get("rank_change_rate") for m in client_metrics),
            "promotion_rate": _mean(m.get("promotion_rate") for m in client_metrics),
            "wall_clock_seconds": time.time() - self.started,
            **{f"drift_{k}": v for k, v in drift.items()},
        }
        if eval_metrics:
            record.update({
                "f1": eval_metrics["f1"],
                "em": eval_metrics["em"],
                "eval_gold_recall_at_3": eval_metrics.get("eval_gold_recall_at_3"),
                "evaluated": True,
            })
        else:
            record["evaluated"] = False
        self.logger.write(record)
        return record


def aggregate_round(client_params: List[Dict], weights: Optional[List[float]] = None) -> Dict:
    """FedAvg. 파티션이 크기 균등이므로 기본은 균등 가중."""
    return dict(fedavg(client_params, weights))


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_flower_strategy(
    initial_parameters,
    num_clients: int,
    evaluate_fn: Callable,
    fit_metrics_fn: Callable = None,
):
    """Flower FedAvg 전략. 전 클라이언트 참여, 서버 측 중앙 평가."""
    import flwr as fl

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,          # 평가는 서버에서만
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_fn or _aggregate_fit_metrics,
    )


def _aggregate_fit_metrics(results):
    """클라이언트가 올린 진단 지표를 예제 수 가중 평균."""
    total = sum(n for n, _ in results) or 1
    keys = {k for _, m in results for k in m if isinstance(m.get(k), (int, float))}
    return {
        k: sum(n * m.get(k, 0.0) for n, m in results) / total
        for k in keys
        if k != "client_id"
    }
