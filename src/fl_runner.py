"""한 run(depth x seed)을 끝까지 실행한다 (Subtask 1.3).

고정 통신 예산, early stopping 없음. 모든 조건이 동일한 라운드 수 R 을 소모하고,
"그 예산이 무엇을 사주는가"로 비교된다. 조기 종료를 두면 결론이 정지 규칙에
의존하게 되고, LoRA 어댑터는 몇 라운드 만에 수렴하므로 라운드 수는 이 규모에서
너무 거친 자가 된다 - 그래서 곡선을 남기고 Week 5 에서 보간한다.

백엔드 두 가지:
  - flower    : 계획서대로 Flower simulation mode (기본)
  - inprocess : 동일한 수식을 순차 루프로. 리허설/디버깅/저사양 환경용이며
                집계·기록·측정 경로는 flower 백엔드와 같은 코드를 쓴다.

Runs one (depth x seed) to completion on a fixed communication budget with no
early stopping. Two backends share the same aggregation and logging code.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from src.communication import (
    get_trainable_state_dict, numpy_to_state_dict,
    set_trainable_state_dict, state_dict_to_numpy,
)
from src.evaluate import build_eval_prompts, evaluate_model
from src.experiment import ExperimentData, RunSpec, split_train_eval
from src.fl_client import ClientState
from src.fl_server import RoundRecorder, aggregate_round
from src.logging_utils import JsonlLogger, save_json
from src.models import build_model, count_trainable, describe_payload, payload_bytes
from src.schedule import schedule_from_config


FLOWER_INCOMPATIBLE = """backend="flower" cannot run this design.

Ray-based Flower simulation puts each client in a separate process and
serialises the objects it sends there. This design deliberately does the
opposite, for two reasons recorded in the config:

  * one 135M model instance shared by all eight clients (eight copies would
    make memory, not science, the limit on scale);
  * optimiser state held in ClientState in **this** process, so small K does not
    leave AdamW permanently in warm-up and confound K with optimiser restarts.

A Ray actor cannot reach this process's state, and three things block
serialisation outright:
  LocalTrainer's cycling iterator   (TypeError: cannot pickle 'itertools.cycle')
  build_dataloader's lambda collate_fn
  the ClientState dict itself

Use backend="inprocess". It is not a downgrade of the measurement: both
backends call the same aggregate_round() and the same RoundRecorder, and with
full participation (participation 1.0) visiting clients sequentially gives the
same aggregate as visiting them in parallel. The backend is recorded in every
run's metadata.

Making "flower" work would mean shipping model and optimiser state to actors
every round - more memory and wall-clock for no change in what is measured."""


def check_backend(backend: str) -> None:
    """flower 를 고르면 **시작 전에** 거부한다.

    그냥 돌리면 Ray 가 140 줄짜리 직렬화 트레이스백을 내는데, 거기엔 왜 안 되는지가
    없다. 라운드 0 평가(1,000 질문 생성)를 마친 뒤에 죽는 것도 낭비다.
    """
    if backend == "flower":
        raise RuntimeError(FLOWER_INCOMPATIBLE)
    if backend != "inprocess":
        raise ValueError(f"unknown backend {backend!r}; choose 'inprocess'")


def run_path(cfg, spec: RunSpec) -> Path:
    return Path(cfg.paths.log_dir) / "runs" / f"{spec.name}.jsonl"


def setup_run(cfg, spec: RunSpec, device: str = None, log=print):
    """모델, 클라이언트 데이터, 평가 프롬프트를 준비한다.

    모델 인스턴스는 **하나**만 만들고 클라이언트마다 파라미터를 갈아끼운다.
    8개 클라이언트가 각자 135M 모델을 들고 있을 이유가 없고(어댑터만 다르다),
    시뮬레이션에서 메모리가 병목이 되면 실험 규모가 제약되기 때문이다.
    """
    data = ExperimentData(cfg, spec.partition_seed)
    tokenizer, model = build_model(cfg, device)
    n_trainable, n_total = count_trainable(model)
    log(f"  LoRA: {n_trainable:,} trainable / {n_total:,} total "
        f"({payload_bytes(n_trainable) / 1e6:.2f} MB per client per direction)")

    held_out = data.held_out_qids()
    clients: Dict[int, ClientState] = {}
    for cid in range(data.num_clients):
        examples = split_train_eval(data.training_examples(cid, spec.depth), held_out)
        clients[cid] = ClientState(cid, examples, tokenizer, model, cfg, device)
        log(f"  client {cid}: {len(examples):,} training examples "
            f"(rank-change {clients[cid].diagnostics['rank_change_rate']:.3f}, "
            f"promotion {clients[cid].diagnostics['promotion_rate']:.3f})")

    eval_prompts = build_eval_prompts(
        data.eval_questions(), data.caches, data.store,
        eval_depth=cfg.eval.eval_depth, top_k=cfg.retrieval.top_k,
    )
    log(f"  held-out evaluation set: {len(eval_prompts):,} questions "
        f"(context fixed at d={cfg.eval.eval_depth} for every condition)")
    return data, tokenizer, model, clients, eval_prompts, n_trainable


def run_experiment(
    cfg,
    spec: RunSpec,
    backend: str = "flower",
    device: str = None,
    log=print,
) -> dict:
    # 백엔드를 **가장 먼저** 검사한다. 모델 적재와 라운드 0 평가(1,000 질문 생성)를
    # 마친 뒤에 죽으면 그 시간이 버려진다.
    check_backend(backend)

    data, tokenizer, model, clients, eval_prompts, n_trainable = setup_run(cfg, spec, device, log)

    schedule = schedule_from_config(cfg, spec.rounds)
    payload = payload_bytes(n_trainable, cfg.model.payload_dtype)
    path = run_path(cfg, spec)

    meta = {
        "run": spec.name,
        "depth": spec.depth,
        "seed": spec.seed,
        "partition_seed": spec.partition_seed,
        "rounds": spec.rounds,
        "local_steps_per_round": spec.local_steps_per_round,
        "total_local_steps": spec.total_local_steps,
        "num_clients": data.num_clients,
        "eval_rounds": schedule,
        "early_stopping": False,
        "backend": backend,
        **describe_payload(n_trainable, data.num_clients, spec.rounds, cfg.model.payload_dtype),
    }

    def evaluate_global(params: Dict) -> Dict[str, float]:
        set_trainable_state_dict(model, numpy_to_state_dict(params))
        return evaluate_model(
            model, tokenizer, eval_prompts, data.store,
            max_new_tokens=cfg.model.max_new_tokens,
            max_length=cfg.model.max_seq_length,
            top_k=cfg.retrieval.top_k,
        )

    with JsonlLogger(path, meta) as logger:
        recorder = RoundRecorder(logger, data.num_clients, payload, meta)
        initial = state_dict_to_numpy(get_trainable_state_dict(model))

        # 학습 전 기저값 - 곡선의 좌단이자 "예산 0 에서 무엇을 사는가"
        baseline = evaluate_global(initial)
        logger.write({"event": "round", "round": 0, "cumulative_bytes": 0,
                      "evaluated": True, **baseline})
        log(f"  round 0 (untrained): F1 {baseline['f1']:.2f} EM {baseline['em']:.2f}")

        final_params = _run_inprocess(
            cfg, spec, model, clients, initial, schedule, evaluate_global, recorder, log
        )

        final = evaluate_global(final_params)
        summary = {
            **meta,
            "final": final,
            "completed": True,
            "wall_clock_seconds": time.time() - recorder.started,
        }
        logger.write({"event": "summary", **summary})

    save_json(summary, Path(cfg.paths.log_dir) / "runs" / f"{spec.name}_summary.json")
    log(f"  done: F1 {final['f1']:.2f} EM {final['em']:.2f} "
        f"| {meta['total_mb']:.1f} MB communicated over {spec.rounds} rounds")
    return summary


def _run_inprocess(cfg, spec, model, clients, initial, schedule, evaluate_global, recorder, log):
    """순차 루프 백엔드. 집계·기록은 flower 백엔드와 동일한 함수를 쓴다."""
    global_params = {k: v.copy() for k, v in initial.items()}

    for rnd in range(1, spec.rounds + 1):
        client_params, client_metrics = [], []
        for cid in sorted(clients):
            set_trainable_state_dict(model, numpy_to_state_dict(global_params))
            metrics = clients[cid].trainer.train_steps(spec.local_steps_per_round)
            client_params.append(state_dict_to_numpy(get_trainable_state_dict(model)))
            client_metrics.append({**metrics, **clients[cid].diagnostics, "client_id": cid})

        aggregated = aggregate_round(client_params)
        eval_metrics = evaluate_global(aggregated) if rnd in schedule else None
        record = recorder.record(rnd, client_params, global_params, client_metrics, eval_metrics)
        global_params = aggregated
        _log_round(log, rnd, spec.rounds, record)

    return global_params


def _run_flower(cfg, spec, model, clients, initial, schedule, evaluate_global, recorder, log):
    """Flower simulation mode. 라운드 기록을 위해 FedAvg 를 감싼 전략을 쓴다."""
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from src.fl_client import build_flower_client

    keys = list(initial)
    state_holder = {"global": {k: v.copy() for k, v in initial.items()}}

    class _RecordingFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            client_params = [
                dict(zip(keys, parameters_to_ndarrays(res.parameters))) for _, res in results
            ]
            client_metrics = [dict(res.metrics) for _, res in results]
            aggregated = aggregate_round(client_params)
            eval_metrics = evaluate_global(aggregated) if server_round in schedule else None
            record = recorder.record(
                server_round, client_params, state_holder["global"], client_metrics, eval_metrics
            )
            state_holder["global"] = aggregated
            _log_round(log, server_round, spec.rounds, record)
            return ndarrays_to_parameters([aggregated[k] for k in keys]), {}

    def client_fn(cid: str):
        # 클라이언트 상태(옵티마이저·데이터 위치)는 clients 딕셔너리에 남아 있어
        # Flower 가 객체를 다시 만들어도 라운드 간 이월된다.
        return build_flower_client(clients[int(cid)], keys, spec.local_steps_per_round)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=spec.rounds),
        strategy=_RecordingFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=len(clients),
            min_available_clients=len(clients),
            initial_parameters=ndarrays_to_parameters([initial[k] for k in keys]),
        ),
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    return state_holder["global"]


def _log_round(log, rnd: int, total: int, record: dict) -> None:
    line = (f"  round {rnd:>3}/{total}  loss {record.get('train_loss', float('nan')):.4f}  "
            f"drift {record.get('drift_divergence', float('nan')):.4f}  "
            f"{record['cumulative_bytes'] / 1e6:7.1f} MB")
    if record.get("evaluated"):
        line += f"  | F1 {record['f1']:.2f}  EM {record['em']:.2f}"
    log(line)
