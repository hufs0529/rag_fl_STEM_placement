#!/usr/bin/env python3
"""Week 1 gate (iii): local-step calibration - K 를 정하고 그로부터 R 을 도출.

라운드 수는 자유 파라미터가 아니다: R = S / K 이고, K(라운드당 로컬 스텝)는
클라이언트가 평균되기 전에 얼마나 멀리 흘러가는지(drift)를 좌우한다. 이 실험의
가설이 "retrieval 품질이 drift 를 줄여 라운드를 줄인다"이므로, **drift 가 관측
가능한 최소 K** 를 골라야 한다. drift 가 없는 곳에서는 drift 를 줄이는 처리를
관측할 수 없다.

절차: K in {4,8,16,32,64} 각각에 대해 짧게 몇 라운드를 돌려 client-update
divergence 를 재고, 임계값을 넘는 가장 작은 K 를 택한다. 그 다음 S 를
클라이언트 데이터 1~2 epoch 으로 두고 R = S/K 를 확정해 설정에 기록한다.

Gate (iii): sweep K, measure client-update divergence and select the smallest K
at which drift is measurable; S then fixes R = S / K.

Usage:
  python scripts/gate3_local_step_calibration.py --dev
  python scripts/gate3_local_step_calibration.py --set gates.local_steps.probe_rounds=3
"""

import argparse
import random
import sys
from collections import defaultdict
from typing import Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.data import PassageStore, build_dataloader, build_example
from src.divergence import client_deltas, summarise
from src.gates import local_step_decision, rounds_from_budget, total_steps_from_passes
from src.local_train import LocalTrainer
from src.logging_utils import banner, load_json, save_json
from src.models import build_model
from src.nq_data import load_questions


def provisional_partition(questions, store, num_clients, seed=42):
    """Week 2 의 topic-cluster 파티션이 나오기 전 임시 분할.

    gold passage 의 title 을 주제 대리변수로 삼아, title 단위로 묶은 뒤
    가장 작은 클라이언트에 차례로 배정한다(그리디 균형화). 의도는 Week 2 와
    같다 - 클라이언트가 서로 겹치지 않는 주제 범위를 갖되 크기는 균등하게.
    K 캘리브레이션에 필요한 것은 "drift 가 생기는 구조"이지 최종 파티션 자체는
    아니므로 이 대리 분할로 충분하다.

    A provisional stand-in for the Week-2 topic-cluster partition: group by gold
    passage title, then greedily fill the smallest client. What gate 3 needs is a
    structure that produces drift, not the final partition.
    """
    by_title = defaultdict(list)
    for q in questions:
        gold = next((p for p in q["gold_passage_ids"] if p in store), None)
        title = store.get(gold)["title"] if gold else ""
        by_title[title].append(q)

    groups = sorted(by_title.values(), key=len, reverse=True)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    clients = [[] for _ in range(num_clients)]
    for group in groups:
        target = min(range(num_clients), key=lambda i: len(clients[i]))
        clients[target].extend(group)
    return clients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--cache", default="results/cache/gate_scratch.json",
                        help="gate 2 가 남긴 스크래치 retrieval 캐시")
    parser.add_argument("--passages", default=None,
                        help="쉼표로 여러 파일을 줄 수 있다. 기본값은 gold + random "
                             "두 파일 - 게이트 ② 의 프로브 코퍼스가 둘에서 나온다")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-questions-per-client", type=int, default=256)
    parser.add_argument("--memory-headroom", type=float, default=0.75,
                        help="추정 피크가 가용 메모리의 이 비율을 넘으면 거부한다")
    args = parser.parse_args()
    cfg = config_from_args(args)

    import numpy as np

    from src.aggregate import fedavg
    from src.communication import (
        get_trainable_state_dict, numpy_to_state_dict,
        set_trainable_state_dict, state_dict_to_numpy,
    )

    data_dir = Path(cfg.paths.data_dir)
    num_clients = cfg.partition.num_clients
    candidate_Ks = list(cfg.gates.local_steps.candidate_K)
    probe_rounds = cfg.gates.local_steps.probe_rounds

    banner(f"Week 1 gate (iii) - local-step calibration | K sweep {candidate_Ks}")

    questions = load_questions(args.questions or data_dir / "questions_usable.jsonl")
    cache = load_json(args.cache)["entries"]
    questions_all = questions          # 캐시로 걸러내기 전 전체 (본실험 규모 계산용)
    questions = [q for q in questions if q["qid"] in cache]
    print(f"  {len(questions):,} questions with cached retrieval")

    # 게이트 ② 의 프로브 코퍼스는 **두 파일**에서 나온다: gold (표본 질문의 것과
    # 다른 질문의 것) 와 정답 미포함 채움재. gold 만 읽으면 채움재 후보를 조회할 때
    # KeyError 가 난다. 반대로 두 파일을 통째로 올리면 93 만 권에 3 GB 가 넘는데,
    # 여기서는 모델까지 함께 올려야 한다. 그래서 캐시에 등장하는 pid 만 남긴다.
    if args.passages:
        passage_paths = [p.strip() for p in str(args.passages).split(",") if p.strip()]
    else:
        passage_paths = [data_dir / "gold_passages.jsonl", data_dir / "random_passages.jsonl"]
    needed = {pid for q in questions for pid in q["gold_passage_ids"]}
    needed.update(pid for qid in (q["qid"] for q in questions)
                  for pid in cache[qid]["candidates"])
    store = PassageStore.from_jsonl_many(passage_paths, keep=needed)
    print(f"  passage store {len(store):,} of {len(needed):,} needed ids "
          f"from {len(passage_paths)} file(s)")

    # 후보에 있는 pid 가 하나라도 없으면 학습 중간에 KeyError 로 죽는다. 먼저 막는다.
    candidate_ids = [pid for q in questions for pid in cache[q["qid"]]["candidates"]]
    absent = store.missing(candidate_ids)
    if absent:
        print(f"\n  {len(absent):,} candidate passages are in the cache but in none of\n"
              f"  {', '.join(str(p) for p in passage_paths)}\n"
              f"  (first few: {', '.join(absent[:5])})\n"
              "  The scratch cache was written against gate (ii)'s probe corpus, which\n"
              "  draws from both the gold file and the answer-free reservoir. Pass every\n"
              "  file that corpus came from via --passages, or re-run gate (ii) to\n"
              "  rewrite the cache.")
        return 1

    clients = provisional_partition(questions, store, num_clients)
    clients = [c[: args.max_questions_per_client] for c in clients]
    print(f"  provisional partition: " + ", ".join(str(len(c)) for c in clients))

    # 캘리브레이션은 baseline 조건(d=0)에서 한다: 처리가 없는 상태에서 drift 가
    # 관측되는 K 를 골라야 처리 효과를 담을 수 있다.
    baseline_depth = 0
    client_examples = [
        [
            build_example(q, cache[q["qid"]]["candidates"], cache[q["qid"]]["scores"],
                          store, baseline_depth, cfg.retrieval.top_k)
            for q in group
        ]
        for group in clients
    ]

    tokenizer, model = build_model(cfg, args.device)
    initial = state_dict_to_numpy(get_trainable_state_dict(model))
    keys = list(initial)
    print(f"  trainable tensors: {len(keys)} | "
          f"{sum(v.size for v in initial.values()):,} parameters")

    # 학습을 시작하기 전에 피크 메모리를 추정해 본다. 실측으로 batch 8 / seq 1024 /
    # float32 에서 프로세스가 7.2 GB 까지 올라가 OOM 킬러에 죽었고, 그때 WSL2
    # 배포판 전체가 불안정해져 편집기 연결까지 끊겼다. 2,976 스텝을 돌리다
    # 죽는 것보다 지금 거부하는 편이 낫다. 모델 적재 자체는 안전하므로
    # (가중치 약 540 MB) 실제 파라미터 수를 센 뒤에 판정한다.
    from src.models import estimate_training_peak_bytes, format_bytes, memory_guard

    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_value = next(model.parameters()).element_size()
    accum = int(cfg.train.get("grad_accumulation", 1) or 1)
    estimate = estimate_training_peak_bytes(
        vocab_size=int(model.config.vocab_size),
        batch_size=cfg.train.batch_size,
        seq_length=cfg.model.max_seq_length,
        n_params=n_params,
        bytes_per_value=bytes_per_value,
        hidden_size=int(model.config.hidden_size),
        n_layers=int(model.config.num_hidden_layers),
    )
    guard = memory_guard(estimate, headroom=args.memory_headroom)
    print(f"  memory: batch {cfg.train.batch_size} x accum {accum} "
          f"(effective {cfg.train.batch_size * accum}) x seq {cfg.model.max_seq_length} "
          f"x {bytes_per_value} bytes")
    print(f"    logits {format_bytes(estimate['logits_forward'])} live x3 = "
          f"{format_bytes(estimate['logits_peak'])} | estimated peak "
          f"{format_bytes(estimate['total'])} | available "
          f"{format_bytes(guard['available_bytes'])}")
    if not guard["safe"]:
        print(f"\n  estimated peak {format_bytes(estimate['total'])} exceeds "
              f"{args.memory_headroom:.0%} of available memory "
              f"({format_bytes(guard['budget_bytes'])}).\n"
              "  The dominant term is the logits tensor: batch x seq x vocab x bytes.\n"
              "  CPU training cannot use bfloat16, so each value costs 4 bytes, and the\n"
              "  stored forward copy, the loss intermediate and the gradient are all\n"
              "  live at the same time.\n"
              "  Lower train.batch_size and raise train.grad_accumulation by the same\n"
              "  factor: the effective batch, and therefore the update, is unchanged\n"
              "  (tests/test_local_train.py proves the equivalence).")
        return 1

    # 중첩 궤적 측정 (snapshot) --------------------------------------------
    #
    # 한 라운드 안에서는 **모든 클라이언트가 같은 global 파라미터에서 출발**한다.
    # 따라서 K=64 를 향해 달리는 도중 4, 8, 16, 32 스텝 지점을 반드시 지나가고,
    # 그 지점의 파라미터를 찍어두면 작은 K 들의 divergence 를 추가 비용 없이 얻는다.
    #
    #   0 ----4----8------16----------32------------------64
    #        snap snap    snap        snap                snap
    #
    # K 마다 처음부터 다시 돌리면 클라이언트당 라운드당 sum(K)=124 스텝이 들지만,
    # 중첩으로 재면 max(K)=64 스텝이면 끝난다 (실측 31 초/스텝 기준 25.6 시간 -> 13.2 시간).
    # 부수 효과로 측정 품질도 올라간다: 다섯 개 K 가 **같은 궤적**에서 측정되므로
    # K 사이 비교에 난수 궤적 차이가 섞이지 않는다.
    #
    # **한계를 분명히 해둔다.** 중첩이 정확한 것은 1 라운드뿐이다. 2 라운드부터는
    # 출발점이 FedAvg 결과이고 그 결과는 K 에 따라 달라지므로, 작은 K 의 실제
    # 본실험 궤적과는 다르다. 여기서는 global 궤적을 K_max 로 전진시키고,
    # 각 라운드의 divergence 는 "그 라운드의 공통 출발점에서 K 스텝 떨어진 거리"로
    # 읽는다. 1 라운드 값은 K 별 무교란 측정이고, 2~3 라운드는 "학습이 진행된
    # 모델에서도 그 갈라짐이 유지되는가"라는 보강 확인이다 - 라운드 2~3 가 원래
    # 담당했던 질문이 바로 그것이다.
    Ks = sorted(candidate_Ks)
    K_max = Ks[-1]
    global_params = {k: v.copy() for k, v in initial.items()}
    trainers = []
    for cid, examples in enumerate(client_examples):
        loader = build_dataloader(
            examples, tokenizer, cfg.train.batch_size,
            cfg.model.max_seq_length, seed=1000 + cid,
        )
        trainers.append(LocalTrainer(model, loader, cfg.train, args.device))

    print(f"  nested probe: {K_max} local steps per client per round covers "
          f"K in {Ks} (vs {sum(Ks)} if each K restarted)")
    print(f"  total training steps: {K_max * probe_rounds * len(trainers):,} "
          f"(instead of {sum(Ks) * probe_rounds * len(trainers):,})")

    # 네 가지를 모두 남긴다. 정규화된 divergence 와 절대 흩어짐은 K 에 대해
    # 반대로 움직이므로(아래 출력 참조), 기준을 다시 정할 때 13 시간을 다시
    # 쓰지 않으려면 둘 다 기록해야 한다.
    per_round: Dict[int, Dict[int, float]] = {}
    cosine_per_round: Dict[int, Dict[int, float]] = {}
    spread_per_round: Dict[int, Dict[int, float]] = {}
    norm_per_round: Dict[int, Dict[int, float]] = {}
    # 진행 출력은 **클라이언트마다** 낸다. divergence 는 클라이언트 전원의 스냅샷이
    # 모여야 계산되므로 라운드당 측정 줄은 5 개뿐이고, 그 사이가 K_max x 클라이언트
    # = 512 스텝(실측 약 4.4 시간)이다. 그 동안 출력이 없으면 멈춘 것과 구별할 수
    # 없다 - 13 시간 작업에서는 그 자체가 결함이다.
    import time as _time

    total_steps = K_max * probe_rounds * len(trainers)
    steps_done = 0
    t_start = _time.time()
    for rnd in range(probe_rounds):
        snapshots = {K: [] for K in Ks}
        for cid, trainer in enumerate(trainers):
            set_trainable_state_dict(model, numpy_to_state_dict(global_params))
            done = 0
            t_client = _time.time()
            for K in Ks:
                trainer.train_steps(K - done)      # 증분만 돌린다
                done = K
                snapshots[K].append(state_dict_to_numpy(get_trainable_state_dict(model)))
            steps_done += K_max
            elapsed = _time.time() - t_start
            per_step = elapsed / max(1, steps_done)
            remaining = (total_steps - steps_done) * per_step
            print(f"      round {rnd + 1}/{probe_rounds} client {cid + 1}/{len(trainers)}: "
                  f"{K_max} steps in {(_time.time() - t_client) / 60:.1f} min | "
                  f"{steps_done}/{total_steps} steps | {per_step:.1f} s/step | "
                  f"eta {remaining / 3600:.1f} h", flush=True)

        per_round[rnd] = {}
        cosine_per_round[rnd] = {}
        spread_per_round[rnd] = {}
        norm_per_round[rnd] = {}
        for K in Ks:
            stats = summarise(client_deltas(global_params, snapshots[K]))
            per_round[rnd][K] = stats["divergence"]
            cosine_per_round[rnd][K] = stats["mean_pairwise_cosine"]
            spread_per_round[rnd][K] = stats["absolute_spread"]
            norm_per_round[rnd][K] = stats["mean_update_norm"]
            print(f"    round {rnd + 1}/{probe_rounds}  K={K:>3}  "
                  f"divergence {stats['divergence']:.4f}  "
                  f"spread {stats['absolute_spread']:.5f}  "
                  f"|update| {stats['mean_update_norm']:.5f}  "
                  f"cos {stats['mean_pairwise_cosine']:+.3f}")
        # global 궤적은 K_max 지점으로 전진시킨다 (위 한계 설명 참조).
        global_params = dict(fedavg(snapshots[K_max]))

    divergence_by_k = {
        K: float(np.mean([per_round[r][K] for r in range(probe_rounds)])) for K in Ks
    }
    divergence_round1 = {K: float(per_round[0][K]) for K in Ks}
    print("  mean divergence over rounds: " +
          ", ".join(f"K={K}:{divergence_by_k[K]:.4f}" for K in Ks))
    print("  round 1 only (unconfounded):  " +
          ", ".join(f"K={K}:{divergence_round1[K]:.4f}" for K in Ks))
    spread_by_k = {
        K: float(np.mean([spread_per_round[r][K] for r in range(probe_rounds)])) for K in Ks
    }
    print("  absolute spread (un-normalised): " +
          ", ".join(f"K={K}:{spread_by_k[K]:.5f}" for K in Ks))
    # 두 지표가 반대로 움직이면 선택 규칙이 어느 쪽을 보는지가 결정을 바꾼다.
    norm_falls = divergence_by_k[Ks[0]] > divergence_by_k[Ks[-1]]
    spread_rises = spread_by_k[Ks[0]] < spread_by_k[Ks[-1]]
    if norm_falls and spread_rises:
        print("\n  NOTE: normalised divergence falls with K while absolute spread rises.\n"
              "  The rule 'smallest K clearing the threshold' then always returns the\n"
              "  smallest candidate, because dividing by ||mean update|| cancels exactly\n"
              "  the growth that makes larger K drift more. Both curves are recorded in\n"
              "  the result JSON, so the criterion can be revisited without re-running.")

    examples_per_client = int(np.median([len(c) for c in client_examples]))
    # 한 step 이 소비하는 예제 수는 batch_size x grad_accumulation 이다.
    effective_batch = cfg.train.batch_size * accum
    # 프로브는 클라이언트당 질문을 --max-questions-per-client 로 제한하므로, 여기서
    # 나온 S 는 **프로브 규모**의 값이다. 본실험의 S 는 클라이언트당 실제 질문 수
    # (전체 질문 / num_clients) 로 다시 계산해야 한다 - 둘을 혼동하면 R 이 실제보다
    # 훨씬 작게 보인다 (프로브 250 질문 vs 본실험 약 10,090 질문, 약 40 배 차이).
    S = total_steps_from_passes(
        examples_per_client, effective_batch,
        cfg.gates.local_steps.passes_over_client_data,
    )
    questions_per_client_full = len(questions_all) // num_clients
    S_full = total_steps_from_passes(
        questions_per_client_full, effective_batch,
        cfg.gates.local_steps.passes_over_client_data,
    )
    # 판정은 **본실험 규모 S** 로 한다. 프로브의 S 는 클라이언트당 질문을 250 개로
    # 제한한 값이라 R 이 실제보다 약 40 배 작게 보이고, 그 R 로 min_rounds 를
    # 검사하면 멀쩡한 K 가 떨어진다.
    result = local_step_decision(
        divergence_round1,                       # K 선택의 근거는 1 라운드뿐이다
        cfg.gates.local_steps.min_divergence,
        total_local_steps=S_full,
        chosen_K=cfg.get_path("gates.local_steps.chosen_K", None),
        spread_by_k={K: float(spread_per_round[0][K]) for K in Ks},
        min_rounds=int(cfg.get_path("gates.local_steps.min_rounds", 20)),
    )
    result.detail.update({
        "measurement": "nested snapshots along one trajectory per round",
        "steps_trained": K_max * probe_rounds * len(trainers),
        "steps_if_restarted_per_K": sum(Ks) * probe_rounds * len(trainers),
        "divergence_round1_only": divergence_round1,
        "absolute_spread_by_k": spread_by_k,
        "spread_per_round": {str(r): {str(K): v for K, v in d.items()}
                             for r, d in spread_per_round.items()},
        "update_norm_per_round": {str(r): {str(K): v for K, v in d.items()}
                                  for r, d in norm_per_round.items()},
        "divergence_per_round": {str(r): {str(K): v for K, v in d.items()}
                                 for r, d in per_round.items()},
        "cosine_per_round": {str(r): {str(K): v for K, v in d.items()}
                             for r, d in cosine_per_round.items()},
        "aggregated_at_K": K_max,
        "examples_per_client_median": examples_per_client,
        "effective_batch": effective_batch,
        "questions_per_client_full_scale": questions_per_client_full,
        "total_local_steps_full_scale": S_full,
        "rounds_full_scale_by_K": {str(K): S_full / K for K in Ks},
        "passes_over_client_data": cfg.gates.local_steps.passes_over_client_data,
        "baseline_depth": baseline_depth,
        "probe_rounds": probe_rounds,
    })

    log_dir = Path(cfg.paths.log_dir) / "gates"
    save_json(result.to_dict(), log_dir / "gate3_local_steps.json")

    if result.passed:
        K = result.detail["chosen_K"]
        # **본실험 규모**의 S 를 기록한다. 프로브의 S(클라이언트당 250 질문)를 쓰면
        # Week 3/4 가 1/40 짜리 예산으로 돌아간다.
        R = rounds_from_budget(S_full, K)
        save_json(
            {"train": {"local_steps_per_round": K, "total_local_steps": S_full, "rounds": R},
             "source": "gate3_local_step_calibration",
             "scale": {
                 "questions_per_client": questions_per_client_full,
                 "effective_batch": effective_batch,
                 "passes_over_client_data": cfg.gates.local_steps.passes_over_client_data,
                 "probe_total_local_steps": S,
                 "probe_questions_per_client": examples_per_client,
             },
             "selection": result.detail.get("selection"),
             "evidence_round1": {
                 "divergence_by_k": {str(k): divergence_round1[k] for k in Ks},
                 "absolute_spread_by_k": {str(k): float(spread_per_round[0][k]) for k in Ks},
                 "divergence_at_chosen_K": result.detail.get("divergence_at_chosen_K"),
                 "spread_ratio_vs_smallest_K": result.detail.get("spread_ratio_vs_smallest_K"),
             },
             "divergence_by_k_mean_over_rounds": divergence_by_k},
            log_dir / "calibrated_schedule.json",
        )
        print(f"\n  wrote {log_dir / 'calibrated_schedule.json'} "
              f"(K={K}, S={S_full:,}, R={R}) - run scripts read this, not a convention")
        print(f"    probe scale was S={S} ({examples_per_client} questions/client); "
              f"the schedule uses the full scale ({questions_per_client_full:,} questions/client)")

    print(f"\n{result.report()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
