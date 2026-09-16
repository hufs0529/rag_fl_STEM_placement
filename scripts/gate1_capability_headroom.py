#!/usr/bin/env python3
"""Week 1 gate (i): capability headroom - 학습 없이 상한을 잰다.

gold passage 를 넣었을 때와 random passage 를 넣었을 때의 answer F1 격차는,
어떤 retrieval 개선이라도 가져다 줄 수 있는 정확도의 **상한**이다.

**few-shot 시연을 쓰는 이유 (실측 근거):** 시연 없이 재면 학습 전 135M 모델이
출력 형식을 몰라 빈 문자열이나 컨텍스트 이어쓰기를 뱉는다(gold F1 1.19,
random F1 0.00). 그러면 게이트가 "맥락을 쓸 수 있는가"가 아니라 "형식을
아는가"를 재게 되어, 모델을 키워도 풀리지 않는 실패로 오진한다. 본실험은
answer-token loss 로 형식을 가르치므로, 여기서는 시연으로 형식만 알려주고
**맥락 활용 능력만 분리해서** 잰다. 격차가
좁으면(계획서 기준 ~5 F1 미만) 이 체크포인트로는 처리 효과를 담을 공간이
없다는 뜻이므로 더 큰 체크포인트로 올린다.

학습이 전혀 필요 없으므로 Week 1 첫날에 돌릴 수 있고, 실패해도 잃는 것이 없다.

Gate (i): the F1 gap between gold and random context bounds what any retrieval
improvement can deliver. Requires no training.

Usage:
  python scripts/gate1_capability_headroom.py --set gates.headroom.n_questions=300
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.gates import headroom_decision
from src.generate import greedy_generate
from src.logging_utils import banner, save_json
from src.metrics import aggregate
from src.models import load_base_model, load_tokenizer
from src.nq_data import load_questions
from src.prompting import build_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None, help="체크포인트 교체 검증용")
    parser.add_argument("--n-shots", type=int, default=None,
                        help="형식 시연 개수 (기본: gates.headroom.n_shots)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    questions_path = args.questions or data_dir / "questions_usable.jsonl"
    passages_path = args.passages or data_dir / "gold_passages.jsonl"
    model_name = args.model or cfg.model.base_model
    n = cfg.gates.headroom.n_questions
    n_shots = args.n_shots if args.n_shots is not None else cfg.get_path("gates.headroom.n_shots", 0)
    top_k = cfg.retrieval.top_k

    banner(f"Week 1 gate (i) - capability headroom | {model_name}")

    questions = load_questions(questions_path)
    store = PassageStore.from_jsonl(passages_path)
    print(f"  {len(questions):,} usable questions | {len(store):,} passages available")

    rng = random.Random(cfg.get_path("seed", 42))
    sampled = rng.sample(questions, min(n, len(questions)))
    all_pids = store.ids()

    gold_prompts, random_prompts, golds = [], [], []
    for q in sampled:
        gold_ids = [pid for pid in q["gold_passage_ids"] if pid in store][:top_k]
        if not gold_ids:
            continue
        # gold 를 top_k 까지 채우되 모자라면 무작위로 채운다 - 컨텍스트 길이를
        # 두 조건에서 동일하게 유지해야 길이 효과가 격차에 섞이지 않는다.
        filler = [p for p in rng.sample(all_pids, top_k) if p not in gold_ids]
        gold_ids = (gold_ids + filler)[:top_k]
        random_ids = rng.sample(all_pids, top_k)

        gold_prompts.append(build_prompt(q["question"], store.many(gold_ids), n_shots))
        random_prompts.append(build_prompt(q["question"], store.many(random_ids), n_shots))
        golds.append(q["answers"])

    print(f"  scoring {len(golds)} questions under gold vs random context "
          f"({n_shots} format demonstrations) ...")
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, cfg.model.dtype, args.device)

    gen = lambda prompts: greedy_generate(
        model, tokenizer, prompts,
        max_new_tokens=cfg.model.max_new_tokens,
        max_length=cfg.model.max_seq_length,
    )
    gold_scores = aggregate(gen(gold_prompts), golds)
    random_scores = aggregate(gen(random_prompts), golds)

    result = headroom_decision(
        f1_gold=gold_scores["f1"],
        f1_random=random_scores["f1"],
        min_gap=cfg.gates.headroom.min_f1_gap,
        fallback_model=cfg.gates.headroom.fallback_model,
    )
    result.detail.update({
        "model": model_name,
        "n_shots": n_shots,
        "n_questions": len(golds),
        "gold_context": gold_scores,
        "random_context": random_scores,
    })

    # 숫자만 보면 "능력 부족"과 "형식 미숙지"를 구분할 수 없다. 실제 출력 몇 개를
    # 같이 남겨서, 실패했을 때 어느 쪽인지 눈으로 확인할 수 있게 한다.
    samples = gen(gold_prompts[:5])
    result.detail["sample_predictions"] = [
        {"answers": g, "prediction": p} for g, p in zip(golds[:5], samples)
    ]
    print("\n  sample predictions under gold context:")
    for g, p in zip(golds[:5], samples):
        print(f"    gold={g[:2]}  ->  {p!r}")

    save_json(result.to_dict(), Path(cfg.paths.log_dir) / "gates" / "gate1_headroom.json")
    print(f"\n  gold context   F1 {gold_scores['f1']:6.2f}  EM {gold_scores['em']:6.2f}")
    print(f"  random context F1 {random_scores['f1']:6.2f}  EM {random_scores['em']:6.2f}")
    print(f"\n{result.report()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
