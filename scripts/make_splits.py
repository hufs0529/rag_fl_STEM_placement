#!/usr/bin/env python3
"""train / val / test 를 **파일로** 확정한다.

왜 필요한가: 지금까지 세 집합이 디스크에 없었다. 코드가 실행할 때마다
결정적으로 재현했을 뿐이라 "이 파일이 test 입니다" 를 가리킬 수 없었다.

그리고 두 집합이 겹쳐 있었다. 게이트(val)와 평가셋(test)을 같은 80,720 개에서
독립적으로 뽑았으므로 우연히 24 개가 겹쳤다(실측, 기대 24.8). 실질적 누출은
없다 - 결과를 내는 모델은 test 를 학습하지 않고, 게이트 ② 는 검색만 재고
게이트 ③ 의 모델은 버려진다. 그래도 "완전히 분리됐나" 에 "2.4% 겹칩니다" 로
답하지 않으려면 지워야 한다.

역할:
  val  (= 게이트 표본)  설정을 정하는 데 쓴 질문. 사전 기준으로 판정했다.
  test (= 평가셋)       최종 비교에만 쓴다. 어떤 결정에도 쓰이지 않는다.
  train                 클라이언트에 배분된 질문 - val/test 제외

게이트 표본은 저장되지 않았으므로 **결정적으로 재현**한다. 게이트들은
random.Random(seed).sample(questions, n) 을 쓰고 seed 는 설정에 있다. n 은
게이트별로 다르고 기록돼 있지만, 표본이 n 에 따라 완전히 달라지므로(접두사가
아니다) 쓰였을 수 있는 n 전부의 합집합을 제외한다. test 를 78,752 개에서
1,000 개 뽑는 것이므로 몇 천 개를 더 제외해도 비용이 없다.

Usage:
  python scripts/make_splits.py
  python scripts/make_splits.py --seed 1001        # 특정 분할의 test 만
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.logging_utils import banner, load_json, save_json
from src.nq_data import build_eval_set, load_questions, save_questions

# 게이트가 썼을 수 있는 표본 크기. 기록된 값(gate1 100, gate2 2000)과 설정
# 기본값(gate1 300)을 모두 포함한다.
PROBE_SIZES = (100, 300, 2000)


def probe_qids(questions, seed: int, sizes=PROBE_SIZES) -> set:
    """게이트가 뽑았을 질문의 합집합. 게이트와 **같은** RNG 호출을 재현한다."""
    out = set()
    for n in sizes:
        rng = random.Random(seed)
        out.update(q["qid"] for q in rng.sample(questions, min(n, len(questions))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--seed", type=int, default=None,
                        help="분할 시드 (기본: 설정의 partition_seeds 전부)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    seed = cfg.get_path("seed", 42)
    banner("splits: val (gate probes) / test (held-out) / train (the rest)")

    questions = load_questions(data_dir / "questions_usable.jsonl")
    probe = probe_qids(questions, seed)
    save_json({"seed": seed, "sizes_unioned": list(PROBE_SIZES),
               "n": len(probe), "qids": sorted(probe),
               "role": "val - used to fix settings in the Week 1 gates, never for results"},
              data_dir / "split_val_qids.json")
    print(f"  val   {len(probe):,} qids -> {data_dir / 'split_val_qids.json'}")

    seeds = [args.seed] if args.seed else list(cfg.partition.partition_seeds)
    for ps in seeds:
        path = data_dir / f"partition_{ps}.json"
        if not path.exists():
            print(f"  partition_{ps}.json 없음 - 건너뜀 (Week 2 를 먼저 돌릴 것)")
            continue
        partition = load_json(path)
        # 클라이언트 수를 설정에서 읽지 않고 **파일에서** 읽는다. 축소 실행이 남긴
        # 2 클라이언트 분할 위에서 설정값 8 로 돌면 KeyError '2' 로 죽는다.
        client_ids = sorted(partition["client_questions"], key=int)
        pool = [q for cid in client_ids for q in partition["client_questions"][cid]]
        if len(client_ids) != cfg.partition.num_clients:
            print(f"  NOTE: partition_{ps}.json has {len(client_ids)} clients but the config "
                  f"says {cfg.partition.num_clients} - this looks like a reduced-scale "
                  "artefact; the test set below is not the production one")
        test = build_eval_set(pool, size=cfg.data.eval_set_size, seed=seed, exclude=probe)
        test_ids = {q["qid"] for q in test}
        save_questions(test, data_dir / f"split_test_{ps}.jsonl")
        n_train = len(pool) - len(test_ids)
        print(f"  partition {ps}: pool {len(pool):,} -> "
              f"test {len(test_ids):,} + train {n_train:,}")
        # 불변식: 세 집합이 서로 겹치지 않아야 한다.
        assert not (test_ids & probe), "test 가 val 과 겹친다"
        print(f"    test ∩ val = 0  ✅   -> {data_dir / f'split_test_{ps}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
