#!/usr/bin/env python3
"""합성 동결 산출물 생성 - 네트워크 없이 파이프라인 배선을 점검하기 위한 도구.

Week 2 의 진짜 산출물(코퍼스 200k, Qdrant, 캐시)을 만들려면 21M passage 다운로드가
필요하다. 리허설과 CI 에서 확인하고 싶은 것은 그게 아니라 **배선**이다:
캐시 -> depth 선택 -> 프롬프트 -> 마스킹 -> 로컬 학습 -> 집계 -> 기록 -> 평가.

그래서 구조만 동일하고 크기는 장난감인 산출물을 만든다. 합성 데이터에서는
정답이 특정 passage 안에만 들어 있고 dense 순위는 일부러 어긋나게 만들어,
depth 를 올리면 recall 이 오르도록 구성한다 - 처리 축이 살아 있는지
합성 수준에서라도 확인할 수 있게 하기 위함이다.

Synthetic frozen artefacts so the pipeline wiring can be rehearsed without
downloading 21M passages. Gold passages are deliberately dense-ranked low so
recall rises with depth, keeping the treatment axis alive at toy scale.

Usage:
  python scripts/make_dev_artefacts.py --dev
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.logging_utils import banner, save_json
from src.nq_data import save_questions
from src.retrieval_cache import RetrievalCache, cache_path

TOPICS = ["physics", "cinema", "geography", "music", "sport", "botany", "history", "law"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--questions-per-client", type=int, default=64)
    parser.add_argument("--passages-per-client", type=int, default=200)
    args = parser.parse_args()
    cfg = config_from_args(args)

    rng = random.Random(0)
    data_dir = Path(cfg.paths.data_dir)
    num_clients = cfg.partition.num_clients
    pool = cfg.retrieval.candidate_pool
    top_k = cfg.retrieval.top_k

    banner(f"synthetic artefacts | {num_clients} clients x {args.questions_per_client} questions")

    store = PassageStore()
    client_passages, client_questions = {}, {}
    all_questions = []

    for cid in range(num_clients):
        topic = TOPICS[cid % len(TOPICS)]
        pids = []
        for i in range(args.passages_per_client):
            pid = f"c{cid}_p{i}"
            store.add({"pid": pid, "title": f"{topic} {i}",
                       "text": f"{topic} filler passage number {i} with no answer inside."})
            pids.append(pid)
        client_passages[str(cid)] = pids

        questions = []
        for qi in range(args.questions_per_client):
            answer = f"{topic}{qi}"
            gold_pid = f"c{cid}_g{qi}"
            store.add({"pid": gold_pid, "title": f"{topic} entry {qi}",
                       "text": f"The recorded value for {topic} case {qi} is {answer}."})
            client_passages[str(cid)].append(gold_pid)
            questions.append({
                "qid": f"c{cid}_q{qi}",
                "question": f"what is the recorded value for {topic} case {qi}?",
                "answers": [answer],
                "gold_passage_ids": [gold_pid],
                "client_id": cid,
            })
        client_questions[str(cid)] = questions
        all_questions.extend(questions)

    store.to_jsonl(data_dir / "corpus.jsonl")
    save_questions(all_questions, data_dir / "questions_corpus.jsonl")

    for seed in cfg.partition.partition_seeds:
        save_json(
            {
                "seed": seed,
                "num_clients": num_clients,
                "n_clusters": num_clients,
                "questions_per_client": {i: len(client_questions[str(i)]) for i in range(num_clients)},
                "client_questions": client_questions,
                "client_passages": client_passages,
                "passage_to_client": {
                    pid: cid for cid in range(num_clients)
                    for pid in client_passages[str(cid)]
                },
            },
            data_dir / f"partition_{seed}.json",
        )

    # 캐시는 **분할 시드별로** 만든다. 캐시 경로에 시드가 들어가기 때문이고,
    # 이 루프가 시드 루프 밖에 있던 동안은 마지막 시드(1002)의 값이 새어 들어와
    # 시드 1001 의 캐시가 아예 만들어지지 않았다 - 리허설을 --seed 1001 로 돌리면
    # 캐시를 못 찾는다.
    for seed in cfg.partition.partition_seeds:
        for cid in range(num_clients):
            questions = client_questions[str(cid)]
            hits, scores = [], []
            for q in questions:
                gold = q["gold_passage_ids"][0]
                others = rng.sample(
                    [p for p in client_passages[str(cid)] if p != gold], min(pool - 1, 200)
                )
                # gold 를 dense 순위 뒤쪽에 숨긴다: d=0/3 은 놓치고 d=10 은 가끔,
                # d=50 은 대부분 건지도록. 구간을 **풀 크기에 비례**해 잡는다 -
                # 20~45 로 고정했더니 dev 의 pool=10 에서 randint(20, 9) 가 되어
                # 빈 범위로 터졌다(리허설이 시작도 못 했다).
                #   pool=50 -> 20~45  (원래 의도 그대로)
                #   pool=10 ->  4~9
                lo = max(top_k + 1, int(0.4 * pool))
                hi = max(lo, min(int(0.9 * pool), pool - 1))
                position = rng.randint(lo, hi)
                ranked = others[:position] + [gold] + others[position:]
                ranked = ranked[:pool]
                hits.append([(pid, 1.0 - 0.01 * i) for i, pid in enumerate(ranked)])
                scores.append({
                    pid: (10.0 if pid == gold else rng.uniform(-2.0, 1.0)) for pid in ranked
                })
            cache = RetrievalCache.build(
                [q["qid"] for q in questions], hits, scores,
                meta={"client_id": cid, "partition_seed": seed,
                      "synthetic": True, "candidate_pool": pool},
            )
            cache.save(cache_path(cfg.paths.cache_dir, cid, seed))

    print(f"  corpus {len(store):,} passages | {len(all_questions):,} questions")
    print(f"  caches -> {Path(cfg.paths.cache_dir) / 'retrieval'}")
    print("  NOTE: synthetic artefacts are for wiring rehearsal only, never for results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
