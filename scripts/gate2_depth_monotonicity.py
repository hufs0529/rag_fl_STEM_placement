#!/usr/bin/env python3
"""Week 1 gate (ii): depth monotonicity - 처리 축이 실제로 존재하는지.

d in {0, 3, 10, 50} 에서 gold recall@3 이 단조 증가해야 한다. 증가하지 않으면
"depth 를 늘리면 검색 품질이 오른다"는 전제가 성립하지 않는 것이고, 그러면
본실험의 처리 축 자체가 없는 셈이므로 depth 집합을 다시 잡아야 한다.

학습은 필요 없고 retrieval 만 필요하다. Week 2 의 Qdrant 컬렉션이 아직 없으므로
소규모 브루트포스 인덱스로 돌린다(동일한 search() 인터페이스).

**프로브 코퍼스는 Week 2 코퍼스의 축소판이어야 한다.** 그러지 않으면 게이트가
아무도 학습하지 않을 코퍼스를 재게 된다. 실제로 그 실수를 한 적이 있다 -
질문당 gold 를 전부(평균 13.4권) 넣고 채움재도 gold 더미에서 가져오니
5,000권 중 2,689권이 표본 질문의 정답지(54%)가 되어, dense 가 이미 91% 를
상위 3개 안에 넣었고 recoverable 이 7.5% 로 주저앉았다. 난이도가 본실험과
전혀 달랐던 것이다.

두 번째 실수는 더 조용했다. 질문당 1 권으로 고치고 나머지를 임의 위키 문서로
채우니, 이번엔 코퍼스의 8% 만 "누군가의 정답지"가 됐다 - Week 2 는 40.4% 다
(80,720 / 200,000). 임의 문서는 이기기 쉽고 **다른 질문의 정답지는 어렵다**:
답이 들어있는 문서라 질의와 어휘가 겹친다.

그래서 구성을 Week 2 와 맞춘다 (src/probe_corpus.py, 테스트 포함):
    표본 질문의 gold    질문당 1 권 (data.gold_per_question_in_corpus)
  + 다른 질문의 gold    gold 비중이 Week 2 와 같아질 만큼 (약 40%)
  + 정답 미포함 채움재  나머지 (data/random_passages.jsonl)
  + 크기는 클라이언트당 인덱스 크기(약 25,000)

불변식: **표본 질문의 gold 는 한 권도 채움재로 들어가면 안 된다.** 질문당 1 권만
정답으로 유지하므로, 나머지가 라벨 없이 앉으면 recall 이 과소계상된다.

BM25 hard negative 는 넣지 않는다. 넣으면 gold 가 더 아래로 밀려
recoverable 이 **늘어나므로**, 빼고 재는 쪽이 보수적이다 - 여기서 통과하면
Week 2 에서는 최소한 그만큼은 나온다.

d=3 이 d=0 과 recall 이 같아야 한다는 점도 함께 확인한다: d=3 은 재정렬만 하고
새 passage 를 승격하지 않는 control 이므로, 정의상 recall@3 은 변하지 않는다.
여기서 값이 달라지면 캐시나 선택 로직에 버그가 있다는 뜻이다.

Gate (ii): gold recall@3 must rise monotonically across depths, otherwise the
treatment axis does not exist. Requires retrieval only. The d=3 control must
leave recall@3 unchanged by construction - a difference there indicates a bug.

Usage:
  python scripts/gate2_depth_monotonicity.py --set gates.monotonicity.n_questions=2000
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.data import PassageStore
from src.embedding import Embedder, passage_text_for_embedding
from src.embedding_cache import load_or_encode
from src.gates import monotonicity_decision
from src.logging_utils import banner, save_json
from src.probe_corpus import plan_probe_corpus, week2_gold_share
from src.metrics import (
    gold_recall_at_k,
    headroom_summary,
    paired_recall_delta,
    retrieval_bucket,
)
from src.rerank import CrossEncoderReranker
from src.retrieval import BruteForceRetriever, dense_ranked_ids
from src.selection import promotion_rate, rank_change_rate, select_passages
from src.nq_data import load_questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--scratch-corpus", type=int, default=25000,
                        help="프로브 코퍼스 크기 (기본 25,000 = Week 2 의 클라이언트당 인덱스 크기)")
    parser.add_argument("--random-passages", default=None,
                        help="채움재 경로 (기본: data/random_passages.jsonl)")
    parser.add_argument("--gold-filler-share", type=float, default=None,
                        help="코퍼스 중 '누군가의 정답지' 비중. 기본값은 Week 2 에서 유도한다 "
                             "(gold_per_question x 질문수 / corpus_size)")
    parser.add_argument("--diagnostic-pools", default="50,100,200",
                        help="후보 풀을 키우면 out_of_pool 이 얼마나 줄어드는지 함께 측정한다. "
                             "재채점은 하지 않으므로 비용은 거의 0 이다")
    parser.add_argument("--embedding-cache", default="results/cache/embeddings",
                        help="코퍼스 임베딩 캐시 디렉터리 (내용 해시 기반, 빈 문자열이면 비활성)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dump-cache", default="results/cache/gate_scratch.json",
                        help="gate 3 가 재사용할 스크래치 retrieval 캐시 경로")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_dir = Path(cfg.paths.data_dir)
    questions = load_questions(args.questions or data_dir / "questions_usable.jsonl")
    store = PassageStore.from_jsonl(args.passages or data_dir / "gold_passages.jsonl")
    depths = list(cfg.treatment.depths)
    pool = cfg.retrieval.candidate_pool
    top_k = cfg.retrieval.top_k

    banner(f"Week 1 gate (ii) - depth monotonicity | depths={depths}")

    rng = random.Random(cfg.get_path("seed", 42))
    sampled = rng.sample(questions, min(cfg.gates.monotonicity.n_questions, len(questions)))

    gold_per_q = cfg.get_path("data.gold_per_question_in_corpus", 1) or 1

    random_path = Path(args.random_passages or data_dir / "random_passages.jsonl")
    if not random_path.exists():
        print(f"  {random_path} not found.\n"
              "  The probe corpus needs answer-free filler, otherwise it is made of gold\n"
              "  passages and dense retrieval finds everything (measured: already 91%).\n"
              "  Re-run scripts/scan_corpus.py, which fills that reservoir in the same pass.")
        return 1
    filler_store = PassageStore.from_jsonl(random_path)

    # 구성 결정은 src/probe_corpus.py 에 있다. 이 로직이 두 번 틀렸기 때문에
    # (54% gold -> already 91%, 그리고 8% gold -> Week 2 의 40.4% 와 불일치)
    # 스크립트에 묻어두지 않고 테스트가 붙은 모듈로 옮겼다.
    gold_share = (
        args.gold_filler_share
        if args.gold_filler_share is not None
        else week2_gold_share(len(questions), cfg.data.corpus_size, gold_per_q)
    )
    plan = plan_probe_corpus(
        sampled,
        contains=lambda pid: pid in store,
        all_gold_ids=store.ids(),
        filler_ids=filler_store.ids(),
        size=args.scratch_corpus,
        gold_share=gold_share,
        gold_per_question=gold_per_q,
        shuffle=rng.shuffle,
    )
    sampled = plan["questions"]
    gold_ids = plan["question_gold"]
    gold_filler = plan["other_question_gold"]
    filler = plan["answer_free_filler"]
    if plan["shortfall"]:
        print(f"  WARNING: probe corpus is {plan['shortfall']:,} passages short of "
              f"{args.scratch_corpus:,} - the filler pool ran out")

    corpus = store.many(gold_ids) + store.many(gold_filler) + filler_store.many(filler)
    for passage in corpus:
        if passage["pid"] not in store:
            store.add(passage)             # 이후 조회가 한 곳에서 되도록
    print(f"  probe corpus {len(corpus):,} passages = "
          f"{len(gold_ids):,} gold ({gold_per_q}/question) + "
          f"{len(gold_filler):,} other questions' gold + {len(filler):,} answer-free filler")
    print(f"  gold share {plan['gold_share']:.1%} (Week 2: {gold_share:.1%})")
    print(f"  {len(sampled):,} questions (Week 2 composition at client-index scale)")

    embedder = Embedder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix, args.device)
    corpus_ids = [p["pid"] for p in corpus]
    corpus_texts = [passage_text_for_embedding(p) for p in corpus]
    # 임베더는 동결이고 코퍼스는 결정적이므로 같은 설정의 재실행은 벡터를 재사용한다.
    # CPU 에서 25,000 권 임베딩이 37 분이고, 설정을 고쳐 다시 돌리는 일이 잦다.
    if args.embedding_cache:
        corpus_vectors = load_or_encode(
            args.embedding_cache, cfg.retrieval.embedding_model, corpus_ids, corpus_texts,
            encode=lambda: embedder.encode_passages(corpus_texts),
        )
    else:
        print("  embedding corpus ...")
        corpus_vectors = embedder.encode_passages(corpus_texts)
    retriever = BruteForceRetriever(corpus_ids, corpus_vectors)

    # 진단용으로 더 깊은 풀을 한 번만 가져온다. 처리에는 앞 pool 개만 쓰므로
    # 결과는 바뀌지 않고, 브루트포스 검색이라 추가 비용은 무시할 수준이다.
    diagnostic_pools = sorted({
        int(x) for x in str(args.diagnostic_pools).split(",") if x.strip()
    } | {pool})
    deep_pool = min(max(diagnostic_pools), len(corpus))
    print(f"  embedding queries and retrieving the top-{deep_pool} pool "
          f"(treatment uses the first {pool}) ...")
    query_vectors = embedder.encode_queries([q["question"] for q in sampled])
    hits = retriever.search(query_vectors, deep_pool)
    deep_candidates = [dense_ranked_ids(h) for h in hits]
    candidates = [c[:pool] for c in deep_candidates]

    print(f"  rescoring the pool once ({len(sampled) * pool:,} pairs) ...")
    reranker = CrossEncoderReranker(
        cfg.rerank.model, cfg.rerank.max_length, cfg.rerank.batch_size, args.device
    )
    scores = reranker.score_batch(
        [q["question"] for q in sampled],
        [store.many(c) for c in candidates],
    )

    # 질문을 세 통으로 분해한다. depth 가 일할 수 있는 구간은 recoverable 뿐이다:
    # already 는 이미 상위 3개 안에 정답지가 있어 모든 조건이 같은 passage 를 고르고,
    # out_of_pool 은 풀 밖이라 d=50 도 닿지 못한다. 둘 다 처리 효과가 정확히 0 이다.
    headroom = headroom_summary([
        retrieval_bucket(cand, q["gold_passage_ids"], top_k)
        for q, cand in zip(sampled, candidates)
    ])
    print(f"    already in dense top-{top_k}  : {headroom['already_at_top_k']:.4f}  (depth cannot help)")
    print(f"    recoverable by reranking     : {headroom['recoverable']:.4f}  <- the working range")
    print(f"    gold outside the pool        : {headroom['out_of_pool']:.4f}  (depth cannot reach)")
    print(f"    ceiling (already+recoverable): {headroom['ceiling']:.4f}")

    # 풀을 키우면 out_of_pool 이 얼마나 recoverable 로 넘어오는가.
    # out_of_pool 이 크면 깊이를 늘려도 손이 닿지 않으므로, candidate_pool 을
    # 올리는 것이 유일한 수단이다. 다만 candidate_pool 은 d 의 최댓값이기도 해서
    # 올리면 처리 격자와 비용 모델을 함께 바꿔야 한다 - 그래서 여기서는 재지만
    # 바꾸지는 않는다.
    pool_scan = {}
    for p_size in diagnostic_pools:
        summary = headroom_summary([
            retrieval_bucket(cand[:p_size], q["gold_passage_ids"], top_k)
            for q, cand in zip(sampled, deep_candidates)
        ])
        pool_scan[str(p_size)] = {
            "recoverable": summary["recoverable"],
            "out_of_pool": summary["out_of_pool"],
            "ceiling": summary["ceiling"],
        }
        mark = "  <- in use" if p_size == pool else ""
        print(f"    pool={p_size:>4}  recoverable {summary['recoverable']:.4f}  "
              f"out_of_pool {summary['out_of_pool']:.4f}  "
              f"ceiling {summary['ceiling']:.4f}{mark}")

    recall_by_depth, diagnostics, recall_vectors = {}, {}, {}
    for depth in depths:
        recalls, changes, promotions = [], [], []
        for q, cand, score_map in zip(sampled, candidates, scores):
            selected = select_passages(cand, score_map, depth, top_k)
            # gold id 대조 - 문자열 재매칭은 "아무 책"을 성공으로 세어 recall 을 부풀린다
            recalls.append(gold_recall_at_k(selected, q["gold_passage_ids"], top_k))
            changes.append(rank_change_rate(cand, selected))
            promotions.append(promotion_rate(cand, selected))
        n = len(recalls)
        recall_vectors[depth] = recalls
        recall_by_depth[depth] = sum(recalls) / n
        diagnostics[str(depth)] = {
            "gold_recall_at_3": sum(recalls) / n,
            "rank_change_rate": sum(changes) / n,
            "promotion_rate": sum(promotions) / n,
        }
        print(f"    d={depth:>2}  recall@3 {recall_by_depth[depth]:.4f}  "
              f"rank-change {diagnostics[str(depth)]['rank_change_rate']:.3f}  "
              f"promotion {diagnostics[str(depth)]['promotion_rate']:.3f}")

    result = monotonicity_decision(
        recall_by_depth,
        headroom=headroom,
        min_recoverable=cfg.get_path("gates.monotonicity.min_recoverable", 0.10),
    )
    best = max(recall_by_depth.values())

    # 상승폭의 불확실성. 비율만 보면 0.0133 이 커 보이지만 그 실체는 판정이
    # 뒤바뀐 질문 몇 개다. 같은 질문을 쌍으로 비교해 그 개수를 그대로 보고한다.
    deepest = max(depths)
    paired = (
        paired_recall_delta(recall_vectors[0], recall_vectors[deepest])
        if 0 in recall_vectors and deepest in recall_vectors and deepest != 0
        else None
    )
    if paired:
        z_text = "n/a" if paired["z"] is None else f"{paired['z']:.2f}"
        print(f"    d=0 -> d={deepest} paired: +{paired['gained']} / -{paired['lost']} questions "
              f"of {paired['n']}  delta {paired['delta']:+.4f} +- {paired['std_error']:.4f}  z {z_text}")
        if not paired["separated_from_zero"]:
            print("    WARNING: the gain is not separated from zero at 2 s.e. "
                  "Monotonicity alone would still pass this gate - raise "
                  "gates.monotonicity.n_questions before relying on the effect size.")

    result.detail.update({
        # 상한 대비, 그리고 **일할 수 있는 폭 대비** 얼마나 건졌는지.
        # 후자가 reranking 의 실제 성적이다.
        "recoverable_captured": (
            (best - headroom["already_at_top_k"]) / headroom["recoverable"]
            if headroom["recoverable"] else None
        ),
        "n_questions": len(sampled),
        "probe_corpus": {
            "size": len(corpus),
            "gold": len(gold_ids),
            "gold_per_question": gold_per_q,
            "other_question_gold": len(gold_filler),
            "answer_free_filler": len(filler),
            "gold_share": plan["gold_share"],
            "gold_share_target": gold_share,
            "hard_negatives": 0,   # 일부러 뺐다 - 넣으면 recoverable 이 늘어나므로 보수적
        },
        "candidate_pool": pool,
        "pool_scan": pool_scan,
        "paired_gain": paired,
        "per_depth": diagnostics,
        "reranker_forward_passes": reranker.forward_passes,
    })
    # control 무결성 확인: d=3 은 recall 을 바꾸면 안 된다.
    if 0 in recall_by_depth and 3 in recall_by_depth:
        delta = abs(recall_by_depth[3] - recall_by_depth[0])
        result.detail["control_recall_delta"] = delta
        if delta > 1e-9:
            print(f"\n  WARNING: d=3 control changed recall@3 by {delta:.6f}; "
                  "it should only reorder. Check the selection/cache path.")

    if args.dump_cache:
        # gate 3 는 같은 후보/점수를 재사용한다 - 게이트 단계에서 임베딩·재채점을
        # 두 번 하지 않기 위해서다 (본실험 캐시는 Week 2 에서 따로 만든다).
        save_json(
            {
                "meta": {"candidate_pool": pool, "corpus_size": len(corpus),
                         "embedding_model": cfg.retrieval.embedding_model,
                         "rerank_model": cfg.rerank.model},
                "entries": {
                    q["qid"]: {"candidates": cand, "scores": score_map}
                    for q, cand, score_map in zip(sampled, candidates, scores)
                },
            },
            args.dump_cache,
        )
        print(f"  wrote scratch retrieval cache -> {args.dump_cache}")

    save_json(result.to_dict(), Path(cfg.paths.log_dir) / "gates" / "gate2_monotonicity.json")
    print(f"\n{result.report()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
