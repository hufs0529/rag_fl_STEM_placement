from src.data import PassageStore
from src.evaluate import build_eval_prompts
from src.retrieval_cache import RetrievalCache


def _fixtures():
    store = PassageStore(
        [{"pid": f"p{i}", "title": f"T{i}", "text": f"body {i}"} for i in range(10)]
    )
    hits = [[(f"p{i}", 1.0 - 0.1 * i) for i in range(10)]]
    scores = [{f"p{i}": float(i) for i in range(10)}]      # dense 와 정반대 순서
    cache = RetrievalCache.build(["q0"], hits, scores)
    questions = [{"qid": "q0", "question": "what?", "answers": ["x"], "client_id": 0,
                  "gold_passage_ids": ["p7"]}]
    return store, {0: cache}, questions


def test_evaluation_context_is_dense_only_regardless_of_training_depth():
    # 평가 컨텍스트를 d=0 으로 고정해야 "모델이 배운 것"만 남는다
    store, caches, questions = _fixtures()
    prompts = build_eval_prompts(questions, caches, store, eval_depth=0)
    assert prompts[0]["selected_ids"] == ["p0", "p1", "p2"]


def test_a_non_zero_eval_depth_would_change_the_context():
    # 이 경로가 살아 있음을 보이되, 설정은 0 으로 고정되어 있다 (eval.eval_depth)
    store, caches, questions = _fixtures()
    prompts = build_eval_prompts(questions, caches, store, eval_depth=10)
    assert prompts[0]["selected_ids"] == ["p9", "p8", "p7"]


def test_eval_prompts_carry_the_answers_for_scoring():
    store, caches, questions = _fixtures()
    prompts = build_eval_prompts(questions, caches, store)
    assert prompts[0]["answers"] == ["x"]
    assert "body 0" in prompts[0]["prompt"]


def test_eval_prompts_carry_gold_ids_for_id_based_recall():
    # 문자열 재매칭은 코퍼스의 "아무 책"까지 성공으로 세므로, 평가 진단도 id 대조를 쓴다
    store, caches, questions = _fixtures()
    prompts = build_eval_prompts(questions, caches, store)
    assert prompts[0]["gold_passage_ids"] == ["p7"]
