from src.data import PassageStore, build_example, selection_diagnostics

QUESTION = {"qid": "q1", "question": "what is the capital of france?", "answers": ["Paris"]}


def _store(n=20):
    return PassageStore(
        [{"pid": f"p{i}", "title": f"T{i}", "text": f"passage body {i}"} for i in range(n)]
    )


def _cache(n=20):
    ids = [f"p{i}" for i in range(n)]
    scores = {pid: -i for i, pid in enumerate(ids)}
    scores["p15"] = 99.0          # dense 가 낮게 본 passage 를 cross-encoder 가 끌어올림
    return ids, scores


def test_every_depth_puts_exactly_three_passages_in_the_prompt():
    store, (ids, scores) = _store(), _cache()
    for depth in (0, 3, 10, 50):
        ex = build_example(QUESTION, ids, scores, store, depth)
        assert len(ex["selected_ids"]) == 3
        assert ex["prompt"].count("[3]") == 1


def test_only_the_passages_differ_between_conditions():
    store, (ids, scores) = _store(), _cache()
    baseline = build_example(QUESTION, ids, scores, store, 0)
    treated = build_example(QUESTION, ids, scores, store, 50)
    assert baseline["selected_ids"] != treated["selected_ids"]
    assert baseline["target"] == treated["target"]
    assert baseline["question"] == treated["question"]


def test_control_depth_leaves_the_passage_set_unchanged():
    store, (ids, scores) = _store(), _cache()
    baseline = build_example(QUESTION, ids, scores, store, 0)
    control = build_example(QUESTION, ids, scores, store, 3)
    assert set(baseline["selected_ids"]) == set(control["selected_ids"])
    assert control["promotion_rate"] == 0.0


def test_diagnostics_average_over_examples():
    store, (ids, scores) = _store(), _cache()
    examples = [build_example(QUESTION, ids, scores, store, d) for d in (0, 50)]
    diag = selection_diagnostics(examples)
    assert diag["n"] == 2
    assert 0.0 < diag["promotion_rate"] <= 1.0


def test_passage_store_round_trips_through_jsonl(tmp_path):
    store = _store(5)
    path = tmp_path / "passages.jsonl"
    assert store.to_jsonl(path) == 5
    reloaded = PassageStore.from_jsonl(path)
    assert len(reloaded) == 5
    assert reloaded.get("p3")["text"] == "passage body 3"


def test_from_jsonl_many_merges_two_files(tmp_path):
    from src.data import PassageStore
    a = tmp_path / "gold.jsonl"
    b = tmp_path / "random.jsonl"
    a.write_text('{"pid": "1", "title": "A", "text": "a"}\n')
    b.write_text('{"pid": "2", "title": "B", "text": "b"}\n')
    store = PassageStore.from_jsonl_many([a, b])
    assert len(store) == 2 and store.get("2")["title"] == "B"


def test_from_jsonl_many_keeps_only_the_wanted_ids(tmp_path):
    from src.data import PassageStore
    path = tmp_path / "p.jsonl"
    path.write_text("".join(
        '{"pid": "%d", "title": "t", "text": "x"}\n' % i for i in range(100)
    ))
    store = PassageStore.from_jsonl_many([path], keep={"3", "7"})
    assert sorted(store.ids()) == ["3", "7"]       # 10 만 권 중 필요한 것만 올린다


def test_from_jsonl_many_tolerates_a_missing_file(tmp_path):
    from src.data import PassageStore
    path = tmp_path / "p.jsonl"
    path.write_text('{"pid": "1", "title": "t", "text": "x"}\n')
    store = PassageStore.from_jsonl_many([path, tmp_path / "absent.jsonl"])
    assert len(store) == 1


def test_missing_names_the_ids_that_are_absent(tmp_path):
    from src.data import PassageStore
    store = PassageStore([{"pid": "1", "title": "t", "text": "x"}])
    assert store.missing(["1", "2", "2", "3"]) == ["2", "3"]   # 중복은 한 번만
    assert store.missing(["1"]) == []
