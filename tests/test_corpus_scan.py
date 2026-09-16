from collections import Counter

import pytest

from src.corpus_scan import ReservoirSampler, ScanStats, extrapolate_usable, scan_for_gold
from src.nq_data import AnswerIndex

QUESTIONS = [{"qid": "a", "question": "capital of france", "answers": ["Paris"]}]

# 스캔 테스트는 1단계 동작을 보므로 겹침 필터를 끈다 (2단계는 test_nq_data 에서)
def _index():
    return AnswerIndex(QUESTIONS, min_question_overlap=0)


def _stream(n=200, every=10):
    """every 번째 passage 만 정답을 포함하도록 만든 합성 스트림."""
    return [
        {"pid": str(i), "title": "t",
         "text": "the capital is Paris" if i % every == 0 else f"unrelated sentence {i}"}
        for i in range(n)
    ]


def test_reservoir_keeps_exactly_k_items():
    sampler = ReservoirSampler(5, seed=0)
    for i in range(100):
        sampler.offer({"pid": str(i)})
    assert len(sampler) == 5
    assert sampler.seen == 100


def test_reservoir_keeps_everything_when_the_stream_is_shorter_than_k():
    sampler = ReservoirSampler(50, seed=0)
    for i in range(7):
        sampler.offer({"pid": str(i)})
    assert len(sampler) == 7


def test_reservoir_is_not_just_the_first_k_items():
    # 위키 덤프는 문서 순서라, 앞에서 k 개만 쓰면 주제가 심하게 치우친다
    sampler = ReservoirSampler(10, seed=1)
    for i in range(1000):
        sampler.offer({"pid": str(i)})
    picked = [int(item["pid"]) for item in sampler.items]
    assert max(picked) > 500


def test_reservoir_samples_roughly_uniformly():
    counts = Counter()
    for seed in range(300):
        sampler = ReservoirSampler(1, seed=seed)
        for i in range(10):
            sampler.offer({"pid": str(i)})
        counts[sampler.items[0]["pid"]] += 1
    # 10 개 위치가 모두 뽑히고, 어느 하나가 과반을 차지하지 않는다
    assert len(counts) == 10
    assert max(counts.values()) < 150


def test_zero_size_reservoir_collects_nothing():
    sampler = ReservoirSampler(0)
    sampler.offer({"pid": "x"})
    assert len(sampler) == 0


def test_random_sample_never_contains_an_answer_bearing_passage():
    # random remainder 에 정답 보유 passage 가 섞이면 recall 이 false positive 를 낸다
    sampler = ReservoirSampler(20, seed=0)
    scan_for_gold(_stream(), _index(), random_sampler=sampler,
                  progress_every=0, log=lambda *a: None)
    assert len(sampler) == 20
    assert all("Paris" not in item["text"] for item in sampler.items)


def test_scan_counts_answer_free_passages_alongside_gold():
    sampler = ReservoirSampler(5, seed=0)
    gold, stats = scan_for_gold(_stream(n=100, every=10), _index(),
                                random_sampler=sampler, progress_every=0, log=lambda *a: None)
    assert stats.passages_with_answer == 10
    assert stats.answer_free_seen == 90
    assert stats.random_sampled == 5
    assert len(gold["a"]) == 10


def test_scan_still_works_without_a_sampler():
    gold, stats = scan_for_gold(_stream(), _index(),
                                progress_every=0, log=lambda *a: None)
    assert stats.random_sampled == 0
    assert gold["a"]


def test_gold_per_question_is_capped_so_common_answers_cannot_dominate():
    _, stats = scan_for_gold(_stream(n=1000, every=2), _index(),
                             max_gold_per_question=20, progress_every=0, log=lambda *a: None)
    assert stats.gold_pairs == 20


def test_sampled_scan_extrapolates_above_the_observed_lower_bound():
    stats = ScanStats(questions_covered=6000, sample_fraction=0.1)
    out = extrapolate_usable(stats, total_questions=87925)
    assert out["estimated_usable"] > out["observed_usable"]


# --- 스트리밍 소스 (Subtask 1.1) -------------------------------------------

from src.corpus_scan import _apply_sampling, _normalise_row


def test_column_aliases_are_normalised_to_pid_title_text():
    # 미러마다 id 열 이름이 다르다 (id / _id). 하류 코드가 하나만 알면 되도록 통일한다
    assert _normalise_row({"id": 1, "title": "Aaron", "text": "t"}) == {
        "pid": "1", "title": "Aaron", "text": "t"}
    assert _normalise_row({"_id": "doc5", "title": None, "text": "t"})["pid"] == "doc5"
    assert _normalise_row({"id": 2, "text": "t"})["title"] == ""


def _rows(n=20):
    return [_normalise_row({"id": i, "title": f"T{i}", "text": f"x{i}"}) for i in range(n)]


def test_sampling_spreads_across_the_stream_rather_than_taking_the_head():
    # 위키 덤프는 문서 순서라 "앞에서 N 개"는 주제가 심하게 치우친다
    picked = [r["pid"] for r in _apply_sampling(_rows(), sample_fraction=0.25)]
    assert picked == ["0", "4", "8", "12", "16"]


def test_full_fraction_passes_everything_through():
    assert len(list(_apply_sampling(_rows(), sample_fraction=1.0))) == 20


def test_max_passages_stops_the_stream_early():
    assert [r["pid"] for r in _apply_sampling(_rows(), 1.0, max_passages=3)] == ["0", "1", "2"]


def test_sampling_and_limit_compose():
    picked = [r["pid"] for r in _apply_sampling(_rows(100), sample_fraction=0.1, max_passages=4)]
    assert picked == ["0", "10", "20", "30"]


def test_scan_counts_false_positives_rejected_by_the_question_overlap_stage():
    questions = [{"qid": "yr", "question": "when was puerto rico added to the usa",
                  "answers": ["1950"]}]
    stream = [
        {"pid": "1", "title": "Norway", "text": "The 1950 census of rural Norway."},
        {"pid": "2", "title": "Puerto Rico", "text": "Puerto rico was added to the usa in 1950."},
    ]
    gold, stats = scan_for_gold(stream, AnswerIndex(questions, min_question_overlap=2),
                                progress_every=0, log=lambda *a: None)
    assert gold == {"yr": ["2"]}
    assert stats.passages_with_answer == 1
    assert stats.passages_answer_string_only == 1     # 1번은 정답만 있고 무관
    assert stats.rejected_pairs == 1


# --- 샤드 단위 표본 (전송량 절감) ------------------------------------------

def _plan(files, fraction):
    """plan_parquet_shards 의 선택 로직만 떼어 검증 (네트워크 없이)."""
    if fraction >= 1.0:
        return files, 1.0
    stride = max(1, min(len(files), int(round(1.0 / fraction))))
    selected = files[::stride]
    return selected, len(selected) / len(files)


def test_shard_sampling_reduces_what_has_to_be_downloaded():
    # 행 단위 stride 는 파일을 다 받은 뒤 버리므로 전송량이 줄지 않는다.
    # 샤드를 건너뛰어야 Week 1 의 go/no-go 가 수십 분에 끝난다.
    files = [f"shard-{i:05d}.parquet" for i in range(157)]
    selected, effective = _plan(files, 0.05)
    assert len(selected) == 8
    assert 0.04 < effective < 0.06


def test_selected_shards_span_the_whole_corpus_not_just_the_head():
    files = [f"shard-{i:05d}.parquet" for i in range(157)]
    selected, _ = _plan(files, 0.05)
    assert selected[0] == files[0]
    assert selected[-1] >= files[140]


def test_full_fraction_reads_every_shard():
    files = [f"shard-{i:05d}.parquet" for i in range(157)]
    selected, effective = _plan(files, 1.0)
    assert selected == files and effective == 1.0


def test_effective_fraction_is_reported_because_shards_are_coarse():
    # 157 샤드이므로 입도가 약 0.64% - 요청한 비율과 실제가 다를 수 있고,
    # 외삽이 틀린 분모를 쓰면 usable question 추정이 통째로 어긋난다
    files = [f"shard-{i:05d}.parquet" for i in range(157)]
    _, effective = _plan(files, 0.001)
    assert effective == 1 / 157


def test_a_fraction_finer_than_one_shard_still_reads_one_shard():
    files = [f"shard-{i:05d}.parquet" for i in range(10)]
    selected, _ = _plan(files, 0.0001)
    assert len(selected) == 1


def test_passages_rejected_by_the_gold_cap_are_not_written_out():
    """상한에 걸린 passage 는 하류에서 아무도 안 쓴다 - 쓰면 파일·메모리만 먹는다."""
    questions = [{"qid": "a", "question": "capital of france", "answers": ["Paris"]}]
    stream = [{"pid": str(i), "title": "t", "text": "the capital of france is Paris"}
              for i in range(10)]
    written = []
    gold, stats = scan_for_gold(
        stream, AnswerIndex(questions, min_question_overlap=0),
        max_gold_per_question=3, on_gold_passage=written.append,
        progress_every=0, log=lambda *a: None,
    )
    assert len(gold["a"]) == 3
    assert [p["pid"] for p in written] == ["0", "1", "2"]   # 상한까지만
    assert stats.passages_with_answer == 3
    assert stats.passages_capped_out == 7


def test_a_passage_still_useful_to_one_question_is_kept():
    # 한 질문은 상한에 찼지만 다른 질문에는 아직 필요한 passage 는 남아야 한다
    questions = [
        {"qid": "a", "question": "capital of france", "answers": ["Paris"]},
        {"qid": "b", "question": "where is the eiffel tower", "answers": ["Paris"]},
    ]
    index = AnswerIndex(questions, min_question_overlap=0)
    stream = [{"pid": str(i), "title": "t", "text": "Paris"} for i in range(4)]
    written = []
    gold, stats = scan_for_gold(stream, index, max_gold_per_question=2,
                                on_gold_passage=written.append,
                                progress_every=0, log=lambda *a: None)
    assert len(written) == 2 and stats.passages_capped_out == 2
    assert len(gold["a"]) == len(gold["b"]) == 2


# --- 체크포인트 / 재개 (Week 1) --------------------------------------------

import json as _json

from src.corpus_scan import (
    CHECKPOINT_VERSION, ScanStats as _ScanStats, _fingerprint, _truncate,
    load_checkpoint, save_checkpoint,
)


def _fp(**over):
    base = {"dataset": "repo", "n_shards": 157, "first_shard": "a.parquet",
            "last_shard": "z.parquet", "n_answer_strings": 58740,
            "min_question_overlap": 2, "max_answer_tokens": 8, "max_gold_per_question": 20}
    base.update(over)
    return base


def test_checkpoint_round_trips(tmp_path):
    path = tmp_path / "ckpt.json"
    stats = _ScanStats(passages_seen=1000, gold_pairs=50)
    save_checkpoint(path, 63, {"q1": ["p1", "p2"]}, stats, {"gold": 1234, "random": 56}, _fp())
    shards, gold, restored, offsets = load_checkpoint(path, _fp())
    assert shards == 63
    assert gold == {"q1": ["p1", "p2"]}
    assert restored.passages_seen == 1000
    assert offsets == {"gold": 1234, "random": 56}


def test_no_checkpoint_means_start_from_scratch(tmp_path):
    assert load_checkpoint(tmp_path / "missing.json", _fp()) is None


def test_resume_is_refused_when_the_gold_criterion_changed(tmp_path):
    """판정 기준이 달라진 채 이어붙이면 앞뒤가 다른 산출물이 조용히 만들어진다."""
    path = tmp_path / "ckpt.json"
    save_checkpoint(path, 10, {}, _ScanStats(), {}, _fp(min_question_overlap=2))
    assert load_checkpoint(path, _fp(min_question_overlap=1), log=lambda *a: None) is None


def test_resume_is_refused_when_the_shard_list_changed(tmp_path):
    path = tmp_path / "ckpt.json"
    save_checkpoint(path, 10, {}, _ScanStats(), {}, _fp(n_shards=157))
    assert load_checkpoint(path, _fp(n_shards=200), log=lambda *a: None) is None


def test_a_corrupt_checkpoint_does_not_crash_the_scan(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text("{not json")
    assert load_checkpoint(path, _fp(), log=lambda *a: None) is None


def test_checkpoint_write_is_atomic(tmp_path):
    # 체크포인트를 쓰다가 죽어도 이전 것이 남아야 한다 -> .tmp 후 replace
    path = tmp_path / "ckpt.json"
    save_checkpoint(path, 1, {}, _ScanStats(), {}, _fp())
    save_checkpoint(path, 2, {}, _ScanStats(), {}, _fp())
    assert not (tmp_path / "ckpt.json.tmp").exists()
    assert _json.loads(path.read_text())["shards_done"] == 2


def test_truncation_removes_a_partially_written_shard(tmp_path):
    """샤드 중간에 죽으면 그 샤드를 다시 돌린다 - 중복 레코드가 생기면 안 된다."""
    path = tmp_path / "gold.jsonl"
    path.write_text('{"pid": "1"}\n{"pid": "2"}\n{"pid": "half')
    _truncate(path, len('{"pid": "1"}\n{"pid": "2"}\n'))
    assert path.read_text() == '{"pid": "1"}\n{"pid": "2"}\n'


def test_fingerprint_captures_the_settings_that_must_not_change():
    index = AnswerIndex(QUESTIONS, min_question_overlap=2)
    fp = _fingerprint(index, "repo", 20, 157, "a.parquet", "z.parquet")
    assert fp["min_question_overlap"] == 2
    assert fp["max_gold_per_question"] == 20
    assert fp["n_shards"] == 157


def _write_shards(tmp_path, n_shards=3, rows=40, n_questions=30):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    files = []
    for s in range(n_shards):
        ids, titles, texts = [], [], []
        for r in range(rows):
            n = s * rows + r
            qi = n % n_questions
            ids.append(n)
            titles.append(f"case {qi}")
            texts.append(f"the recorded value for case {qi} is ans{qi}"
                         if n % 3 == 0 else f"unrelated filler text number {n}")
        path = tmp_path / f"shard-{s}.parquet"
        pq.write_table(pa.table({"id": ids, "title": titles, "text": texts}), path)
        files.append(str(path))
    return files


@pytest.fixture
def local_fs(monkeypatch):
    """HfFileSystem 대신 로컬 파일을 열게 해 네트워크 없이 샤드 스캔을 돌린다."""
    import huggingface_hub

    class _LocalFS:
        def open(self, path, mode="rb"):
            return open(path, mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda *a, **k: _LocalFS())


def _run(outdir, files, index, resume):
    from src.corpus_scan import scan_sharded

    outdir.mkdir(parents=True, exist_ok=True)
    return scan_sharded(
        files, index, gold_path=outdir / "gold.jsonl", random_path=outdir / "rand.jsonl",
        checkpoint_path=outdir / "ckpt.json", dataset="test", max_gold_per_question=20,
        random_pool_size=30, resume=resume, log=lambda *a: None,
    )


@pytest.mark.slow
def test_resuming_mid_scan_reproduces_the_uninterrupted_result(tmp_path, local_fs):
    """끊겼다 이어서 돌린 결과가 처음부터 한 번에 돌린 것과 **같아야** 한다.

    수 시간짜리 스캔이 네트워크 한 번 끊겼다고 날아가지 않게 하려고 넣은
    기능이므로, 재개가 산출물을 바꾸면 기능 자체가 무의미하다.
    """
    questions = [{"qid": f"q{i}", "question": f"what is the recorded value for case {i}",
                  "answers": [f"ans{i}"]} for i in range(30)]
    index = AnswerIndex(questions, min_question_overlap=2)
    files = _write_shards(tmp_path)

    full = tmp_path / "full"
    gold_a, stats_a, rand_a = _run(full, files, index, resume=False)

    # 샤드 2개까지 처리한 뒤 죽은 상황을 만든다 (부분 레코드까지 남겨서)
    part = tmp_path / "resumed"
    _run(part, files[:2], index, resume=False)
    with (part / "gold.jsonl").open("a") as fh:
        fh.write('{"pid": "half-written')

    gold_b, stats_b, rand_b = _run(part, files, index, resume=True)

    assert gold_a == gold_b
    assert stats_a.passages_seen == stats_b.passages_seen
    assert rand_a == rand_b
    # 파일까지 바이트 단위로 같아야 한다 - 중복도 누락도 없다는 뜻
    assert (full / "gold.jsonl").read_text() == (part / "gold.jsonl").read_text()
    assert "half-written" not in (part / "gold.jsonl").read_text()


@pytest.mark.slow
def test_a_checkpoint_is_written_after_each_shard(tmp_path, local_fs):
    import json as _j

    questions = [{"qid": "q0", "question": "what is the recorded value for case 0",
                  "answers": ["ans0"]}]
    files = _write_shards(tmp_path, n_shards=3, n_questions=1)
    out = tmp_path / "run"
    _run(out, files, AnswerIndex(questions, min_question_overlap=2), resume=False)
    assert _j.loads((out / "ckpt.json").read_text())["shards_done"] == 3
