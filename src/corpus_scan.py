"""psgs_w100(21M) 스트리밍 스캔 - answer-bearing passage 수집 (Week 1 / Subtask 1.1).

계획서 Week 1 의 작업: "scan psgs_w100 for answer-bearing passages; confirm
usable question count". 전체 21M 스캔은 한 번만 수행하면 되고, Week 1 에서는
표본 비율 스캔으로 usable question 수를 추정해 go/no-go 를 빠르게 판단한다
(scripts/scan_corpus.py --sample-fraction).

스캔 결과는 세 가지 산출물을 남긴다:
  1) gold_by_qid     : 질문 -> gold passage id 목록 (Week 2 코퍼스 구성의 씨앗)
  2) random 표본      : **정답을 포함하지 않은** passage 의 균일 무작위 표본.
                        Week 2 코퍼스의 "random Wikipedia remainder" 성분이 된다.
  3) 커버리지 통계    : usable question 수, 질문당 gold 수 분포

2) 가 필요한 이유: 코퍼스를 gold 와 hard negative 로만 채우면 20만 개 전부가
"누군가의 정답을 담은 passage"가 되어, 검색 난이도가 계획서가 의도한 것과
달라진다. random remainder 는 스캔을 한 번 더 돌지 않고 **같은 통과에서**
reservoir sampling 으로 확보한다.

Streaming scan of psgs_w100 to collect answer-bearing passages. The full pass is
run once; Week 1 uses a sampled pass to estimate the usable question count and
make the go/no-go call quickly.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from src.nq_data import AnswerIndex


@dataclass
class ScanStats:
    passages_seen: int = 0
    passages_with_answer: int = 0
    passages_capped_out: int = 0           # gold 였으나 질문별 상한에 걸려 버려짐
    passages_answer_string_only: int = 0   # 1단계는 통과했으나 질문 겹침에서 탈락
    gold_pairs: int = 0
    rejected_pairs: int = 0                # 거짓 양성으로 버린 (질문, passage) 쌍
    questions_covered: int = 0
    answer_free_seen: int = 0
    random_sampled: int = 0
    sample_fraction: float = 1.0
    gold_per_question_histogram: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passages_seen": self.passages_seen,
            "passages_with_answer": self.passages_with_answer,
            "passages_capped_out": self.passages_capped_out,
            "passages_answer_string_only": self.passages_answer_string_only,
            "gold_pairs": self.gold_pairs,
            "rejected_pairs": self.rejected_pairs,
            "questions_covered": self.questions_covered,
            "answer_free_seen": self.answer_free_seen,
            "random_sampled": self.random_sampled,
            "sample_fraction": self.sample_fraction,
            "gold_per_question_histogram": self.gold_per_question_histogram,
        }


class ReservoirSampler:
    """스트림에서 균일 무작위 k 개를 뽑는다 (Algorithm R).

    21M passage 를 메모리에 올릴 수 없고, 전체를 본 뒤에 뽑으려면 한 번 더
    훑어야 한다. reservoir sampling 은 스트림을 한 번만 지나가면서도 **모든
    passage 가 같은 확률로 뽑히도록** 보장한다 - 코퍼스의 random 성분이
    "앞쪽 20만 개"가 되어버리는 것을 막는다(위키 덤프는 알파벳/문서 순이라
    앞쪽만 쓰면 주제가 심하게 치우친다).

    Uniform random sample of k items from a stream in one pass, so the corpus's
    random component is genuinely random rather than "the first k passages",
    which in a Wikipedia dump would be a severely skewed topic slice.
    """

    def __init__(self, size: int, seed: int = 0):
        self.size = max(0, int(size))
        self._rng = random.Random(seed)
        self._items: List[dict] = []
        self.seen = 0

    def offer(self, item: dict) -> None:
        if self.size == 0:
            return
        self.seen += 1
        if len(self._items) < self.size:
            self._items.append(item)
            return
        j = self._rng.randrange(self.seen)
        if j < self.size:
            self._items[j] = item

    @property
    def items(self) -> List[dict]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


def stream_psgs_w100(
    dataset: str = "kenhktsui/wiki_dpr_e5",
    config: Optional[str] = None,
    split: str = "train",
    sample_fraction: float = 1.0,
    max_passages: Optional[int] = None,
    source: str = "parquet",
    columns: Sequence[str] = ("id", "title", "text"),
    files: Optional[Sequence[str]] = None,
) -> Iterator[dict]:
    """psgs_w100 을 스트리밍으로 흘린다 -> {pid, title, text}.

    100-word passage 가 이미 고정되어 있으므로 chunking 단계는 필요 없다
    (계획서 Subtask 1.1).

    표본 비율은 소스마다 적용 지점이 다르다:
      parquet  - 샤드 단위 (전송량이 비율대로 준다)
      datasets - 행 단위  (전송량은 줄지 않는다)

    source="parquet" (기본)
        parquet 파일을 직접 읽으며 **필요한 열만 투영**한다. 정본인 `wiki_dpr`
        은 loading script 기반이라 datasets>=3 에서 로드가 불가능하고
        ("Dataset scripts are no longer supported"), parquet 미러는 e5 임베딩
        열을 함께 들고 있어 그냥 읽으면 쓰지도 않을 벡터로 대역폭의 90%를
        쓴다. 열 투영이 86GB 를 ~8GB 로 줄인다.

    source="datasets"
        datasets<3 환경에서 정본 wiki_dpr 을 그대로 쓰고 싶을 때의 경로.

    Streams psgs_w100. The canonical `wiki_dpr` is a script-based dataset and
    cannot be loaded on datasets>=3, so the default reads a parquet mirror with
    column projection - the mirror carries an unused embedding column that is
    ~90% of the bytes.
    """
    if source == "parquet":
        if files is None:
            files, _ = plan_parquet_shards(dataset, sample_fraction)
        # 표본은 이미 샤드 선택에서 적용되었으므로 행 stride 는 걸지 않는다
        yield from _apply_sampling(_stream_parquet(files, columns), 1.0, max_passages)
    else:
        yield from _apply_sampling(
            _stream_datasets(dataset, config, split), sample_fraction, max_passages
        )


def plan_parquet_shards(repo: str, sample_fraction: float = 1.0) -> tuple:
    """읽을 parquet 샤드 목록과 **실제** 표본 비율을 정한다.

    표본 추출을 행이 아니라 **샤드 단위**로 하는 이유: 행 단위로 건너뛰면
    파일은 이미 다 받은 뒤에 버리는 것이라 전송량도 시간도 거의 줄지 않는다.
    샤드를 건너뛰면 전송량이 같은 비율로 준다 (Week 1 의 go/no-go 추정이
    10시간이 아니라 수십 분에 끝나는 차이).

    샤드는 passage id 순으로 정렬되어 있으므로 일정 간격으로 고르면 코퍼스
    전 구간에 퍼진다. 다만 입도(granularity)가 샤드 하나(약 0.64%)라,
    요청한 비율과 실제 비율이 다를 수 있어 **실제 비율을 함께 돌려준다** -
    usable question 외삽이 틀린 분모를 쓰지 않도록.

    Shard-level sampling, because row-level striding downloads everything and
    throws most of it away. Returns the effective fraction so the extrapolation
    uses the right denominator.
    """
    from huggingface_hub import HfFileSystem

    files = sorted(HfFileSystem().glob(f"datasets/{repo}/**/*.parquet"))
    if not files:
        raise RuntimeError(
            f"no parquet files found in dataset repo '{repo}'. "
            "Set data.passage_source: datasets to use a loading-script dataset instead."
        )
    if sample_fraction >= 1.0:
        return files, 1.0
    stride = max(1, min(len(files), int(round(1.0 / sample_fraction))))
    selected = files[::stride]
    return selected, len(selected) / len(files)


def _stream_parquet(
    files: Sequence[str],
    columns: Sequence[str] = ("id", "title", "text"),
    batch_size: int = 2048,
) -> Iterator[dict]:
    """주어진 parquet 샤드를 순서대로, 지정한 열만 읽어 흘린다."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for path in files:
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
                for row in batch.to_pylist():
                    yield _normalise_row(row)


def _stream_datasets(repo: str, config: Optional[str], split: str) -> Iterator[dict]:
    """datasets 라이브러리 경로 (loading script 를 지원하는 버전에서만)."""
    from datasets import load_dataset

    ds = (
        load_dataset(repo, config, split=split, streaming=True)
        if config
        else load_dataset(repo, split=split, streaming=True)
    )
    for row in ds:
        yield _normalise_row(row)


def _normalise_row(row: dict) -> dict:
    """미러마다 다른 열 이름을 {pid, title, text} 로 통일한다."""
    pid = row.get("id", row.get("_id", row.get("pid")))
    return {
        "pid": str(pid),
        "title": (row.get("title") or "").strip(),
        "text": row["text"],
    }


def _apply_sampling(
    rows: Iterable[dict], sample_fraction: float = 1.0, max_passages: Optional[int] = None
) -> Iterator[dict]:
    """표본 비율(stride)과 개수 상한을 적용한다.

    stride 방식을 쓰는 이유: 스트림 전체를 보기 전에 뽑아야 하므로 무작위
    추출이 불가능하고, 고정 간격이면 적어도 코퍼스 전 구간에 고르게 퍼진다
    ("앞에서 N 개"는 위키 덤프에서 주제가 심하게 치우친다).
    """
    stride = max(1, int(round(1.0 / sample_fraction))) if sample_fraction < 1.0 else 1
    emitted = 0
    for i, row in enumerate(rows):
        if stride > 1 and i % stride != 0:
            continue
        yield row
        emitted += 1
        if max_passages and emitted >= max_passages:
            return


def process_passage(
    passage: dict,
    index: AnswerIndex,
    gold_by_qid: Dict[str, List[str]],
    stats: "ScanStats",
    max_gold_per_question: int = 20,
    on_gold_passage=None,
    random_sampler: "ReservoirSampler" = None,
) -> None:
    """passage 한 건을 판정해 gold / random 중 한쪽으로 보낸다.

    평평한 스트림 스캔(scan_for_gold)과 샤드 단위 스캔(scan_sharded)이 **같은**
    판정 로직을 쓰도록 분리했다. 재개 기능이 붙은 쪽만 다르게 판정하면
    산출물이 조용히 달라진다.
    """
    stats.passages_seen += 1
    hits, raw_hits = index.match_detailed(passage["text"], passage.get("title", ""))
    stats.rejected_pairs += raw_hits - len(hits)
    if raw_hits and not hits:
        # 정답 문자열은 있었지만 질문과 무관한 passage - 거짓 양성
        stats.passages_answer_string_only += 1

    if not hits:
        # 어느 질문의 gold 로도 인정되지 않은 passage 만 random 표본에 들어간다.
        # "정답 문자열이 없다"는 뜻이 아니다 - 실측으로 answer_free_seen 6,371,022 중
        # 6,370,394 건이 정답 문자열을 품고 있고 질문 겹침 2 단계에서 탈락했다.
        # 문자열이 아예 없는 passage 는 21M 중 628 건뿐이다. 그래서 recall 은
        # metrics.gold_recall_at_k 처럼 gold id 소속으로만 재야 한다. 문자열 매칭으로
        # 재면 이 remainder 가 적중으로 집계된다 (실측 과대계상 1.9%, 단답 6.6%).
        stats.answer_free_seen += 1
        if random_sampler is not None:
            random_sampler.offer(passage)
        return

    recorded = False
    for qid in hits:
        bucket = gold_by_qid.setdefault(qid, [])
        if len(bucket) < max_gold_per_question:
            bucket.append(passage["pid"])
            stats.gold_pairs += 1
            recorded = True

    # **실제로 기록된 passage 만** 본문을 내보낸다. 상한에 걸려 버려진 passage 는
    # 하류에서 아무도 참조하지 않으므로, 쓰면 파일과 메모리만 먹는다.
    if recorded:
        stats.passages_with_answer += 1
        if on_gold_passage is not None:
            on_gold_passage(passage)
    else:
        stats.passages_capped_out += 1


def scan_for_gold(
    passages: Iterable[dict],
    index: AnswerIndex,
    max_gold_per_question: int = 20,
    progress_every: int = 250_000,
    on_gold_passage=None,
    random_sampler: "ReservoirSampler" = None,
    log=print,
) -> tuple:
    """passage 스트림을 훑어 질문별 gold passage id 를 모은다.

    max_gold_per_question 은 흔한 정답(예: "1999")을 가진 질문이 코퍼스를
    독차지하는 것을 막는다 - Week 2 의 200k 코퍼스 예산을 지키기 위한 상한.

    max_gold_per_question caps questions with very common answer strings so a
    handful of them cannot dominate Week 2's 200k corpus budget.
    """
    gold_by_qid: Dict[str, List[str]] = defaultdict(list)
    stats = ScanStats()

    for passage in passages:
        process_passage(passage, index, gold_by_qid, stats,
                        max_gold_per_question, on_gold_passage, random_sampler)
        if progress_every and stats.passages_seen % progress_every == 0:
            log(
                f"  scanned {stats.passages_seen:,} passages | "
                f"{len(gold_by_qid):,} questions covered | "
                f"{len(random_sampler) if random_sampler is not None else 0:,} random sampled"
            )

    stats.questions_covered = len(gold_by_qid)
    stats.random_sampled = len(random_sampler) if random_sampler is not None else 0
    histogram: Dict[str, int] = defaultdict(int)
    for golds in gold_by_qid.values():
        bucket = "1" if len(golds) == 1 else "2-5" if len(golds) <= 5 else "6-20"
        histogram[bucket] += 1
    stats.gold_per_question_histogram = dict(histogram)
    return dict(gold_by_qid), stats


def extrapolate_usable(stats: ScanStats, total_questions: int) -> dict:
    """표본 스캔에서 전체 usable question 수를 추정한다.

    한 질문의 gold 가 코퍼스에 균일하게 흩어져 있다고 보면, 표본 비율 f 로
    스캔했을 때 관측된 커버리지는 하한이다. 관측 커버리지와, 질문당 gold 가
    Poisson(lambda) 이라 볼 때의 보정치를 함께 보고한다.

    Estimate the full-scan usable-question count from a sampled pass. The
    observed coverage is a lower bound; a Poisson correction for per-question
    gold counts is reported alongside it so the Week-1 go/no-go call is not made
    on the lower bound alone.
    """
    f = stats.sample_fraction
    observed = stats.questions_covered
    if f >= 1.0:
        return {
            "observed_usable": observed,
            "estimated_usable": observed,
            "coverage_rate": observed / total_questions if total_questions else 0.0,
        }
    # P(적어도 하나 관측) = 1 - exp(-lambda f)  ->  lambda 를 역산해 f=1 로 환산
    import math

    rate = observed / total_questions if total_questions else 0.0
    rate = min(rate, 1 - 1e-9)
    lam = -math.log(1 - rate) / f if rate > 0 else 0.0
    estimated_rate = 1 - math.exp(-lam)
    return {
        "observed_usable": observed,
        "estimated_usable": int(round(estimated_rate * total_questions)),
        "coverage_rate": rate,
        "estimated_coverage_rate": estimated_rate,
        "implied_gold_per_question": lam,
    }


def save_gold_map(gold_by_qid: Dict[str, List[str]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(gold_by_qid, fh)


def load_gold_map(path: str) -> Dict[str, List[str]]:
    with Path(path).open() as fh:
        return json.load(fh)


def save_passages(passages: Iterable[dict], path: str) -> int:
    """passage 레코드를 JSONL 로. gold / random 표본 모두 같은 형식이다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as fh:
        for passage in passages:
            fh.write(json.dumps(passage) + "\n")
            n += 1
    return n


# ============================================================================
# 샤드 단위 스캔 + 재개 (Week 1)
#
# 21M 스캔은 수 시간짜리인데 HF Hub 네트워크는 간헐적으로 끊긴다. 재시도를
# 다 소진하면 프로그램이 죽고, 재개 기능이 없으면 **몇 시간치가 통째로**
# 날아간다. 그래서 샤드 하나를 끝낼 때마다 체크포인트를 남긴다.
#
# 설계상 두 가지를 바꿨다:
#  1) random 표본을 전역 reservoir 가 아니라 **샤드별 할당량**으로 뽑는다.
#     전역 reservoir(30만 x 660B = 200MB)를 샤드마다 저장하면 쓰기 비용이
#     너무 크고, 샤드마다 같은 수를 뽑는 층화 추출이 오히려 분산이 작다.
#     샤드는 행 수가 같으므로 균일성도 유지된다.
#  2) 출력 파일은 append 로 쓰고, 체크포인트에 **바이트 오프셋**을 남긴다.
#     샤드 중간에 죽으면 그 오프셋까지 잘라내고 해당 샤드를 다시 돌린다 -
#     그래야 중복 레코드가 생기지 않는다.
#
# Shard-level checkpointing so a multi-hour scan survives a dropped connection.
# ============================================================================

CHECKPOINT_VERSION = 2


def _fingerprint(index: AnswerIndex, dataset: str, max_gold_per_question: int,
                 n_shards: int, first_shard: str, last_shard: str) -> dict:
    """재개가 **같은 조건**에서만 이어지도록 하는 지문.

    판정 기준이나 데이터셋이 바뀐 채로 이어붙이면 앞뒤가 다른 산출물이
    조용히 만들어진다. 하나라도 다르면 재개를 거부한다.
    """
    return {
        "dataset": dataset,
        "n_shards": n_shards,
        "first_shard": first_shard,
        "last_shard": last_shard,
        "n_answer_strings": len(index),
        "min_question_overlap": index.min_question_overlap,
        "max_answer_tokens": index.max_answer_tokens,
        "max_gold_per_question": max_gold_per_question,
    }


def save_checkpoint(path, shards_done: int, gold_by_qid: Dict[str, List[str]],
                    stats: ScanStats, offsets: Dict[str, int], fingerprint: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "shards_done": shards_done,
        "offsets": offsets,
        "fingerprint": fingerprint,
        "stats": stats.to_dict(),
        "gold_by_qid": gold_by_qid,
    }
    # 원자적 교체: 체크포인트를 쓰다가 죽으면 이전 것이 남아야 한다
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)


def load_checkpoint(path, fingerprint: dict, log=print):
    """체크포인트를 읽어 (shards_done, gold_by_qid, stats, offsets) 를 돌려준다.

    없거나 조건이 다르면 None - 호출부는 처음부터 시작한다.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception as exc:
        log(f"  checkpoint unreadable ({exc}); starting from scratch")
        return None

    if payload.get("version") != CHECKPOINT_VERSION:
        log("  checkpoint was written by a different version; starting from scratch")
        return None

    stored = payload.get("fingerprint", {})
    differences = [k for k, v in fingerprint.items() if stored.get(k) != v]
    if differences:
        log(f"  checkpoint does not match the current settings ({', '.join(differences)}); "
            "starting from scratch rather than stitching two different scans together")
        return None

    stats = ScanStats(**{k: v for k, v in payload["stats"].items()
                         if k in ScanStats.__dataclass_fields__})
    return (
        payload["shards_done"],
        {k: list(v) for k, v in payload["gold_by_qid"].items()},
        stats,
        payload.get("offsets", {}),
    )


def _truncate(path, offset: int) -> None:
    """체크포인트 시점까지 잘라낸다 - 샤드 중간에 죽어 생긴 부분 레코드 제거."""
    p = Path(path)
    if p.exists():
        with p.open("r+b") as fh:
            fh.truncate(offset)


def scan_sharded(
    files: Sequence[str],
    index: AnswerIndex,
    gold_path,
    random_path,
    checkpoint_path,
    dataset: str = "",
    columns: Sequence[str] = ("id", "title", "text"),
    max_gold_per_question: int = 20,
    random_pool_size: int = 300_000,
    resume: bool = True,
    checkpoint_every: int = 1,
    seed: int = 0,
    progress_every: int = 250_000,
    log=print,
) -> tuple:
    """parquet 샤드를 하나씩 처리하고 샤드 경계마다 체크포인트를 남긴다.

    반환: (gold_by_qid, stats, random_written)
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    files = list(files)
    fingerprint = _fingerprint(
        index, dataset, max_gold_per_question, len(files),
        Path(files[0]).name if files else "", Path(files[-1]).name if files else "",
    )

    start_shard, gold_by_qid, stats = 0, {}, ScanStats()
    offsets = {"gold": 0, "random": 0}

    restored = load_checkpoint(checkpoint_path, fingerprint, log) if resume else None
    if restored:
        start_shard, gold_by_qid, stats, offsets = restored
        _truncate(gold_path, offsets.get("gold", 0))
        _truncate(random_path, offsets.get("random", 0))
        log(f"  resuming after shard {start_shard}/{len(files)} "
            f"({stats.passages_seen:,} passages already scanned, "
            f"{len(gold_by_qid):,} questions covered)")
    else:
        Path(gold_path).parent.mkdir(parents=True, exist_ok=True)
        Path(gold_path).write_text("")
        Path(random_path).write_text("")

    # 샤드마다 같은 수를 뽑는 층화 추출. 전역 reservoir 를 매 샤드 저장하는
    # 비용(200MB x 157)을 피하면서 균일성은 유지한다.
    per_shard_quota = max(1, random_pool_size // max(1, len(files)))
    random_written = 0

    fs = HfFileSystem()
    gold_fh = Path(gold_path).open("a")
    random_fh = Path(random_path).open("a")
    try:
        for shard_i in range(start_shard, len(files)):
            sampler = ReservoirSampler(per_shard_quota, seed=seed + shard_i)
            with fs.open(files[shard_i], "rb") as handle:
                parquet = pq.ParquetFile(handle)
                for batch in parquet.iter_batches(batch_size=2048, columns=list(columns)):
                    for row in batch.to_pylist():
                        process_passage(
                            _normalise_row(row), index, gold_by_qid, stats,
                            max_gold_per_question,
                            lambda p: gold_fh.write(json.dumps(p) + "\n"),
                            sampler,
                        )
                        if progress_every and stats.passages_seen % progress_every == 0:
                            log(f"  scanned {stats.passages_seen:,} passages | "
                                f"{len(gold_by_qid):,} questions covered | "
                                f"shard {shard_i + 1}/{len(files)}")

            for item in sampler.items:
                random_fh.write(json.dumps(item) + "\n")
                random_written += 1
            stats.random_sampled = random_written

            gold_fh.flush()
            random_fh.flush()
            if checkpoint_every and (shard_i + 1) % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path, shard_i + 1, gold_by_qid, stats,
                    {"gold": gold_fh.tell(), "random": random_fh.tell()}, fingerprint,
                )
    finally:
        gold_fh.close()
        random_fh.close()

    stats.questions_covered = len(gold_by_qid)
    histogram: Dict[str, int] = defaultdict(int)
    for golds in gold_by_qid.values():
        bucket = "1" if len(golds) == 1 else "2-5" if len(golds) <= 5 else "6-20"
        histogram[bucket] += 1
    stats.gold_per_question_histogram = dict(histogram)
    return gold_by_qid, stats, random_written
