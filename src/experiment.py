"""한 run(= depth x seed)의 설정 해석과 데이터 조립 (Subtask 1.3).

한 run 을 정의하는 것은 (depth d, seed s) 두 값뿐이고, 나머지는 전부 동일하다.
이 모듈은 그 두 값으로부터 결정되는 것들을 한 곳에 모은다:

  - seed -> 파티션 배정 (시드 5개에 파티션 2개를 번갈아)
  - depth -> 각 클라이언트의 학습 예제(같은 캐시, 다른 선택)
  - K, S, R -> gate 3 가 남긴 calibrated_schedule.json 에서 주입

Resolving one run (depth x seed) and assembling its data. Only d and s differ
between runs; K, S and R come from the gate-3 artefact rather than convention.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.data import PassageStore, build_example
from src.logging_utils import load_json
from src.nq_data import build_eval_set
from src.retrieval_cache import RetrievalCache, cache_path


@dataclass
class RunSpec:
    depth: int
    seed: int
    partition_seed: int
    local_steps_per_round: int
    total_local_steps: int
    rounds: int

    @property
    def name(self) -> str:
        return f"d{self.depth}_s{self.seed}"


def partition_seed_for(seed: int, partition_seeds) -> int:
    """시드 -> 파티션 배정. 두 배정을 번갈아 써서 결과가 한 파티션에 묶이지 않게 한다.

    Two partitions are assigned across the seeds so the result is not tied to
    one allocation (plan, Subtask 1.1).
    """
    seeds = list(partition_seeds)
    return seeds[(int(seed) - 1) % len(seeds)]


def resolve_schedule(cfg, calibrated_path: str = None) -> Dict[str, int]:
    """K, S, R 을 확정한다. gate 3 산출물이 있으면 그것이 우선한다.

    설정 파일의 값은 null 로 두었다. 라운드 수를 관례로 정하지 않았다는 사실이
    코드에 남아야 하기 때문이다 - 값이 없고 게이트 산출물도 없으면 에러를 낸다.
    """
    path = Path(calibrated_path or Path(cfg.paths.log_dir) / "gates" / "calibrated_schedule.json")
    if path.exists():
        return dict(load_json(path)["train"])

    schedule = {
        "local_steps_per_round": cfg.get_path("train.local_steps_per_round"),
        "total_local_steps": cfg.get_path("train.total_local_steps"),
        "rounds": cfg.get_path("train.rounds"),
    }
    if any(v is None for v in schedule.values()):
        raise RuntimeError(
            "K/S/R are undetermined: run scripts/gate3_local_step_calibration.py first, "
            "or set train.local_steps_per_round / total_local_steps / rounds explicitly "
            "(the plan fixes these by pilot, not by convention)."
        )
    return schedule


def make_run_spec(cfg, depth: int, seed: int, calibrated_path: str = None) -> RunSpec:
    schedule = resolve_schedule(cfg, calibrated_path)
    return RunSpec(
        depth=depth,
        seed=seed,
        partition_seed=partition_seed_for(seed, cfg.partition.partition_seeds),
        local_steps_per_round=int(schedule["local_steps_per_round"]),
        total_local_steps=int(schedule["total_local_steps"]),
        rounds=int(schedule["rounds"]),
    )


class ExperimentData:
    """한 파티션에 대한 동결 산출물(코퍼스, 파티션, 캐시)을 담고,
    depth 별 학습 예제를 만들어 준다. depth 를 바꿔도 **캐시는 같다**."""

    def __init__(self, cfg, partition_seed: int):
        data_dir = Path(cfg.paths.data_dir)
        self.cfg = cfg
        self.partition_seed = partition_seed
        self.store = PassageStore.from_jsonl(data_dir / "corpus.jsonl")
        self.partition = load_json(data_dir / f"partition_{partition_seed}.json")
        self.num_clients = cfg.partition.num_clients
        self._val_qids = None
        # **캐시는 분할 시드와 짝이 맞아야 한다.** 두 분할은 클라이언트마다 다른
        # passage 를 주므로 후보 풀도 다르다. 시드 없이 읽던 동안, 마지막에 돌린
        # 시드의 캐시가 모든 실행에 쓰여서 클라이언트가 자기가 갖지도 않은
        # passage 의 후보 목록을 받았다 - 오류 없이.
        self.caches: Dict[int, RetrievalCache] = {
            cid: RetrievalCache.load(cache_path(cfg.paths.cache_dir, cid, partition_seed))
            for cid in range(self.num_clients)
        }

    def client_questions(self, client_id: int) -> List[dict]:
        return [
            {**q, "client_id": client_id}
            for q in self.partition["client_questions"][str(client_id)]
        ]

    def training_examples(self, client_id: int, depth: int) -> List[dict]:
        """같은 캐시, 다른 depth. 프롬프트에는 항상 3개의 passage 가 들어간다."""
        cache = self.caches[client_id]
        out = []
        for q in self.client_questions(client_id):
            if q["qid"] not in cache:
                continue
            candidates, scores = cache.get(q["qid"])
            out.append(
                build_example(q, candidates, scores, self.store, depth, self.cfg.retrieval.top_k)
            )
        return out

    def eval_questions(self, size: Optional[int] = None) -> List[dict]:
        """고정 held-out 평가셋(= test). 모든 run 에서 동일하도록 run seed 와 무관하게 뽑는다.

        평가 질문은 학습에 쓰이지 않는다: 각 클라이언트에서 균등하게 떼어내
        학습 예제 집합에서 제외한다(held_out_qids).

        그리고 **val(게이트 표본)을 제외한다.** 둘을 같은 80,720 개에서 독립적으로
        뽑으면 24 개가 겹친다(실측). 결과를 내는 모델이 test 를 학습하지는 않지만,
        설정을 정한 질문과 최종 비교에 쓰는 질문이 겹치지 않아야 "세 집합이 분리됐다"
        고 말할 수 있다. 목록은 scripts/make_splits.py 가 data/split_val_qids.json
        에 남긴다 - 파일이 없으면 제외 없이 돌아가고 경고를 찍는다.
        """
        size = size or self.cfg.data.eval_set_size
        pool = []
        for cid in range(self.num_clients):
            pool.extend(self.client_questions(cid))
        return build_eval_set(pool, size=size, exclude=self.val_qids())

    def val_qids(self) -> set:
        """게이트가 본 질문(val). 없으면 빈 집합 + 경고."""
        if self._val_qids is None:
            path = Path(self.cfg.paths.data_dir) / "split_val_qids.json"
            if path.exists():
                self._val_qids = set(load_json(path)["qids"])
            else:
                self._val_qids = set()
                print(f"  WARNING: {path} not found; the evaluation set is not excluding "
                      "the gate (val) questions. Run scripts/make_splits.py so test and "
                      "val are disjoint.")
        return self._val_qids

    def held_out_qids(self, size: Optional[int] = None) -> set:
        return {q["qid"] for q in self.eval_questions(size)}


def split_train_eval(examples: List[dict], held_out: set) -> List[dict]:
    """평가셋 질문을 학습 예제에서 제외 - 누출 방지."""
    return [ex for ex in examples if ex["qid"] not in held_out]
