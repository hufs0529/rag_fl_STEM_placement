"""본실험 20 run 의 목록·상태·무결성 (Subtask 1.3).

4 depth x 5 seed = 20 run. 각 run 은 **동일한 통신 예산**을 소모하며 early
stopping 이 없다. 이 모듈은 세 가지를 담당한다:

  1) 실행 목록 생성 (depth 를 바깥 루프로 두어, 중간에 멈춰도 시드별로
     조건이 고르게 채워지도록)
  2) 재개(resume) - 이미 완료된 run 은 건너뛴다
  3) 무결성 점검 - 20 run 이 정말 같은 조건에서 돌았는지 사후 검증

Enumeration, resumption and integrity checking for the 20 main runs.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from src.experiment import RunSpec, make_run_spec
from src.logging_utils import load_json, read_jsonl


def enumerate_runs(cfg, calibrated_path: str = None) -> List[RunSpec]:
    """실행 순서: seed 를 바깥, depth 를 안쪽에 둔다.

    이렇게 하면 중간에 중단되어도 "시드 1..k 에 대해 네 조건이 모두 있는"
    상태가 되어, 부분 결과만으로도 조건 간 비교가 성립한다. depth 를 바깥에
    두면 중단 시 baseline 만 5개 있고 treatment 는 0개인 쓸모없는 상태가 된다.

    Seeds outer, depths inner: an interrupted sweep then still has all four
    conditions for the seeds it finished, so a partial result is still a
    comparison rather than five baselines and nothing to compare them to.
    """
    return [
        make_run_spec(cfg, depth, seed, calibrated_path)
        for seed in cfg.treatment.seeds
        for depth in cfg.treatment.depths
    ]


def summary_path(cfg, spec: RunSpec) -> Path:
    return Path(cfg.paths.log_dir) / "runs" / f"{spec.name}_summary.json"


def is_complete(cfg, spec: RunSpec) -> bool:
    path = summary_path(cfg, spec)
    if not path.exists():
        return False
    try:
        return bool(load_json(path).get("completed"))
    except Exception:
        return False


def pending(cfg, specs: Sequence[RunSpec]) -> List[RunSpec]:
    return [spec for spec in specs if not is_complete(cfg, spec)]


def load_curve(cfg, spec: RunSpec) -> List[dict]:
    """한 run 의 학습 곡선: 평가된 라운드만, (round, bytes, f1, em)."""
    path = Path(cfg.paths.log_dir) / "runs" / f"{spec.name}.jsonl"
    return [
        {
            "round": r["round"],
            "cumulative_bytes": r["cumulative_bytes"],
            "f1": r["f1"],
            "em": r["em"],
        }
        for r in read_jsonl(path)
        if r.get("event") == "round" and r.get("evaluated")
    ]


def check_integrity(cfg, specs: Sequence[RunSpec]) -> dict:
    """20 run 이 같은 조건이었는지 사후 검증 (Subtask 1.3).

    조건 간 비교가 성립하려면 다음이 모두 같아야 한다:
      - 라운드 수 R, 클라이언트 수 N, payload 크기 -> 통신 예산이 동일
      - 평가 라운드 스케줄 -> 보간이 같은 지지점 위에서 이루어짐
      - 평가셋 크기
    그리고 조건별로 달라야 하는 것: 처리 진단(promotion_rate).
    """
    summaries: Dict[str, dict] = {}
    for spec in specs:
        path = summary_path(cfg, spec)
        if path.exists():
            summaries[spec.name] = load_json(path)

    def collect(field):
        return {s[field] for s in summaries.values() if field in s}

    issues = []
    for field in ("rounds", "num_clients", "payload_bytes_per_client_per_direction",
                  "local_steps_per_round", "total_bytes"):
        values = collect(field)
        if len(values) > 1:
            issues.append({"field": field, "values": sorted(values),
                           "problem": "differs across runs; the budget is not identical"})

    schedules = {tuple(s["eval_rounds"]) for s in summaries.values() if "eval_rounds" in s}
    if len(schedules) > 1:
        issues.append({"field": "eval_rounds",
                       "problem": "curves were evaluated at different rounds; "
                                  "interpolation would not be comparable"})

    if any(s.get("early_stopping") for s in summaries.values()):
        issues.append({"field": "early_stopping",
                       "problem": "a run stopped early; the fixed-budget protocol is broken"})

    return {
        "runs_expected": len(specs),
        "runs_complete": len(summaries),
        "identical_budget": not issues,
        "issues": issues,
        "total_bytes": sorted(collect("total_bytes")),
    }


def group_by_depth(specs: Sequence[RunSpec]) -> Dict[int, List[RunSpec]]:
    out: Dict[int, List[RunSpec]] = {}
    for spec in specs:
        out.setdefault(spec.depth, []).append(spec)
    return out
