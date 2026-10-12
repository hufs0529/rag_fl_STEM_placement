"""본실험 run 의 목록·상태·무결성 (Subtask 1.3).

4 depth x 4 seed = 16 run. 각 run 은 **동일한 통신 예산**을 소모하며 early
stopping 이 없다. 이 모듈은 세 가지를 담당한다:

  1) 실행 목록 생성 (seed 를 바깥 루프로 두어, 중간에 멈춰도 시드별로
     네 조건이 고르게 채워지도록)
  2) 재개(resume) - 이미 완료된 run 은 건너뛴다
  3) 무결성 점검 - 전 run 이 정말 같은 조건에서 돌았는지 사후 검증

시드 수가 짝수인 이유는 configs/experiment_config.yaml 의 treatment.seeds
주석에 있다 - 두 토픽 파티션을 고르게 쓰기 위해서다.

Enumeration, resumption and integrity checking for the main runs.
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
    """한 run 의 학습 곡선: 평가된 라운드만, (round, bytes, f1, em).

    **라운드 번호로 묶고 나중 기록을 쓴다.** 파일 순서를 그대로 믿으면, 두 번의
    시도가 쌓인 로그에서 곡선이 0..7, 0..59 로 되돌아가고 rounds_to_target 이
    "라운드 0 에서 목표 달성" 같은 값을 낸다. fl_runner 는 이제 run 로그를 새로
    열지만(fresh=True), 그 전에 쌓인 파일도 읽을 수 있어야 한다.
    """
    path = Path(cfg.paths.log_dir) / "runs" / f"{spec.name}.jsonl"
    by_round: Dict[int, dict] = {}
    for r in read_jsonl(path):
        if r.get("event") == "round" and r.get("evaluated"):
            by_round[r["round"]] = {
                "round": r["round"],
                "cumulative_bytes": r["cumulative_bytes"],
                "f1": r["f1"],
                "em": r["em"],
            }
    return [by_round[k] for k in sorted(by_round)]


def check_integrity(cfg, specs: Sequence[RunSpec]) -> dict:
    """전 run 이 같은 조건이었는지 사후 검증 (Subtask 1.3).

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
                  "local_steps_per_round", "total_bytes",
                  "batch_size", "grad_accumulation", "examples_per_step"):
        values = collect(field)
        if len(values) > 1:
            issues.append({"field": field, "values": sorted(values),
                           "problem": "differs across runs; the budget is not identical"})
        # 일부 run 에만 있는 필드는 "같다"고 말할 수 없다. 필드가 생기기 전에 돌린
        # run 이 섞여 있으면 그 run 의 설정은 기록에 없으므로 검증 불가다.
        missing = [name for name, s in summaries.items() if field not in s]
        if values and missing:
            issues.append({"field": field, "missing_in": sorted(missing),
                           "problem": "not recorded for every run; these runs predate the field "
                                      "and their setting cannot be verified - redo them"})

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
