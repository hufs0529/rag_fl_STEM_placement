#!/usr/bin/env python3
"""Week 4: 20 run 사후 무결성 점검 (Subtask 1.3).

분석에 들어가기 전에 "정말 한 축만 달랐는가"를 확인한다. 확인 항목:

  - 통신 예산(R, N, payload, total bytes)이 모든 run 에서 동일한가
  - 평가 라운드 스케줄이 동일한가 (보간이 같은 지지점 위에 있어야 한다)
  - 어떤 run 도 조기 종료하지 않았는가
  - 처리가 조건별로 **실제로 다르게** 걸렸는가 (promotion_rate)
  - 곡선에 결측 평가 지점이 없는가

Usage:
  python scripts/check_runs.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.logging_utils import banner, read_jsonl, save_json
from src.run_matrix import check_integrity, enumerate_runs, group_by_depth, is_complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--calibrated", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    specs = enumerate_runs(cfg, args.calibrated)
    banner("Week 4 - run integrity check")

    integrity = check_integrity(cfg, specs)
    print(f"  complete runs: {integrity['runs_complete']}/{integrity['runs_expected']}")
    print(f"  identical budget: {integrity['identical_budget']}")
    for issue in integrity["issues"]:
        print(f"    ISSUE {issue['field']}: {issue['problem']}")

    # 처리 진단: depth 가 커질수록 promotion_rate 가 커져야 하고, d in {0,3} 은 0 이어야 한다
    treatment = {}
    for depth, group in sorted(group_by_depth(specs).items()):
        rates, missing = [], 0
        for spec in group:
            path = Path(cfg.paths.log_dir) / "runs" / f"{spec.name}.jsonl"
            if not is_complete(cfg, spec) or not path.exists():
                missing += 1
                continue
            rounds = [r for r in read_jsonl(path) if r.get("event") == "round" and r["round"] > 0]
            if rounds:
                rates.append(rounds[-1].get("promotion_rate", 0.0))
        treatment[depth] = {
            "mean_promotion_rate": sum(rates) / len(rates) if rates else None,
            "runs": len(rates),
            "missing": missing,
        }

    problems = []
    for depth, info in treatment.items():
        rate = info["mean_promotion_rate"]
        if rate is None:
            continue
        if depth in (0, 3) and rate != 0.0:
            problems.append(f"d={depth} control promoted passages (rate {rate:.4f})")
        if depth > 3 and rate == 0.0:
            problems.append(f"d={depth} promoted nothing; the treatment did not take effect")
    ordered = [treatment[d]["mean_promotion_rate"] for d in sorted(treatment)
               if treatment[d]["mean_promotion_rate"] is not None]
    if ordered != sorted(ordered):
        problems.append("promotion rate is not monotone in depth")

    print("\n  treatment diagnostics (promotion rate = fraction of prompt passages "
          "dense retrieval would not have selected):")
    for depth in sorted(treatment):
        info = treatment[depth]
        rate = info["mean_promotion_rate"]
        print(f"    d={depth:>2}  {('%.4f' % rate) if rate is not None else '   n/a'}  "
              f"({info['runs']} runs, {info['missing']} missing)")

    report = {"integrity": integrity, "treatment": treatment, "problems": problems}
    save_json(report, Path(cfg.paths.log_dir) / "week4" / "run_checks.json")

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  all checks passed - one axis varied, budget identical")
    return 0 if not problems and integrity["identical_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
