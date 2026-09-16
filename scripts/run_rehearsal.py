#!/usr/bin/env python3
"""Week 3: 축소 규모 리허설 - 본실험 전에 배선을 전부 통과시킨다.

리허설이 확인하는 것(계획서 Week 3 "reduced-scale rehearsal"):
  1) 캐시 -> depth 선택 -> 프롬프트 -> answer-token 마스킹 경로가 살아 있는가
  2) 조건 간 차이가 **실제로 걸리는가** (rank-change / promotion > 0 at d>3)
  3) 통신 바이트 회계가 R x N x |AB| x 2 와 일치하는가
  4) drift 와 곡선이 로그에 남는가
  5) 평가 컨텍스트가 모든 조건에서 d=0 으로 고정되어 있는가

결과의 숫자 자체는 의미가 없다(합성 산출물 또는 축소 스케일). 통과 여부만 본다.

Usage:
  python scripts/run_rehearsal.py --dev --synthetic
  python scripts/run_rehearsal.py --dev --backend inprocess
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.experiment import make_run_spec
from src.fl_runner import run_experiment
from src.logging_utils import banner, read_jsonl, save_json
from src.models import payload_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--backend", choices=["flower", "inprocess"], default="inprocess")
    parser.add_argument("--synthetic", action="store_true",
                        help="합성 산출물을 먼저 생성 (네트워크 없이 배선만 점검)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    cfg = config_from_args(args)

    if args.synthetic:
        import subprocess
        here = Path(__file__).resolve().parent
        forwarded = ["--config", args.config] + (["--dev"] if args.dev else [])
        subprocess.run([sys.executable, str(here / "make_dev_artefacts.py")] + forwarded, check=True)

    checks, summaries = [], {}
    for depth in cfg.treatment.depths:
        banner(f"rehearsal | depth {depth}")
        spec = make_run_spec(cfg, depth, args.seed)
        summary = run_experiment(cfg, spec, backend=args.backend, device=args.device)
        summaries[depth] = summary

        records = read_jsonl(Path(cfg.paths.log_dir) / "runs" / f"{spec.name}.jsonl")
        rounds = [r for r in records if r.get("event") == "round" and r["round"] > 0]
        evaluated = [r for r in rounds if r.get("evaluated")]

        expected = spec.rounds * cfg.partition.num_clients * payload_bytes(
            summary["trainable_parameters"], cfg.model.payload_dtype
        ) * 2
        checks.append({
            "depth": depth,
            "rounds_logged": len(rounds) == spec.rounds,
            "curve_points": len(evaluated),
            "bytes_match_formula": rounds[-1]["cumulative_bytes"] == expected,
            "drift_logged": all("drift_divergence" in r for r in rounds),
            "treatment_took_effect": (
                rounds[-1]["promotion_rate"] > 0 if depth > 3 else True
            ),
            "control_promotes_nothing": (
                rounds[-1]["promotion_rate"] == 0 if depth in (0, 3) else True
            ),
        })

    # 평가 컨텍스트가 조건 간 동일한지: eval recall@3 이 모든 depth 에서 같아야 한다
    eval_recalls = {
        d: s["final"].get("eval_gold_recall_at_3") for d, s in summaries.items()
    }
    recall_values = {round(v, 6) for v in eval_recalls.values() if v is not None}
    eval_context_fixed = len(recall_values) <= 1

    report = {
        "backend": args.backend,
        "checks": checks,
        "eval_gold_recall_by_depth": eval_recalls,
        "eval_context_fixed_at_d0": eval_context_fixed,
        "passed": all(all(v is True for k, v in c.items() if k != "depth" and isinstance(v, bool))
                      for c in checks) and eval_context_fixed,
    }
    save_json(report, Path(cfg.paths.log_dir) / "week3" / "rehearsal.json")

    banner("rehearsal checks")
    for c in checks:
        print(f"  depth {c['depth']:>2}: " + "  ".join(
            f"{k}={v}" for k, v in c.items() if k != "depth"))
    print(f"\n  evaluation context fixed at d=0 across conditions: {eval_context_fixed}")
    print(f"  {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
