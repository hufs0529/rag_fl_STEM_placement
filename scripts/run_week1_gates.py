#!/usr/bin/env python3
"""Week 1 게이트 3종을 순서대로 돌리고 go/no-go 를 한 파일로 정리.

게이트는 서로 의존한다: (i) 은 학습 불필요, (ii) 는 retrieval 만 필요하고
gate 3 가 쓸 스크래치 캐시를 남기며, (iii) 은 그 캐시 위에서 짧은 FL 프로브를
돌린다. 따라서 순서대로 돌리고, 실패해도 나머지를 계속 돌려 전체 그림을
남긴다(--stop-on-fail 로 조기 중단 가능).

Runs the three Week-1 gates in order and writes a single go/no-go summary.

Usage:
  python scripts/run_week1_gates.py
  python scripts/run_week1_gates.py --dev --stop-on-fail
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.gates import GateResult, summarise_gates
from src.logging_utils import banner, load_json, save_json

GATES = [
    ("gate1_capability_headroom.py", "gate1_headroom.json"),
    ("gate2_depth_monotonicity.py", "gate2_monotonicity.json"),
    ("gate3_local_step_calibration.py", "gate3_local_steps.json"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--stop-on-fail", action="store_true")
    args, passthrough = parser.parse_known_args()
    cfg = config_from_args(args)

    here = Path(__file__).resolve().parent
    log_dir = Path(cfg.paths.log_dir) / "gates"
    forwarded = ["--config", args.config] + (["--dev"] if args.dev else [])
    for override in args.overrides:
        forwarded += ["--set", override]

    results = []
    for script, output in GATES:
        banner(f"running {script}")
        proc = subprocess.run([sys.executable, str(here / script)] + forwarded + passthrough)
        path = log_dir / output
        if path.exists():
            results.append(GateResult(**load_json(path)))
        if proc.returncode != 0 and args.stop_on_fail:
            print(f"\n{script} failed and --stop-on-fail was set; stopping here.")
            break

    summary = summarise_gates(results)
    save_json(summary, log_dir / "week1_summary.json")

    banner("Week 1 gate summary")
    for result in results:
        print("  " + result.report())
    print(f"\n  {summary['verdict']}")
    return 0 if summary["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
