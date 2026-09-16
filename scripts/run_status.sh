#!/usr/bin/env bash
# 진행 중이거나 끝난 run 들의 상태를 한 눈에.
#
# 로그만 보면 안 되는 이유: 라운드 기록은 jsonl 에 쌓이고 터미널에는 거의 안 나온다.
# wall_clock_seconds 는 **누적**이므로 거기서 ETA 를 바로 뽑을 수 있다.
#
#   bash scripts/run_status.sh
#   watch -n 60 'bash ~/rag-fl/scripts/run_status.sh'
set -u
cd "$(dirname "$0")/.."

python3 - <<'INNER'
import glob, json, os, subprocess

paths = sorted(glob.glob("results/logs/runs/*.jsonl"))
if not paths:
    print("  results/logs/runs/ 에 run 이 없다")
    raise SystemExit(0)

alive = subprocess.run(["pgrep", "-f", "scripts/run_single_run.py"],
                       capture_output=True).returncode == 0
stale = []

print(f"  {'run':<10} {'라운드':>9} {'경과':>8} {'ETA':>8} {'F1':>7} "
      f"{'promo':>6} {'rank':>6} {'drift':>7} {'bytes':>9}  상태")
for p in paths:
    rows = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    meta = next((r for r in rows if "run" in r and r.get("event") != "round"), {})
    rounds = [r for r in rows if r.get("event") == "round"]
    summary = next((r for r in rows if r.get("event") == "summary"), None)
    name = meta.get("run") or os.path.basename(p)[:-6]
    total = meta.get("rounds", 0)
    done = max((r["round"] for r in rounds), default=-1)
    el = max((r.get("wall_clock_seconds") or 0 for r in rounds), default=0)
    trained = max(0, done)                      # 라운드 0 은 학습 전 평가
    eta = (el / trained * (total - trained)) if trained and total else 0
    evald = [r for r in rounds if r.get("evaluated")]
    f1 = evald[-1].get("f1") if evald else None
    last = rounds[-1] if rounds else {}

    # 프로세스가 없는데 summary 도 없으면 "진행" 이 아니라 중단이다.
    # 0/59 에 멈춘 run 을 진행으로 찍으면 죽은 것을 모른다.
    if summary:
        state = "완료"
    elif alive:
        state = "진행"
    else:
        state = "중단"
        stale.append(p)

    # 합성 리허설이 경로 분리 전 코드로 운영 경로에 쓴 흔적
    if total and total < 20 and not meta.get("payload_mb"):
        state += " (합성)"

    def fmt(v, w, d=3):
        return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

    print(f"  {name:<10} {f'{done}/{total}':>9} {el/60:>7.1f}m "
          f"{(0 if summary else eta/60):>7.1f}m {fmt(f1, 7, 2)} "
          f"{fmt(last.get('promotion_rate'), 6)} {fmt(last.get('rank_change_rate'), 6)} "
          f"{fmt(last.get('drift_divergence'), 7)} "
          f"{last.get('cumulative_bytes', 0)/1e9:>8.2f}G  {state}")
    if summary:
        print(f"    최종 F1 {summary['final']['f1']:.2f} EM {summary['final']['em']:.2f} "
              f"| 총 {summary['wall_clock_seconds']/60:.1f}분 "
              f"| 20회 추정 {summary['wall_clock_seconds']*20/3600:.1f}시간")

if stale:
    print()
    print("  미완 run 파일이 남아 있다. run_single_run 은 jsonl 이 있으면 건너뛰므로")
    print("  (Week 4 의 중단-재개 기능) 지워야 다시 돈다:")
    print("    rm -f " + " ".join(stale))
INNER

echo
RUN=$(pgrep -af "scripts/run_single_run.py" 2>/dev/null | grep -v grep | head -1 || true)
if [ -n "${RUN:-}" ]; then
  PID=${RUN%% *}
  printf "  실행 중  pid %-7s 경과 %-9s CPU %-9s 메모리 %s\n" "$PID" \
    "$(ps -o etime= -p "$PID" | tr -d ' ')" "$(ps -o time= -p "$PID" | tr -d ' ')" \
    "$(ps -o rss= -p "$PID" | awk '{printf "%.1fGB", $1/1048576}')"
else
  echo "  실행 중인 run 없음"
fi

for f in w3_*.log; do
  [ -f "$f" ] || continue
  # grep -c 는 0 건일 때 "0" 을 찍고 exit 1 을 내므로 `|| echo 0` 을 붙이면
  # 출력이 "0\n0" 이 되어 [ 가 integer expected 로 터진다.
  N=$(grep -cE "Traceback|Error|failed" "$f" 2>/dev/null | head -1)
  if [ "${N:-0}" -gt 0 ] 2>/dev/null; then
    echo "  ⚠ $f 에 오류 ${N}건:"
    grep -E "Traceback|Error|failed" "$f" | tail -2 | sed 's/^/      /'
  fi
done
