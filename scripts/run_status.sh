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
    # ETA 는 **연속 라운드의 차이**로 낸다. el/round 로 나누면 라운드 1 에 섞인
    # 일회성 비용(모델 적재, 라운드 0 평가)이 전 라운드에 퍼져 과대추정된다
    # (실측: 라운드 1 에서 247 분으로 나왔으나 실제는 약 75 분).
    # 그리고 평가는 매 라운드가 아니다 - 앞 10 라운드는 매번, 그 뒤는 5 마다다.
    # 평가 있는 라운드와 없는 라운드의 소요를 따로 재서 남은 일정에 적용한다.
    by_round = {r["round"]: r for r in rounds}
    ev_dt, tr_dt = [], []
    for rn in sorted(by_round):
        prev = by_round.get(rn - 1)
        if rn < 1 or prev is None:
            continue
        a, b = prev.get("wall_clock_seconds"), by_round[rn].get("wall_clock_seconds")
        if a is None or b is None:
            continue
        (ev_dt if by_round[rn].get("evaluated") else tr_dt).append(b - a)
    sched = set(meta.get("eval_rounds") or [])
    remaining = [r for r in range(done + 1, (total or 0) + 1)]
    if ev_dt or tr_dt:
        ev = sum(ev_dt) / len(ev_dt) if ev_dt else (sum(tr_dt) / len(tr_dt))
        tr = sum(tr_dt) / len(tr_dt) if tr_dt else ev
        eta = sum(ev if (not sched or r in sched) else tr for r in remaining)
    else:
        trained = max(0, done)
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
