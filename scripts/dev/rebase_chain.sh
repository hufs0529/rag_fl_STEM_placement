#!/usr/bin/env bash
# week1..week6 은 선형 체인이다. week1 을 고치면 뒤를 전부 다시 쌓아야 한다.
# README 의 테스트 개수는 브랜치마다 다르므로 실측으로 갱신한다.
#
#   scripts/dev/rebase_chain.sh            # week2..week6 전부
#   scripts/dev/rebase_chain.sh week3      # week3 **부터 week6 까지**
set -u
cd "$(dirname "$0")/../.."

# 분기점을 **먼저 전부** 구한다. one() 이 부모를 다시 쓰면 그 뒤로는 원래
# 분기점을 찾을 수 없다. 그리고 W^ 는 브랜치가 커밋 하나만 들고 있을 때만
# 분기점과 같다 - week4 처럼 여러 커밋이 쌓인 브랜치에 W^ 를 쓰면 마지막 하나만
# 옮기고 나머지를 조용히 버린다.
declare -A BASE
for n in 2 3 4 5 6; do
  BASE[week$n]=$(git merge-base "week$((n-1))" "week$n" 2>/dev/null || true)
done

one() {
  local W=$1 PARENT=$2 OLD BEFORE AFTER
  OLD=${BASE[$W]:-}
  if [ -z "$OLD" ]; then
    echo "  !! $W 의 분기점을 못 찾았다"; return 1
  fi
  # 리베이스 전 커밋 제목을 남긴다. 아래에서 "어디로도 가지 않은 커밋" 을 찾는다.
  local TD; TD=$(mktemp -d)
  git log --format=%s "$OLD..$W" > "$TD/before"
  BEFORE=$(wc -l < "$TD/before" | tr -d ' ')
  git rebase --onto "$PARENT" "$OLD" "$W" >/dev/null 2>&1 || true
  while [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; do
    local C
    C=$(git status --porcelain | grep -E '^(UU|DU|UD|AA)' | awk '{print $2}')
    for f in $C; do
      case "$f" in
        README.md) git checkout --theirs "$f" >/dev/null 2>&1 || true; git add "$f";;
        .gitignore|configs/*|docs/week1.md|src/*|tests/*|requirements.txt)
          git checkout --ours "$f" >/dev/null 2>&1 || true; git add "$f";;
        results/*|*.log|*.pid|*_done.txt|*_watch.sh)
          git rm --cached -q --force "$f" >/dev/null 2>&1 || git rm -q --force "$f" >/dev/null 2>&1;;
        *) echo "  !! 예상 밖 충돌: $f  (손으로 해결 후 git rebase --continue)"; return 1;;
      esac
    done
    GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || break
  done
  # 커밋이 조용히 사라졌는지 본다. 단순히 개수를 비교하면 오경보가 난다:
  # 부모가 이미 다시 쓰인 뒤에 merge-base 를 구하면 분기점이 실제보다 뒤로
  # 잡히고, 그 구간의 부모 커밋들은 patch-id 가 같아 git 이 정당하게 버린다.
  # 그래서 "리베이스 전 제목이 리베이스 후 브랜치에도, 부모에도 없는" 커밋만
  # 유실로 본다.
  AFTER=$(git rev-list --count "$PARENT..$W")
  git log --format=%s "$PARENT..$W" > "$TD/after"
  git log --format=%s -200 "$PARENT" >> "$TD/after"
  local LOST
  LOST=$(grep -vxF -f "$TD/after" "$TD/before" || true)
  if [ -n "$LOST" ]; then
    echo "  !! $W 에서 커밋이 유실됐다 (${BEFORE} -> ${AFTER}). git reflog 로 복구한다:"
    echo "$LOST" | sed 's/^/     - /'
    rm -rf "$TD"; return 1
  fi
  rm -rf "$TD"

  # 테스트는 **실패 여부로** 판정한다. passed 개수만 뽑으면 "302 passed" 를
  # 찍으면서 그 옆의 "1 failed" 를 숨긴다 - 실제로 그렇게 지나간 적이 있다.
  local OUT N
  OUT=$(.venv/bin/python -m pytest tests -q 2>&1)
  N=$(printf '%s' "$OUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
  if printf '%s' "$OUT" | grep -qE '[0-9]+ (failed|error)'; then
    echo "  !! $W 에서 테스트 실패:"
    printf '%s' "$OUT" | grep -E '^(FAILED|ERROR)' | head -5 | sed 's/^/     /'
    return 1
  fi
  sed -i -E "s/^pytest -q( +)# [0-9]+ tests on this branch/pytest -q\1# ${N} tests on this branch/" README.md
  git add README.md && git commit -q --amend --no-edit
  echo "$W -> $(git rev-parse --short HEAD) | ${AFTER} commits | ${N} passed"
}

# 인자는 **시작 브랜치**이며, 거기서부터 week6 까지 한 번의 호출로 처리한다.
# 브랜치 하나만 처리하고 끝내면 안 된다: one() 은 마지막에 README 테스트 수를
# 넣으려고 부모 tip 을 amend 하므로, 다음 호출에서 자식의 기록된 분기점이
# 사라진다. 그러면 merge-base 가 그보다 뒤로 잡히고, 부모의 tip 이 자식 위에
# **중복 복제**된다 (amend 로 patch-id 가 달라져 git 이 버리지 못한다).
# 한 호출 안에서는 BASE 를 맨 위에서 전부 구해 두므로 그 문제가 없다.
START=${1:-week2}
START=${START#week}
if [ "$START" -lt 2 ] || [ "$START" -gt 6 ]; then
  echo "시작 브랜치는 week2..week6 중 하나여야 한다"; exit 2
fi
for n in $(seq "$START" 6); do one "week$n" "week$((n-1))" || exit 1; done
