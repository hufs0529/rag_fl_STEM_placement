#!/usr/bin/env bash
# week1..week6 은 선형 체인이다. week1 을 고치면 뒤를 전부 다시 쌓아야 한다.
# README 의 테스트 개수는 브랜치마다 다르므로 실측으로 갱신한다.
#
#   scripts/dev/rebase_chain.sh            # week2..week6 전부
#   scripts/dev/rebase_chain.sh week3      # week3 하나만 (부모는 week2)
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
  BEFORE=$(git rev-list --count "$OLD..$W")
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
  # 커밋이 하나라도 사라졌으면 멈춘다. 조용히 잃는 것이 가장 비싼 실패다.
  AFTER=$(git rev-list --count "$PARENT..$W")
  if [ "$BEFORE" != "$AFTER" ]; then
    echo "  !! $W 의 커밋이 ${BEFORE} -> ${AFTER} 로 변했다. git reflog 로 복구하고"
    echo "     분기점 계산을 확인한다 (OLD=$OLD)"
    return 1
  fi
  local N
  N=$(.venv/bin/python -m pytest tests -q 2>&1 | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
  sed -i -E "s/^pytest -q( +)# [0-9]+ tests on this branch/pytest -q\1# ${N} tests on this branch/" README.md
  git add README.md && git commit -q --amend --no-edit
  echo "$W -> $(git rev-parse --short HEAD) | ${BEFORE} commits | ${N} passed"
}

if [ $# -eq 1 ]; then
  n=${1#week}; one "week$n" "week$((n-1))"
else
  for n in 2 3 4 5 6; do one "week$n" "week$((n-1))" || exit 1; done
fi
