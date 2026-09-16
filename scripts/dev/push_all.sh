#!/usr/bin/env bash
# week1..week6 은 선형 체인이라 week1 을 고치면 뒤 브랜치가 전부 다시 쌓인다.
# SHA 가 교체되므로 일반 push 는 non-fast-forward 로 거부된다 - force 가 정상 경로다.
#
# --force-with-lease 를 쓰는 이유: 내가 아는 원격 상태와 다르면 멈춘다.
# 다른 기기에서 push 한 것이 있으면 덮어쓰지 않고 거부한다.
set -euo pipefail
cd "$(dirname "$0")/../.."

BRANCHES="main week1 week2 week3 week4 week5 week6"

echo "원격 상태를 먼저 가져온다 (--force-with-lease 가 이걸 기준으로 판단한다)"
git fetch origin

echo
printf "  %-8s %-10s %-10s %s\n" branch 원격 로컬 상태
for b in $BRANCHES; do
  r=$(git rev-parse --short "origin/$b" 2>/dev/null || echo "-")
  l=$(git rev-parse --short "$b")
  if [ "$r" = "-" ]; then s="새로 생성"
  elif [ "$r" = "$l" ]; then s="변경 없음"
  elif git merge-base --is-ancestor "origin/$b" "$b" 2>/dev/null; then s="정방향"
  else s="재작성 (force 필요)"; fi
  printf "  %-8s %-10s %-10s %s\n" "$b" "$r" "$l" "$s"
done

echo
read -r -p "push 할까요? [y/N] " ok
[ "$ok" = "y" ] || { echo "중단"; exit 0; }

git push --force-with-lease -u origin $BRANCHES
echo
echo "완료. 원격 상태:"
git ls-remote --heads origin | sed 's/^/  /'
