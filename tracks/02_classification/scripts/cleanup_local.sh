#!/usr/bin/env bash
# 로컬 서빙 검증이 남긴 프로세스와 파일을 정리합니다.
#
# 사용:
#   bash scripts/cleanup_local.sh              # 목록만 표시
#   bash scripts/cleanup_local.sh --yes        # 실제 삭제
#   KEEP_MODEL=1 bash scripts/cleanup_local.sh --yes   # 모델은 유지
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"      # 트랙 디렉토리(scripts의 부모)
DO_IT=0
[ "${1:-}" = "--yes" ] && DO_IT=1

log() { printf '\033[1;36m[cleanup-local]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[cleanup-local:ERROR]\033[0m %s\n' "$*" >&2; }

# --- 1) vLLM 서버 프로세스 (GPU 점유 해제) ---
# 자기 자신(이 스크립트를 띄운 셸)이 잡히지 않도록 실제 서버 프로세스만 고른다.
mapfile -t PIDS < <(pgrep -f 'vllm (serve|bench)|vllm\.entrypoints' 2>/dev/null | grep -v "^$$\$" || true)
if [ "${#PIDS[@]}" -gt 0 ]; then
  log "실행 중인 vLLM 프로세스: ${PIDS[*]}"
  if [ "$DO_IT" = "1" ]; then
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null && log "  killed $p"; done
    for _ in $(seq 1 10); do
      sleep 1
      pgrep -f 'vllm (serve|bench)|vllm\.entrypoints' >/dev/null 2>&1 || break
    done
    pgrep -f 'vllm (serve|bench)|vllm\.entrypoints' >/dev/null 2>&1 \
      && err "  일부 프로세스가 남았습니다. 확인: pgrep -af vllm" \
      || log "  vLLM 종료 확인"
  fi
else
  log "실행 중인 vLLM 프로세스 없음"
fi

# --- 2) 삭제 대상 목록 ---
TARGETS=()
[ "${KEEP_MODEL:-0}" = "1" ] || TARGETS+=("$HERE/local_model")
TARGETS+=("$HERE/local_model/_model.tar.gz" "$HERE/model.tar.gz" "$HERE/bench" "$HERE/req.json")

log "삭제 대상:"
FOUND=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then
    sz="$(du -sh "$t" 2>/dev/null | cut -f1)"
    printf '    %-8s %s\n' "$sz" "$t"
    FOUND=1
  fi
done
[ "$FOUND" = "0" ] && log "    (이미 정리됨)"

if [ "${KEEP_MODEL:-0}" = "1" ]; then
  log "KEEP_MODEL=1이므로 local_model을 유지합니다."
fi

# --- 3) 실행 ---
if [ "$DO_IT" != "1" ]; then
  log "목록만 표시했습니다. 실제로 지우려면:  bash scripts/cleanup_local.sh --yes"
  exit 0
fi
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] && rm -rf "$t" && log "삭제: $t"
done

log "완료."
log "AWS 엔드포인트와 관련 리소스는 99_cleanup.ipynb에서 삭제하세요."
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null \
  | sed 's/^/[cleanup-local] GPU 사용량: /' || true
