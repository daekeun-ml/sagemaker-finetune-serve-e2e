#!/usr/bin/env bash
# cleanup_local.sh — 02b(로컬 서빙 검증)가 남긴 로컬 리소스 정리.
#
# 왜: AWS 리소스(endpoint 등)는 99_cleanup이 지우지만, 02b는 **로컬**에 다음을 남긴다:
#   - vLLM 서버 프로세스        → GPU 메모리를 계속 점유(다음 학습/서빙이 OOM)
#   - local_model/              → 모델 압축 해제본. E4B가 약 15GB
#   - _model.tar.gz             → 다운로드한 아티팩트(보통 해제 후 자동 삭제되지만 중단되면 남음)
#   - bench/, req.json          → 벤치마크 결과, curl payload
#   과금은 없지만 디스크와 GPU를 먹으므로 실습을 마쳤으면 정리한다.
#
# 사용:
#   bash scripts/cleanup_local.sh              # 목록만 보여줌(안전 기본값)
#   bash scripts/cleanup_local.sh --yes        # 실제 삭제
#   KEEP_MODEL=1 bash scripts/cleanup_local.sh --yes   # 모델은 남기고 나머지만(재검증 예정일 때)
#
# 🔴 vLLM 종료는 kill <pid>로 정밀하게 한다('pkill -f vllm'은 실행 중인 셸/노트북까지 죽일 수 있음).
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
      && err "  일부 프로세스가 남았습니다 — 확인: pgrep -af vllm" \
      || log "  vLLM 종료 확인 ✅"
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
[ "$FOUND" = "0" ] && log "    (없음 — 이미 정리됨)"

if [ "${KEEP_MODEL:-0}" = "1" ]; then
  log "KEEP_MODEL=1 → local_model 은 유지합니다(다시 검증할 때 재다운로드 없이 사용)"
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
log "🔴 AWS 리소스(endpoint/config/model)는 이 스크립트가 건드리지 않습니다 → 99_cleanup.ipynb 를 실행하세요."
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null \
  | sed 's/^/[cleanup-local] GPU 사용량: /' || true
