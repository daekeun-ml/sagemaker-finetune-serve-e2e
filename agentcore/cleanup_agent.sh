#!/usr/bin/env bash
# AgentCore 로컬 프로세스와 배포 리소스를 정리합니다.
#
# 정리 대상:
#   [로컬]  agentcore dev 서버 프로세스, 프로젝트 폴더(gemmaextraction/)
#   [AWS]   agentcore deploy로 만든 Runtime + ECR + CDK 스택  (--aws 플래그를 줄 때만)
#
# 사용:
#   bash agentcore/cleanup_agent.sh            # 로컬만 정리 (dev 프로세스 + 프로젝트 폴더)
#   bash agentcore/cleanup_agent.sh --aws      # 로컬 + AWS 배포 리소스까지 정리(destroy)
#   KEEP_PROJECT=1 bash agentcore/cleanup_agent.sh   # 프로젝트 폴더는 남기고 프로세스만 정리
#
# dev 서버는 조회한 PID만 종료합니다.
set -uo pipefail

PROJECT_NAME="${PROJECT_NAME:-gemmaextraction}"
PORT="${PORT:-8080}"
DO_AWS=0
[ "${1:-}" = "--aws" ] && DO_AWS=1

HERE="$(cd "$(dirname "$0")" && pwd)"        # .../agentcore
PROJ_DIR="$HERE/$PROJECT_NAME"

log() { printf '\033[1;36m[cleanup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[cleanup:ERROR]\033[0m %s\n' "$*" >&2; }

# --- 1) 로컬 dev 서버 프로세스 종료 (kill by pid; pkill 안 씀) ---
log "로컬 dev 서버 프로세스 확인/종료..."
KILLED=0
for p in $(pgrep -f 'agentcore dev' 2>/dev/null); do kill "$p" 2>/dev/null && { log "  killed agentcore dev $p"; KILLED=1; }; done
for p in $(pgrep -f "${PROJECT_NAME}agent" 2>/dev/null); do kill "$p" 2>/dev/null && { log "  killed uvicorn child $p"; KILLED=1; }; done
if [ "$KILLED" = "0" ]; then
  log "  실행 중인 dev 프로세스가 없습니다."
else
  # 뭔가 죽였을 때만 포트 닫힘을 폴링(최대 ~6s)
  code=000
  for _ in $(seq 1 6); do
    sleep 1
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://localhost:${PORT}/ping" 2>/dev/null || echo 000)"
    [ "$code" = "000" ] && break
  done
  [ "$code" = "000" ] && log "  포트 ${PORT} 닫힘" \
    || err "  포트 ${PORT}가 아직 열려있음(남은 프로세스: pgrep -f 'agentcore dev')"
fi

# --- 2) (선택) AWS 배포 리소스 destroy ---
if [ "$DO_AWS" = "1" ]; then
  if [ -d "$PROJ_DIR" ]; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; nvm use 20 >/dev/null 2>&1 || true; fi
    log "AWS 배포 리소스 정리 시도 (agentcore destroy)..."
    ( cd "$PROJ_DIR" && agentcore destroy 2>/dev/null ) \
      && log "  agentcore destroy 완료" \
      || {
        err "  'agentcore destroy' 실패 또는 미지원. CDK로 직접 시도하세요:"
        echo "     cd $PROJ_DIR/agentcore/cdk && npx cdk destroy"
        echo "     (또는 콘솔에서 Bedrock AgentCore Runtime + ECR 이미지 + CloudFormation 스택 삭제)"
      }
  else
    err "  프로젝트 폴더가 없습니다($PROJ_DIR). 배포 리소스는 콘솔이나 `agentcore status`로 확인하세요."
  fi
else
  log "AWS 정리는 건너뜀(로컬만). 배포한 적 있으면: bash agentcore/cleanup_agent.sh --aws"
fi

# --- 3) 로컬 프로젝트 폴더 삭제 ---
if [ "${KEEP_PROJECT:-0}" = "1" ]; then
  log "프로젝트 폴더 유지(KEEP_PROJECT=1): $PROJ_DIR"
elif [ -d "$PROJ_DIR" ]; then
  rm -rf "$PROJ_DIR" && log "프로젝트 폴더 삭제: $PROJ_DIR"
else
  log "프로젝트 폴더 없음(이미 정리됨)"
fi

log "완료. SLM 엔드포인트는 99_cleanup.ipynb 또는 SageMaker 콘솔에서 삭제하세요."
