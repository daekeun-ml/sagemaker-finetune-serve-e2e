#!/usr/bin/env bash
# AgentCore 에이전트를 배포 전에 로컬에서 검증합니다.
#
# 왜: agentcore deploy(AWS)는 수 분+과금. 그 전에 로컬 dev 서버(:8080)로 "에이전트가 실제로
#     추론하나?"를 몇 초 만에 확인한다. reasoning=Bedrock Claude, 추출=SLM endpoint tool.
#
# 전제:
#   1) Node>=20 + @aws/agentcore 설치됨   (없으면: bash agentcore/setup_agentcore_cli.sh)
#   2) 에이전트 프로젝트 생성됨            (없으면: bash agentcore/create_agent.sh)
#   3) SLM endpoint가 InService
#
# 사용:
#   bash agentcore/verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION] [PROMPT]
# 예:
#   bash agentcore/verify_local.sh gemma-extraction-hf-1784954006 us-east-1
#
# dev 서버는 백그라운드에서 실행하고 기록한 PID로 종료합니다.
set -uo pipefail

SLM_ENDPOINT_NAME="${1:?usage: verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION] [PROMPT]}"
# 리전 우선순위: 인자 > 셸 환경변수 > 저장소 .env > 기본값
REPO_ENV="$(cd "$(dirname "$0")/.." && pwd)/.env"
AWS_REGION_DOTENV=""
if [ -z "${AWS_REGION:-}" ] && [ -f "$REPO_ENV" ]; then
    # 값에서 공백과 따옴표를 제거합니다.
    AWS_REGION_DOTENV="$(sed -n 's/^AWS_REGION=//p' "$REPO_ENV" | tail -1 \
        | tr -d '[:space:]' | tr -d "\"'")"
fi
AWS_REGION_IN="${2:-${AWS_REGION:-${AWS_REGION_DOTENV:-us-west-2}}}"
PROMPT="${3:-Extract a tool call as JSON from: Book a flight from Seoul to Paris on 2026-09-15.}"
BEDROCK_MODEL="${BEDROCK_CLAUDE_MODEL_ID:-global.anthropic.claude-sonnet-5}"
PROJECT_NAME="${PROJECT_NAME:-gemmaextraction}"
PORT="${PORT:-8080}"

log() { printf '\033[1;36m[verify]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[verify:ERROR]\033[0m %s\n' "$*" >&2; }

# --- nvm Node 20 로드 (PATH에 agentcore가 있도록) ---
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.nvm/nvm.sh"; nvm use 20 >/dev/null 2>&1 || true
fi
command -v agentcore >/dev/null 2>&1 || { err "agentcore CLI 없음. 실행: bash agentcore/setup_agentcore_cli.sh"; exit 1; }

# --- 프로젝트 폴더 확인 ---
HERE="$(cd "$(dirname "$0")" && pwd)"        # .../agentcore
PROJ_DIR="$HERE/$PROJECT_NAME"
if [ ! -d "$PROJ_DIR" ]; then
  err "에이전트 프로젝트 '$PROJ_DIR' 없음. 먼저 실행: bash agentcore/create_agent.sh"; exit 1
fi

export SLM_ENDPOINT_NAME AWS_REGION="$AWS_REGION_IN" BEDROCK_CLAUDE_MODEL_ID="$BEDROCK_MODEL"
log "endpoint=$SLM_ENDPOINT_NAME region=$AWS_REGION model=$BEDROCK_CLAUDE_MODEL_ID"

LOG="/tmp/agentcore_dev_verify.log"
cd "$PROJ_DIR"

# --- 1) dev 서버 백그라운드 기동 (setsid + stdin 분리) ---
log "dev 서버 기동 중 (백그라운드)..."
setsid agentcore dev --skip-deploy --logs </dev/null >"$LOG" 2>&1 &

# --- 2) 준비 대기 (최대 ~60s, /ping 200) ---
READY=0
for i in $(seq 1 20); do
  sleep 3
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:${PORT}/ping" 2>/dev/null || echo 000)"
  if [ "$code" = "200" ]; then READY=1; log "서버 준비됨 (ping 200, ${i}회차)"; break; fi
done
if [ "$READY" != "1" ]; then
  err "dev 서버가 안 떴습니다. 로그: $LOG"; tail -20 "$LOG" 2>/dev/null
  for p in $(pgrep -f 'agentcore dev' 2>/dev/null); do kill "$p" 2>/dev/null; done
  exit 1
fi

# --- 3) 실제 추론 ---
log "추론 호출: \"$PROMPT\""
echo "────────────────── SLM tool 추론 결과 ──────────────────"
agentcore dev --stream "$PROMPT" </dev/null 2>&1 | tail -30
echo "─────────────────────────────────────────────────────────"

# --- 4) dev 서버 정밀 종료 (pkill 아님). uvicorn reloader 자식까지 정리 ---
log "dev 서버 종료..."
for p in $(pgrep -f 'agentcore dev' 2>/dev/null); do kill "$p" 2>/dev/null && log "  killed $p"; done
for p in $(pgrep -f "${PROJECT_NAME}agent" 2>/dev/null); do kill "$p" 2>/dev/null; done  # uvicorn 자식
# kill 직후의 지연을 고려해 포트가 닫힐 때까지 최대 15초 기다립니다.
CLOSED=0
for _ in $(seq 1 8); do
  sleep 2
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:${PORT}/ping" 2>/dev/null || echo 000)"
  if [ "$code" = "000" ]; then CLOSED=1; break; fi
done
[ "$CLOSED" = "1" ] && log "서버 종료 확인 (:${PORT} 닫힘)" \
  || err "포트 ${PORT}가 아직 열려있음. 남은 프로세스: pgrep -f 'agentcore dev'"
log "완료. (서버 로그: $LOG)"
