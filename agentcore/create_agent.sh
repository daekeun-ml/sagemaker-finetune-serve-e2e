#!/usr/bin/env bash
# @aws/agentcore 프로젝트를 비대화형으로 생성합니다.
#
# 전제: Node >= 20 + @aws/agentcore 설치. 없으면 먼저:  bash setup_agentcore_cli.sh
#
# 사용:  bash agentcore/create_agent.sh
#        # 생성 후, 스캐폴딩의 app/<agent>/main.py 에 SLM tool을 이식한다(아래 5번 안내).
set -euo pipefail

# 이름은 영숫자만 사용하며 프로젝트 이름은 최대 23자입니다.
PROJECT_NAME="${PROJECT_NAME:-gemmaextraction}"     # 영숫자, 문자로 시작, <=23
AGENT_NAME="${AGENT_NAME:-gemmaextractionagent}"    # 영숫자
FRAMEWORK="${FRAMEWORK:-Strands}"                   # 우리 app.py가 Strands 기반
OUT_DIR="${OUT_DIR:-.}"                              # 생성 위치(기본 현재 폴더 = agentcore/)

# nvm Node 20 로드(설치돼 있다고 가정; 아니면 setup_agentcore_cli.sh 먼저)
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.nvm/nvm.sh"; nvm use 20 >/dev/null 2>&1 || true
fi
command -v agentcore >/dev/null 2>&1 || { echo "ERROR: agentcore CLI 없음. 먼저: bash setup_agentcore_cli.sh" >&2; exit 1; }

cd "$(dirname "$0")"   # agentcore/ 로 이동

echo "[create] non-interactive 생성: project=$PROJECT_NAME agent=$AGENT_NAME framework=$FRAMEWORK"
# 커스텀 코드 경로에서는 모델 ID를 생성된 model/load.py에서 설정합니다.
agentcore create \
  --project-name "$PROJECT_NAME" \
  --name "$AGENT_NAME" \
  --framework "$FRAMEWORK" \
  --model-provider Bedrock \
  --memory none \
  --protocol HTTP \
  --build CodeZip \
  --language Python \
  --network-mode PUBLIC \
  --skip-install \
  --output-dir "$OUT_DIR"

AGENT_DIR="$OUT_DIR/$PROJECT_NAME/app/$AGENT_NAME"

# SLM 도구 템플릿을 생성된 프로젝트에 복사합니다.
TPL="$(dirname "$0")/templates"
if [ -f "$TPL/main.py" ] && [ -d "$AGENT_DIR" ]; then
  cp "$TPL/main.py" "$AGENT_DIR/main.py"
  echo "[create] 이식: $AGENT_DIR/main.py ← templates/main.py (extract_structured_json SLM tool)"
fi
if [ -f "$TPL/load.py" ] && [ -d "$AGENT_DIR/model" ]; then
  cp "$TPL/load.py" "$AGENT_DIR/model/load.py"
  echo "[create] 이식: $AGENT_DIR/model/load.py ← templates/load.py (env 기반 모델 ID)"
fi

# --- 의존성 설치 (uv sync) ---
if command -v uv >/dev/null 2>&1; then
  echo "[create] uv sync ($AGENT_DIR)..."
  ( cd "$AGENT_DIR" && uv sync >/dev/null 2>&1 ) && echo "[create] uv sync 완료" || echo "[create] uv sync 실패(수동: cd $AGENT_DIR && uv sync)"
fi

cat <<EOF

에이전트 준비 완료: $OUT_DIR/$PROJECT_NAME/

다음 단계: 로컬 검증, 배포

1) 로컬 검증:
     bash agentcore/verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION]
   예: bash agentcore/verify_local.sh gemma-extraction-hf-1784954006 us-west-2

2) 배포(AWS):
     export SLM_ENDPOINT_NAME=<endpoint> AWS_REGION=<리전> BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5
     cd $OUT_DIR/$PROJECT_NAME && agentcore deploy
EOF
