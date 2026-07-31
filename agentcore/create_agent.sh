#!/usr/bin/env bash
# create_agent.sh — @aws/agentcore 프로젝트를 non-interactive로 생성 (대화형 마법사 대체).
#
# 왜: `agentcore create`는 대화형으로 Name/Framework/Model/... 을 하나씩 물어 번거롭다.
#     모든 항목을 flag로 주면(=[non-interactive]) 한 번에 생성된다(검증 2026-07, CLI v0.24.2).
#
# 전제: Node >= 20 + @aws/agentcore 설치. 없으면 먼저:  bash setup_agentcore_cli.sh
#
# 사용:  bash agentcore/create_agent.sh
#        # 생성 후, 스캐폴딩의 app/<agent>/main.py 에 SLM tool을 이식한다(아래 5번 안내).
set -euo pipefail

# --- 이름은 영숫자만(하이픈 불가), project-name 최대 23자 (CLI 제약, 실측) ---
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
# 🔴 agent-path(--framework 등)와 harness-only(--model-id) flag는 섞을 수 없다(CLI 제약, 실측).
#    커스텀 코드 에이전트는 --framework 경로 → 모델 ID는 생성된 model/load.py 에서 지정(아래 4번).
#    --model-provider Bedrock: AWS 자격증명만으로 동작(Anthropic/OpenAI/Gemini는 API 키 필요).
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

# --- 🔴 SLM tool 자동 이식: 킷 템플릿(main.py/load.py)을 스캐폴딩에 덮어써 손수 편집 불필요 ---
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

────────────────────────────────────────────────────────────
✅ 에이전트 준비 완료: $OUT_DIR/$PROJECT_NAME/
   (SLM tool 이식 + 모델 ID env화 + uv sync 자동 완료)

다음: 로컬 검증 → 배포

1) 로컬 검증 (스크립트 한 방 — dev 기동·추론·종료 자동):
     bash agentcore/verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION]
   예: bash agentcore/verify_local.sh gemma-extraction-hf-1784954006 us-west-2

2) 배포(AWS):
     export SLM_ENDPOINT_NAME=<endpoint> AWS_REGION=<리전> BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5
     cd $OUT_DIR/$PROJECT_NAME && agentcore deploy   # ARM64 → ECR → Runtime endpoint
────────────────────────────────────────────────────────────
EOF
