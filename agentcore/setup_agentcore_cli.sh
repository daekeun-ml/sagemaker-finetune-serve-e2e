#!/usr/bin/env bash
# Node.js와 @aws/agentcore CLI를 사용자 홈에 설치합니다.
#
# 사용:
#   bash agentcore/setup_agentcore_cli.sh
#   # 설치 후, 같은 터미널에서 nvm을 계속 쓰려면:  source ~/.nvm/nvm.sh && nvm use 20
#
# nvm과 Node.js는 사용자 홈에 설치하며 sudo를 사용하지 않습니다.
set -euo pipefail

NODE_MAJOR="${NODE_MAJOR:-20}"          # 필요 시 NODE_MAJOR=22 로 오버라이드
NVM_VERSION="${NVM_VERSION:-v0.40.1}"   # nvm 설치 태그
AGENTCORE_PKG="${AGENTCORE_PKG:-@aws/agentcore}"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup:ERROR]\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1) 현재 Node가 이미 20+ 인가? (그러면 nvm 건너뛰기)
# ---------------------------------------------------------------------------
node_major() { command -v node >/dev/null 2>&1 && node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0; }

CUR="$(node_major)"
if [ "$CUR" -ge "$NODE_MAJOR" ] 2>/dev/null; then
  log "현재 Node v$(node --version | tr -d 'v')을 사용합니다."
else
  log "현재 Node 메이저=${CUR} (< ${NODE_MAJOR}). nvm으로 Node ${NODE_MAJOR} 설치를 진행합니다."

  # ---------------------------------------------------------------------------
  # 2) nvm 설치 (없을 때만)
  # ---------------------------------------------------------------------------
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    log "nvm(${NVM_VERSION}) 설치 중..."
    curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
  else
    log "nvm 이미 설치됨: $NVM_DIR"
  fi

  # ---------------------------------------------------------------------------
  # 3) nvm 로드 + Node 설치/활성화
  # ---------------------------------------------------------------------------
  # shellcheck disable=SC1090
  . "$NVM_DIR/nvm.sh"
  if ! command -v nvm >/dev/null 2>&1; then
    err "nvm 로드 실패. 새 터미널을 열고 다시 실행하거나 'source $NVM_DIR/nvm.sh'를 확인하세요."
    exit 1
  fi
  log "Node ${NODE_MAJOR} 설치/활성화..."
  nvm install "$NODE_MAJOR"
  nvm use "$NODE_MAJOR"
  nvm alias default "$NODE_MAJOR" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# 4) Node 버전 확인
# ---------------------------------------------------------------------------
CUR="$(node_major)"
if [ "$CUR" -lt "$NODE_MAJOR" ] 2>/dev/null; then
  err "Node가 여전히 메이저=${CUR}. 'source $HOME/.nvm/nvm.sh && nvm use ${NODE_MAJOR}' 후 재실행하세요."
  exit 1
fi
log "Node: $(node --version) | npm: $(npm --version) | which node: $(command -v node)"

# ---------------------------------------------------------------------------
# 5) @aws/agentcore 설치
# ---------------------------------------------------------------------------
log "${AGENTCORE_PKG} 전역 설치 중..."
npm install -g "$AGENTCORE_PKG"

if command -v agentcore >/dev/null 2>&1; then
  log "완료: $(agentcore --version 2>/dev/null || echo 'agentcore 설치됨')"
  log "   agentcore 경로: $(command -v agentcore)"
else
  err "agentcore 명령을 PATH에서 못 찾음. 새 터미널을 열거나 'source $HOME/.nvm/nvm.sh && nvm use ${NODE_MAJOR}' 후 확인하세요."
  exit 1
fi

cat <<EOF

설치 완료. 현재 터미널에서 agentcore를 사용하려면 아래 명령을 실행하세요.

     source \$HOME/.nvm/nvm.sh && nvm use ${NODE_MAJOR}

   다음 단계:
     cd $(cd "$(dirname "$0")" && pwd)
     agentcore create   # 프레임워크=Strands, 모델=Bedrock
     agentcore dev       # 로컬 핫리로드
     agentcore deploy    # Runtime 배포
EOF
