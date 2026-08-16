#!/usr/bin/env bash
# 학습 산출물을 로컬 vLLM 서버로 실행합니다.
#
# Gemma 4를 지원하는 vLLM 0.19 이상이 필요합니다.
#
# 사용:
#   # 텍스트 모델(re-export된 gemma4_text 또는 텍스트 전용): 그냥
#   bash serve_local_vllm.sh /path/to/model
#   # 멀티모달 base를 '텍스트로만' 서빙(re-export 안 한 경우): 이미지/오디오 입력 차단
#   TEXT_ONLY=1 bash serve_local_vllm.sh /path/to/multimodal_model
#   # 멀티모달로 서빙(이미지 입력 허용):
#   MULTIMODAL=1 bash serve_local_vllm.sh /path/to/multimodal_model
#
# 서버가 뜨면(:8000, OpenAI 호환) 다른 터미널/노트북에서 호출. Ctrl-C로 종료.
set -euo pipefail

MODEL_DIR="${1:?usage: serve_local_vllm.sh <model_dir> (env: TEXT_ONLY=1 | MULTIMODAL=1, PORT, MAX_LEN)}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-2048}"
SERVED_NAME="${SERVED_NAME:-gemma-local}"
# 24GB급 GPU에서 메모리 부족을 피하도록 동시 요청 수를 낮춥니다.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' CLI not found. Install: pip install 'vllm>=0.19'  (gemma-4 needs >=0.19)" >&2
  exit 1
fi

ARGS=(serve "$MODEL_DIR"
      --port "$PORT"
      --served-model-name "$SERVED_NAME"
      --max-model-len "$MAX_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --trust-remote-code)

# 텍스트 전용 모드에서는 이미지와 오디오 입력을 차단합니다.
if [[ "${TEXT_ONLY:-0}" == "1" ]]; then
  echo "[serve] TEXT_ONLY: disabling image+audio inputs (--limit-mm-per-prompt image=0,audio=0)"
  ARGS+=(--limit-mm-per-prompt '{"image":0,"audio":0}')
elif [[ "${MULTIMODAL:-0}" == "1" ]]; then
  echo "[serve] MULTIMODAL: image inputs enabled"
  # 기본 limit 사용(이미지 허용). 필요 시 --limit-mm-per-prompt 로 조정.
  :
fi

echo "[serve] vllm ${ARGS[*]}"
echo "[serve] OpenAI-compatible server: http://localhost:${PORT}/v1  (Ctrl-C to stop)"
exec vllm "${ARGS[@]}"
