#!/usr/bin/env bash
# serve_local_vllm.sh — 로컬 GPU에서 학습 산출 모델을 vLLM로 띄워 SageMaker 배포 전 프리플라이트.
#
# 왜: DJL LMI/vLLM DLC 배포는 GPU 프로비저닝~컨테이너 pull로 수 분 걸린다. 그 전에 로컬 GPU에서
#     같은 vLLM 엔진으로 몇 초 만에 "이 모델이 서빙되나?"를 검증한다(예: 멀티모달 base를 텍스트로
#     저장했을 때 image-processor 에러가 로컬에서 즉시 재현/차단됨).
#
# 요구: vLLM >= 0.19 (gemma-4 지원). `pip install "vllm>=0.19"` 또는 vLLM DLC와 동일 버전.
#       gemma-4 텍스트 재-export 모델(config model_type=gemma4_text)은 순수 텍스트로 로드된다.
#
# 사용:
#   # 텍스트 모델(재-export된 gemma4_text 또는 텍스트 전용): 그냥
#   bash serve_local_vllm.sh /path/to/model
#   # 멀티모달 base를 '텍스트로만' 서빙(재-export 안 한 경우): 이미지/오디오 입력 차단
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
# 🔴 24GB급 GPU(L4 등)에서의 CUDA OOM 방지. vLLM 기본 max-num-seqs=256은 샘플러 logits 버퍼를
#    256 x vocab(262,144) x 4B = 256MiB 로 잡아, gemma-4 가중치(~15GB) + KV 캐시와 겹치면 터진다.
#    로컬 검증은 동시 요청이 1~2건이라 32로 낮춰도 손실이 없다. 큰 GPU면 올려도 된다.
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

# 🔴 멀티모달 base를 '텍스트로만' 서빙: 모든 mm 모달리티 입력을 0으로. gemma-4 E계열/12B는 audio도 있으니 함께 0.
#    (재-export된 gemma4_text 모델은 이미 텍스트 arch라 이 플래그가 불필요하지만, 줘도 무해.)
if [[ "${TEXT_ONLY:-0}" == "1" ]]; then
  echo "[serve] TEXT_ONLY: disabling image+audio inputs (--limit-mm-per-prompt image=0,audio=0)"
  ARGS+=(--limit-mm-per-prompt '{"image":0,"audio":0}')
elif [[ "${MULTIMODAL:-0}" == "1" ]]; then
  echo "[serve] MULTIMODAL: image inputs enabled"
  # 기본 limit 사용(이미지 허용). 필요 시 --limit-mm-per-prompt 로 조정.
  :
fi

echo "[serve] vllm ${ARGS[*]}"
echo "[serve] OpenAI-compatible server → http://localhost:${PORT}/v1  (Ctrl-C to stop)"
exec vllm "${ARGS[@]}"
