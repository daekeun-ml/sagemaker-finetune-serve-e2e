#!/usr/bin/env bash
# 로컬 vLLM 서버의 지연과 처리량을 측정합니다.
#
# serve와 sweep 모드는 다른 터미널에서 vLLM 서버를 먼저 실행해야 합니다.
#         bash scripts/serve_local_vllm.sh <MODEL_DIR>
#
# 사용:
#   bash scripts/bench_local_vllm.sh <MODEL_DIR>                 # 온라인 측정
#   MODE=sweep  bash scripts/bench_local_vllm.sh <MODEL_DIR>     # 동시성 단계별 측정
#   MODE=startup bash scripts/bench_local_vllm.sh <MODEL_DIR>    # 모델 로딩 시간
#   MODE=throughput bash scripts/bench_local_vllm.sh <MODEL_DIR> # 오프라인 배치 처리량(서버 불필요)
#
# 옵션: PORT, INPUT_LEN, OUTPUT_LEN, NUM_PROMPTS, CONCURRENCY, SAVE
set -uo pipefail

MODEL_DIR="${1:?usage: bench_local_vllm.sh <MODEL_DIR>   (env: MODE=serve|sweep|startup|throughput)}"
MODE="${MODE:-serve}"
PORT="${PORT:-8000}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-8}"
SAVE="${SAVE:-0}"
BASE_URL="http://localhost:${PORT}"

log() { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bench:ERROR]\033[0m %s\n' "$*" >&2; }

command -v vllm >/dev/null 2>&1 || { err "vllm CLI가 없습니다. uv pip install 'vllm>=0.25'"; exit 1; }

SAVE_ARGS=()
if [ "$SAVE" = "1" ]; then
  mkdir -p ./bench
  SAVE_ARGS=(--save-result --result-dir ./bench)
  log "결과를 ./bench 에 저장합니다"
fi

# 서버가 필요한 모드는 벤치마크 전에 상태를 확인합니다.
need_server() { [ "$MODE" = "serve" ] || [ "$MODE" = "sweep" ]; }
SERVED_NAME="$MODEL_DIR"      # startup/throughput은 직접 로드하므로 경로를 쓴다
if need_server; then
  body="$(curl -s --max-time 3 "${BASE_URL}/v1/models" 2>/dev/null)"
  if [ -z "$body" ] || ! printf '%s' "$body" | grep -q '"data"'; then
    err "vLLM 서버가 ${BASE_URL} 에 없습니다."
    echo "   다른 터미널에서 먼저 실행하세요:  bash scripts/serve_local_vllm.sh ${MODEL_DIR}"
    exit 1
  fi
  # 요청에는 서버가 등록한 모델 이름을 사용합니다.
  SERVED_NAME="$(printf '%s' "$body" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
  [ -n "$SERVED_NAME" ] || SERVED_NAME="$MODEL_DIR"
  log "서버 확인됨: ${BASE_URL}  (served model: ${SERVED_NAME})"
fi

run_serve() {   # $1 = concurrency
  local c="$1"
  log "serve: concurrency=${c}, in=${INPUT_LEN}, out=${OUTPUT_LEN}, prompts=${NUM_PROMPTS}"
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$SERVED_NAME" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$c" \
    --percentile-metrics ttft,tpot,itl,e2el \
    "${SAVE_ARGS[@]}"
}

case "$MODE" in
  serve)
    run_serve "$CONCURRENCY"
    ;;
  sweep)
    log "동시성 단계: 1, 4, 8, 16. 각 구간의 TTFT, TPOT, 처리량을 비교하세요."
    for c in 1 4 8 16; do
      run_serve "$c"
      echo
    done
    log "해석: 처리량 증가가 멈추고 P99 지연만 커지는 구간이 한계입니다."
    log "      그 아래 동시성을 목표로 인스턴스 수/크기를 정하세요."
    ;;
  startup)
    log "startup: 모델 로딩 시간 측정"
    vllm bench startup --model "$MODEL_DIR"
    log "해석: 이 시간이 길면 endpoint를 자주 내리지 말고 유지하는 편이 낫습니다."
    ;;
  throughput)
    log "throughput: 오프라인 배치 처리량"
    vllm bench throughput \
      --model "$MODEL_DIR" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS"
    ;;
  *)
    err "MODE=$MODE 는 지원하지 않습니다. serve | sweep | startup | throughput"
    exit 1
    ;;
esac

log "완료. 검증이 끝나면 Ctrl-C로 로컬 vLLM 서버를 종료하세요."
