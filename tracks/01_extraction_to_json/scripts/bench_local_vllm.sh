#!/usr/bin/env bash
# bench_local_vllm.sh — 로컬 vLLM 서버 성능 측정(vllm bench). 배포 인스턴스·동시성 결정 근거를 만든다.
#
# 왜: SageMaker 인스턴스를 고르거나 오토스케일링을 정할 때 "감"으로 하면 과·소 프로비저닝이 된다.
#     배포 전에 로컬에서 재두면 TTFT/TPOT/처리량의 '경향'을 근거로 정할 수 있다.
#     (로컬 GPU ≠ 클라우드 GPU라 절대값은 다르지만, 동시성 한계·입력 길이 영향은 그대로 참고 가능.)
#
# 전제: 다른 터미널에서 vLLM 서버가 떠 있어야 한다(serve 모드).
#         bash scripts/serve_local_vllm.sh <MODEL_DIR>
#
# 사용:
#   bash scripts/bench_local_vllm.sh <MODEL_DIR>                 # serve(온라인) 1회 — 기본
#   MODE=sweep  bash scripts/bench_local_vllm.sh <MODEL_DIR>     # 동시성 1→4→8→16 스윕(한계 찾기)
#   MODE=startup bash scripts/bench_local_vllm.sh <MODEL_DIR>    # 콜드스타트(모델 로딩 시간)
#   MODE=throughput bash scripts/bench_local_vllm.sh <MODEL_DIR> # 오프라인 배치 처리량(서버 불필요)
#
# 옵션(env): PORT(8000) INPUT_LEN(1024) OUTPUT_LEN(256) NUM_PROMPTS(50) CONCURRENCY(8) SAVE(0|1)
#
# 🔴 플래그 이름은 설치된 vLLM CLI에서 확인한 것(`vllm bench serve --help=all`). 버전이 올라가면 재확인.
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

command -v vllm >/dev/null 2>&1 || { err "vllm CLI 없음 → uv pip install 'vllm>=0.25'"; exit 1; }

SAVE_ARGS=()
if [ "$SAVE" = "1" ]; then
  mkdir -p ./bench
  SAVE_ARGS=(--save-result --result-dir ./bench)
  log "결과를 ./bench 에 저장합니다"
fi

# --- 서버가 필요한 모드는 먼저 살아있는지 확인 (없으면 몇 분 기다리다 실패하는 걸 방지) ---
need_server() { [ "$MODE" = "serve" ] || [ "$MODE" = "sweep" ]; }
SERVED_NAME="$MODEL_DIR"      # startup/throughput은 직접 로드하므로 경로를 쓴다
if need_server; then
  body="$(curl -s --max-time 3 "${BASE_URL}/v1/models" 2>/dev/null)"
  if [ -z "$body" ] || ! printf '%s' "$body" | grep -q '"data"'; then
    err "vLLM 서버가 ${BASE_URL} 에 없습니다."
    echo "   다른 터미널에서 먼저 실행하세요:  bash scripts/serve_local_vllm.sh ${MODEL_DIR}"
    exit 1
  fi
  # 🔴 --model 에는 '서버에 등록된 이름'을 줘야 한다. 경로를 주면 매칭 실패로 전부 404(Not Found)가
  #    나면서 결과가 0으로 찍힌다(실측). serve_local_vllm.sh는 --served-model-name gemma-local 로 띄우므로
  #    /v1/models 에서 실제 id를 읽어 쓴다.
  #    ⚠️ sed로 첫 "id"를 긁으면 안 된다 — 응답 안에 permission의 "id":"modelperm-..."가 섞여 있어
  #       그걸 잡으면 역시 404가 난다(실측). data[0].id를 정확히 파싱한다.
  SERVED_NAME="$(printf '%s' "$body" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
  [ -n "$SERVED_NAME" ] || SERVED_NAME="$MODEL_DIR"
  log "서버 확인됨: ${BASE_URL}  (served model: ${SERVED_NAME})"
fi

run_serve() {   # $1 = concurrency
  local c="$1"
  log "── serve: concurrency=${c}, in=${INPUT_LEN}, out=${OUTPUT_LEN}, prompts=${NUM_PROMPTS}"
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
    # 🔴 동시성을 올려가며 '처리량은 더 안 오르는데 P99만 치솟는' 지점을 찾는다 = 이 GPU의 한계.
    log "동시성 스윕: 1 → 4 → 8 → 16 (각 구간의 TTFT/TPOT/처리량을 비교하세요)"
    for c in 1 4 8 16; do
      run_serve "$c"
      echo
    done
    log "해석: 처리량 증가가 멈추고 P99 지연만 커지는 구간이 한계입니다."
    log "      그 아래 동시성을 목표로 인스턴스 수/크기를 정하세요."
    ;;
  startup)
    # endpoint가 InService까지 걸리는 시간(콜드스타트) 가늠 — 서버 불필요.
    log "── startup: 모델 로딩 시간 측정 (서버 없이 직접 로드)"
    vllm bench startup --model "$MODEL_DIR"
    log "해석: 이 시간이 길면 endpoint를 자주 내리지 말고 유지하는 편이 낫습니다."
    ;;
  throughput)
    # 오프라인 일괄 처리량 — Batch Transform 규모 산정용. 서버 불필요(직접 로드).
    log "── throughput: 오프라인 배치 처리량 (서버 없이 직접 로드 — GPU 여유 필요)"
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

log "완료. 🔴 로컬 vLLM 서버는 검증이 끝나면 Ctrl-C로 종료하세요(GPU 점유)."
