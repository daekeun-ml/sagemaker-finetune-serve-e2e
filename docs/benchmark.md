# 부하 벤치마크 — endpoint 속도 측정

배포한 endpoint가 얼마나 빨리 응답하는지 재고, 그 수치를 로컬 vLLM 측정치와 나란히 비교합니다.

!!! abstract "한 줄 요약"
    `python pipelines/run_benchmark.py --course <코스>` — TTFT / TPOT / ITL / E2EL을 mean, median,
    p50/p95/p99로 냅니다. 지표 정의는 `vllm bench serve`를 그대로 따릅니다.

평가(`--stages eval`)와는 다른 물음입니다. 평가는 답이 맞는지, 벤치마크는 얼마나 빨리 오는지를
봅니다. 배포한 모델이 정확도는 좋은데 첫 토큰이 2초 뒤에 오면 쓸 수 없습니다.

## 실행

```bash
# 코스의 상태 파일에서 endpoint 이름을 읽습니다
python pipelines/run_benchmark.py --course classification

# 이름을 직접 주면 이 kit 밖에서 만든 endpoint도 잴 수 있습니다
python pipelines/run_benchmark.py --endpoint-name my-endpoint

# 실행할 명령만 확인하고 끝냅니다(측정하지 않습니다)
python pipelines/run_benchmark.py --course classification --print-command
```

리소스를 만들지 않습니다. 이미 도는 endpoint를 호출할 뿐입니다.

## 지표

| 지표 | 정의 | 무엇을 말해 주나 |
|---|---|---|
| TTFT | 내용이 있는 첫 청크 도착 − 요청 전송 | 사용자가 기다리는 체감 시간 |
| TPOT | (전체 지연 − TTFT) / (출력 토큰 − 1) | 토큰을 써 내려가는 속도 |
| ITL | 연속한 청크 사이의 간격 (TTFT 제외) | 생성이 고르게 흐르는지 |
| E2EL | 요청 전송 → 마지막 청크 | 응답 전체를 받는 데 걸리는 시간 |

지표마다 mean, median, 그리고 `benchmark.percentiles`(기본 `"50,95,99"`)가 나옵니다. 평균만 보면
꼬리 지연을 놓칩니다. 평균이 좋아도 p99가 나쁘면 일부 요청은 늘 느립니다.

## 실측 예시

이 kit의 분류 코스 endpoint를 실제로 측정한 결과입니다.

**서빙 조건** — `ml.g6.2xlarge`(L4 24GB) 1대, vLLM DLC 0.26.0,
`max_model_len 1024` / `max_num_seqs 32` / `gpu_memory_utilization 0.90`, gemma-4 E4B QLoRA 머지본

**부하** — random 데이터셋 20건, 동시 4, 입력 256 / 출력 128 토큰

```
============ Serving Benchmark Result ============
Successful requests:                     20
Failed requests:                         0
Maximum request concurrency:             4
Benchmark duration (s):                  26.37
Total input tokens:                      4100
Total generated tokens:                  2560
Request throughput (req/s):              0.76
Output token throughput (tok/s):         97.07
Peak output token throughput (tok/s):    104.00
Peak concurrent requests:                8.00
Total token throughput (tok/s):          252.54
```

| 지표 | mean | median | p50 | p95 | p99 |
|---|---|---|---|---|---|
| TTFT (ms) | 180.98 | 161.45 | 161.45 | 387.79 | 390.06 |
| TPOT (ms) | 39.89 | 39.89 | 39.89 | 39.92 | 39.93 |
| ITL (ms) | 39.58 | 39.88 | 39.88 | 40.61 | 41.61 |
| E2EL (ms) | 5247.48 | 5227.77 | 5227.77 | 5452.21 | 5454.17 |

TTFT의 p95가 median의 2.4배입니다. 동시 4로 보내면 뒤에 온 요청이 앞 요청의 prefill을 기다리기
때문이고, median만 봤다면 보이지 않았을 차이입니다.

TPOT 39.9ms는 초당 25토큰입니다. L4 한 장에서 E4B를 bf16으로 서빙할 때의 값이니, 더 빠른 응답이
필요하면 GPU를 올리는 것이 먼저입니다.

출력 20건 전부가 `finish_reason=length`로 잘립니다. random 데이터셋에서는 `ignore_eos`가 강제로
켜져 항상 `output_len`까지 생성하기 때문이며, 이것이 정상입니다. 요청마다 출력 길이가 달라지면
TPOT이 "모델이 얼마나 빨리 쓰나"가 아니라 "얼마나 짧게 답했나"를 재게 됩니다.

## `vllm bench serve`를 참조해 만들었습니다

측정은 [sm-endpoint-bmt](https://github.com/daekeun-ml/sm-endpoint-bmt)가 합니다. vLLM의
`vllm bench serve`를 참조해, 지표 공식과 필드명, 출력 표까지 맞춘 도구입니다. 그래서 로컬 vLLM
실행 결과와 이 endpoint 결과를 나란히 놓고 비교할 수 있습니다.

정의를 맞췄다는 주장은 실제로 확인했습니다. 로컬 L40S에서 `google/gemma-4-E4B-it`을 vLLM
0.26.0으로 띄우고, **같은 서버 프로세스에** 같은 부하(20건, rate 4, 동시 8, 입력 256 → 출력 128)를
두 도구로 각각 흘렸습니다.

| | `vllm bench serve` | sm-endpoint-bmt | 차이 |
|---|---|---|---|
| Total input tokens | 5307 | 5307 | 0.0% |
| Total generated tokens | 2560 | 2560 | 0.0% |
| Request throughput (req/s) | 2.76 | 2.78 | 0.6% |
| Output throughput (tok/s) | 353.78 | 355.95 | 0.6% |
| Peak concurrent requests | 12 | 12 | 0.0% |
| Mean TPOT (ms) | 16.15 | 15.88 | 1.6% |
| Median ITL (ms) | 15.85 | 15.86 | 0.0% |

전송 계층은 다릅니다. vLLM 쪽은 HTTP로 직접 붙고, sm-endpoint-bmt는 boto3
`invoke_endpoint_with_response_stream`으로 SageMaker Runtime을 호출합니다. 그래서 검증용 프록시가
vLLM의 SSE를 AWS event stream으로 다시 포장하는데, 이때 `PayloadPart`를 **7바이트마다 일부러
쪼갭니다**. 토큰 수가 정확히 일치하는 것이 핵심입니다. part를 각각 파싱하는 구현이라면 여기서
토큰이 사라집니다.

!!! info "왜 kit 안에서 다시 구현하지 않았나"
    같은 지표를 두 곳에서 구현하면, 숫자가 갈리는 순간 어느 쪽이 맞는지 판단할 근거가 없어집니다.
    `run_benchmark.py`가 하는 일은 셋뿐입니다: endpoint 이름을 찾고, `config.yaml`의 `benchmark`
    섹션을 그 도구의 CLI 인자로 옮기고, 호출합니다. `--print-command`로 두 번째 단계의 결과를
    눈으로 볼 수 있습니다.

## 설정

`config.yaml`의 `benchmark` 섹션이 정합니다.

| 키 | 기본값 | 뜻 |
|---|---|---|
| `num_prompts` | 20 | 보낼 요청 수 |
| `max_concurrency` | 4 | 동시 요청 수 |
| `request_rate` | `"inf"` | `"inf"`는 한꺼번에. 숫자면 초당 요청 수 |
| `input_len` / `output_len` | 256 / 128 | random 프롬프트의 토큰 길이 |
| `num_warmups` | 2 | 워밍업 요청 수(지표에서 제외) |
| `percentiles` | `"50,95,99"` | 낼 백분위 |
| `save_results` | `true` | 결과 JSON 저장 |

기본값은 확인용으로 작습니다. 용량 산정을 하려면 `num_prompts`를 수백 건으로, `max_concurrency`를
실제 동시 사용자 수로 올리세요.

워밍업은 지표에서 제외됩니다. 첫 호출은 TLS 수립과 컨테이너 캐시가 섞여 TTFT 백분위를 혼자
끌어올립니다.

`request_rate`를 숫자로 주면 도착 간격이 지수분포를 따릅니다. 고정 간격은 실제 트래픽보다 고르게
들어가 큐 대기를 과소평가합니다.

## 설정을 명령줄에서 덮기

`--` 뒤에 쓴 인자는 그 도구에 그대로 전달되며 `config.yaml`보다 우선합니다.

```bash
# 더 큰 부하로
python pipelines/run_benchmark.py --course classification -- \
  --num-prompts 500 --max-concurrency 32

# SLO 만족 비율(goodput)
python pipelines/run_benchmark.py --course classification -- \
  --goodput ttft:200 tpot:50

# 부하를 올려 가며 한계점 찾기
python pipelines/run_benchmark.py --course classification -- \
  --ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 20

# 토큰 수를 정확히 세기(tokenizer 필요)
python pipelines/run_benchmark.py --course classification -- \
  --tokenizer google/gemma-4-E4B-it
```

ShareGPT·HuggingFace 데이터셋, CloudWatch `ModelLatency` 대조도 그 도구에 있습니다. 전체 옵션은
`python -m sagemaker_benchmark --help`로 봅니다.

## 결과 파일

`save_results: true`(기본)면 요청별 값까지 JSON으로 남습니다. `--course`를 주면 그 코스의
`data/`에, 없으면 리포 루트의 `bench_results/`에 저장됩니다. 파일명에 endpoint 이름이 들어가므로
둘 다 gitignore 대상입니다.

키 이름은 vLLM의 `--save-result` 출력과 맞췄습니다(`mean_ttft_ms`, `p99_ttft_ms` 등).

## SageMaker Specifics

vLLM의 표에 없는 절이 하나 더 나옵니다. AWS 경계에서만 생기는 것들입니다.

| 항목 | 왜 보는가 |
|---|---|
| `finish_reason=length`로 잘린 요청 수 | 잘린 답변을 성공으로 세면 지연 수치가 틀어집니다 |
| EOS로 끝난 요청 수 | 위와 짝을 이룹니다 |
| usage 프레임이 없던 요청 수 | 그러면 출력 토큰 수가 추정치가 되고, TPOT도 근사가 됩니다 |
| 예외별 오류 분포 | 실패가 throttling인지 타임아웃인지 갈립니다 |

## 리전이 어긋나면 전부 실패합니다

리전은 다른 스테이지와 같은 값(`common.config.AWS_REGION`)을 쓰고, 우선순위는
**셸 env > `.env` > 기본값**입니다. 셸에 `AWS_REGION`이 남아 있으면 `.env` 값이 무시되고, endpoint가
다른 리전에 있으면 요청이 전부 `Endpoint ... not found`로 실패합니다. 출력은 0으로 채운 표가 되어
그냥 보면 "느린 것"처럼 읽힙니다.

그때 다른 리전을 찾아 실행할 명령을 알려 주고 종료 코드 1을 씁니다.

```
🔴 성공한 요청이 없습니다(실패 20건).
   요청한 리전: us-east-1   endpoint: gemma-classification-vllm-...
   → us-east-1 에는 이 endpoint 가 없습니다.

   ✅ us-west-2 에 있습니다(상태 InService). 리전이 어긋났습니다:
        AWS_REGION=us-west-2 python pipelines/run_benchmark.py --endpoint-name ...
```

!!! warning "측정도 과금입니다"
    벤치마크는 리소스를 만들지 않지만, real-time endpoint는 요청이 없어도 삭제 전까지 시간당
    과금됩니다. 측정이 끝나면 `python pipelines/run_<course>.py --stages cleanup`을 실행하세요.

## 설치

sm-endpoint-bmt는 코어 의존성이라 `uv sync`로 함께 설치됩니다. 단 그 도구는 **Python 3.12 이상**이고
이 kit은 3.10부터 지원하므로, 3.10/3.11 환경에서는 설치되지 않습니다. `run_benchmark.py`가 그 경우를
감지해 무엇이 빠졌는지 알려 줍니다.
