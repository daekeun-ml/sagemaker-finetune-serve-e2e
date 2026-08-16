# 속도 측정: SageMaker AI Endpoint의 TTFT, TPOT, ITL을 `vllm bench serve` 규약으로

!!! abstract "한 줄 요약"
    `python pipelines/run_benchmark.py --course <코스>`: TTFT / TPOT / ITL / E2EL을 mean, median,
    p50/p95/p99로 냅니다. 지표 정의는 `vllm bench serve`를 그대로 따르므로 로컬 vLLM 측정치와
    나란히 놓고 비교할 수 있습니다.

평가(`--stages eval`)와는 다른 물음입니다. 평가는 답이 맞는지, 벤치마크는 얼마나 빨리 오는지를
봅니다. 배포한 모델이 정확도는 좋은데 첫 토큰이 2초 뒤에 오면 쓸 수 없습니다.

## 왜 별도 도구가 필요한가

vLLM을 직접 띄우면 `vllm bench serve` 한 줄로 TTFT, TPOT, ITL이 나옵니다. SageMaker AI Endpoint에도
부하 측정 수단이 있지만, 재는 대상이 다릅니다.

| Tool | Metrics | Token-level |
|---|---|---|
| `vllm bench serve` | TTFT / TPOT / ITL / E2EL, goodput, ramp-up | 포함 |
| SageMaker Inference Recommender | `ModelLatency`, `MaxInvocations`, `CostPerHour`, `CostPerInference`, CPU, 메모리 사용률 | 미포함 |
| CloudWatch endpoint 지표 | `ModelLatency`, `OverheadLatency`, `Invocations` | 미포함 |

Inference Recommender는 인스턴스 타입과 컨테이너 설정을 바꿔 가며 **어느 조합이 싸고 빠른지**를
골라 줍니다. 그 판단에는 요청 단위(`ModelLatency`)와 비용 단위(`CostPerInference`) 지표가 맞습니다.
인스턴스를 고르는 단계에서는 그것으로 충분합니다.

LLM 서빙에서 하나 더 필요한 것이 스트리밍 안쪽입니다. 응답 전체가 5초 걸려도 첫 토큰이 0.2초에
오면 쓸 만하고, 첫 토큰이 2초 뒤에 오면 총 시간이 같아도 느리게 느껴집니다. 두 경우의
`ModelLatency`는 같습니다.

그래서 SageMaker Runtime 전송을 감싸고 지표를 직접 계산하는 도구가 필요했습니다. 정의는
`vllm bench serve`에서 그대로 가져왔습니다. 그러면 로컬 vLLM에서 잰 값과 endpoint에서 잰 값을
나란히 놓을 수 있습니다.

!!! question "vLLM이 SageMaker보다 빠른 것 아닌가"
    같은 GPU라면 **모델이 토큰을 만드는 속도는 같습니다.** 둘 다 안에서 vLLM이 돌기 때문입니다
    (이 프로젝트의 기본 서빙 컨테이너가 vLLM DLC입니다). 실측에서 TPOT이 1.6%, ITL이 0.0% 차이였던
    이유입니다.

    달라지는 것은 TTFT입니다. HTTP로 직접 붙는 대신 SageMaker Runtime을 한 번 더 거치므로 그만큼이
    붙습니다. 실측 대조는 아래 [vLLM과 나란히 놓고 대조](#vllm과-나란히-놓고-대조)에 있습니다.

## 실행 방법

endpoint를 지정하는 방법이 두 가지입니다. 코스를 주면 배포가 상태 파일에 남긴 이름을 쓰고,
이름을 직접 주면 이 프로젝트 밖에서 만든 endpoint도 잽니다. `--course`는 코스를 실행하는 것이 아니라
endpoint 이름을 찾을 위치를 정합니다.

```bash
# 코스의 상태 파일에서 endpoint 이름을 읽습니다
python pipelines/run_benchmark.py --course classification

# 이름을 직접 주면 이 프로젝트 밖에서 만든 endpoint도 잴 수 있습니다
python pipelines/run_benchmark.py --endpoint-name my-endpoint

# 실행할 명령만 확인하고 끝냅니다(측정하지 않습니다)
python pipelines/run_benchmark.py --course classification --print-command
```

이미 도는 endpoint를 호출할 뿐이라 리소스를 만들지 않습니다. 측정 도구는 코어 의존성이라
`uv sync`로 함께 설치됩니다.

!!! warning "그래도 과금은 계속됩니다"
    real-time endpoint는 요청이 없어도 삭제 전까지 시간당 과금됩니다. 측정이 끝나면
    `python pipelines/run_<course>.py --stages cleanup`을 실행하세요.

## 측정 지표

[![스트리밍 응답의 네 가지 지연 지표를 요청 처리 파이프라인 위에 표시한 다이어그램. 위쪽에 네 지표의 정의가 나열된다: Time to First Token(TTFT)은 요청을 보낸 후 첫 토큰을 생성하는 데 걸리는 시간, Inter-Token Latency(ITL)는 연속된 두 토큰 사이의 실제 시간 간격, Time per Output Token(TPOT)은 각 후속 토큰을 생성하는 평균 시간 간격, End-to-End Latency(E2EL)는 요청을 보낸 시점부터 최종 토큰을 받을 때까지의 시간이다. 가운데에는 입력 프롬프트, 토큰화, Prefill, Decode, 디토큰화, 최종 출력 순서가 표시되고 아래쪽에는 여러 요청의 지연 구간이 겹친 타임라인이 있다](images/latency.png)](images/latency.png)

| Metric | Definition | What it tells you |
|---|---|---|
| TTFT | 내용이 있는 첫 청크 도착 − 요청 전송 | 사용자가 기다리는 체감 시간 |
| TPOT | (전체 지연 − TTFT) / (출력 토큰 − 1) | 토큰을 써 내려가는 속도 |
| ITL | 연속한 청크 사이의 간격 (TTFT 제외) | 생성이 고르게 흐르는지 |
| E2EL | 요청 전송 → 마지막 청크 | 응답 전체를 받는 데 걸리는 시간 |

TTFT는 위 그림에서 **토큰화와 Prefill 구간**입니다. 입력이 길면 prefill이 길어져 TTFT가 늘고,
동시 요청이 많으면 앞 요청의 prefill을 기다려 더 늘어납니다.

TPOT과 ITL은 같은 것을 다르게 봅니다. 요청 하나만 보면 두 값이 같지만, ITL은 토큰 사이 간격을
**하나하나** 모으고 TPOT은 요청별로 **평균을 낸 뒤** 그것들을 모읍니다. 그래서 한 요청 안에서
간격이 튀면 ITL의 p99에는 드러나고 TPOT에는 묻힙니다. 아래 실측에서 ITL p99가 41.61ms인데
TPOT p99가 39.93ms인 이유입니다.

네 지표 모두 mean, median과 함께 `benchmark.percentiles`(기본 `"50,95,99"`)를 냅니다.
평균만 보면 꼬리 지연을 놓칩니다. 평균이 좋아도 p99가 나쁘면 일부 사용자는 늘 느립니다.

## 측정 결과

이 프로젝트의 분류 코스 endpoint를 측정한 값입니다.

**서빙 조건**: `ml.g6.2xlarge`(L4 24GB) 1대, vLLM DLC 0.26.0,
`max_model_len 1024` / `max_num_seqs 32` / `gpu_memory_utilization 0.90`, gemma-4 E4B QLoRA 머지본

**부하**: random 데이터셋 20건, 동시 4, 입력 256 / 출력 128 토큰

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

| Metric | mean | median | p50 | p95 | p99 |
|---|---|---|---|---|---|
| TTFT (ms) | 180.98 | 161.45 | 161.45 | 387.79 | 390.06 |
| TPOT (ms) | 39.89 | 39.89 | 39.89 | 39.92 | 39.93 |
| ITL (ms) | 39.58 | 39.88 | 39.88 | 40.61 | 41.61 |
| E2EL (ms) | 5247.48 | 5227.77 | 5227.77 | 5452.21 | 5454.17 |

TTFT의 p95가 median의 2.4배입니다. 위에서 말한 prefill 대기가 숫자로 나타난 것이고, median만
봤다면 보이지 않았을 차이입니다.

TPOT 39.9ms는 초당 25토큰입니다. L4 한 장에서 E4B를 bf16으로 서빙할 때의 값이니, 더 빠른 응답이
필요하면 GPU를 올리는 것이 먼저입니다.

출력 20건 전부가 `finish_reason=length`로 잘립니다. random 데이터셋에서는 `ignore_eos`가 강제로
켜져 항상 `output_len`까지 생성하기 때문이며, 이것이 정상입니다. 요청마다 출력 길이가 달라지면
TPOT이 모델의 생성 속도보다 응답 길이의 영향을 더 크게 받게 됩니다.

## vLLM과 나란히 놓고 대조

측정은 [sm-endpoint-bmt](https://github.com/daekeun-ml/sm-endpoint-bmt)가 합니다. `vllm bench serve`를
참조해 지표 공식과 필드명, 출력 표까지 맞춘 도구입니다.

두 도구의 지표 정의가 같은지 직접 확인했습니다. 로컬 L40S에서 `google/gemma-4-E4B-it`을 vLLM
0.26.0으로 띄우고, **같은 서버 프로세스에** 같은 부하(20건, rate 4, 동시 8, 입력 256 → 출력 128)를
두 도구로 각각 흘렸습니다.

| Metric | `vllm bench serve` | sm-endpoint-bmt | Δ |
|---|---|---|---|
| Total input tokens | 5307 | 5307 | 0.0% |
| Total generated tokens | 2560 | 2560 | 0.0% |
| Request throughput (req/s) | 2.76 | 2.78 | 0.6% |
| Output throughput (tok/s) | 353.78 | 355.95 | 0.6% |
| Peak concurrent requests | 12 | 12 | 0.0% |
| Mean TPOT (ms) | 16.15 | 15.88 | 1.6% |
| Median ITL (ms) | 15.85 | 15.86 | 0.0% |
| Median TTFT (ms) | 54.84 | 48.22 | 12.1% |

TTFT만 차이가 큽니다. 두 실행이 같은 시각이 아니었고(1시간 남짓 간격), 그 사이 서버 상태가
달라졌기 때문으로 봅니다. TTFT 표준편차도 11.45 → 6.43으로 줄었습니다. 계산식 차이라면 TPOT과
ITL도 같이 벌어져야 하는데 각각 1.6%와 0.0%이므로, 정의는 일치하고 실행 조건이 달랐다는 쪽이
설명이 됩니다.

전송 계층이 다르므로(위 참고) 대조에는 프록시가 필요했습니다. vLLM의 SSE를 AWS event stream으로
다시 포장하면서 `PayloadPart`를 **7바이트 단위로 나눴습니다.** 두 도구의 토큰 수가 일치하는지 확인하기 위한 조건입니다.
각 part를 독립적으로 파싱하는 구현에서는 이 과정에서 토큰이 누락될 수 있습니다.

!!! info "프로젝트에서 직접 구현하지 않은 이유"
    같은 지표를 두 곳에서 구현하면, 숫자가 갈리는 순간 어느 쪽이 맞는지 판단할 근거가 없어집니다.
    `run_benchmark.py`가 하는 일은 셋뿐입니다: endpoint 이름을 찾고, `config.yaml`의 `benchmark`
    섹션을 그 도구의 CLI 인자로 옮기고, 호출합니다. `--print-command`로 두 번째 단계의 결과를
    눈으로 볼 수 있습니다.

## 부하를 조절하기

건수, 동시성, 부하율은 `config.yaml`의 `benchmark` 섹션에서 정합니다.

| Key | Default | Description |
|---|---|---|
| `num_prompts` | 20 | 보낼 요청 수 |
| `max_concurrency` | 4 | 동시 요청 수 |
| `request_rate` | `"inf"` | `"inf"`는 한꺼번에. 숫자면 초당 요청 수 |
| `input_len` / `output_len` | 256 / 128 | random 프롬프트의 토큰 길이 |
| `num_warmups` | 2 | 워밍업 요청 수(지표에서 제외) |
| `percentiles` | `"50,95,99"` | 함께 보고할 백분위 |
| `save_results` | `true` | 결과 JSON 저장 |

기본값은 배포가 잘 됐는지 확인하는 정도로 작게 잡았습니다. 용량을 산정하려면 `num_prompts`를
수백 건으로, `max_concurrency`를 실제 동시 사용자 수로 올리세요.

워밍업은 지표에서 제외됩니다. 첫 호출은 TLS 수립과 컨테이너 캐시가 섞여 TTFT 백분위를 혼자
끌어올립니다.

`request_rate`를 숫자로 주면 도착 간격이 지수분포를 따릅니다. 고정 간격은 실제 트래픽보다 고르게
들어가 큐 대기를 과소평가합니다.

## 한 번만 다르게 돌리고 싶을 때

`--` 뒤에 쓴 인자는 그대로 전달되고 `config.yaml` 값을 덮습니다. 설정 파일을 고치지 않고
이번 실행만 바꿉니다.

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

ShareGPT, HuggingFace 데이터셋으로 바꾸거나 CloudWatch `ModelLatency`와 대조하는 것도 됩니다.
전체 옵션은 `python -m sagemaker_benchmark --help`에 있습니다.

## 결과를 파일로 남기기

`save_results: true`(기본)면 요청별 값까지 JSON으로 남깁니다. 저장 위치는 `--course`를 주면 그
코스의 `data/`, 없으면 저장소 root의 `bench_results/`입니다. 파일명에 endpoint 이름이 들어가서 둘 다
gitignore 대상입니다.

키 이름은 vLLM의 `--save-result` 출력과 맞췄습니다(`mean_ttft_ms`, `p99_ttft_ms` 등).

## 결과를 해석할 때 확인할 항목

출력 맨 아래에 `SageMaker Specifics`가 붙습니다. 지표 자체가 아니라 **위의 지표를 그대로 인용해도
되는지** 판단하게 해 주는 항목들입니다.

```
---------------SageMaker Specifics----------------
Truncated (finish_reason=length):        20
Stopped at EOS:                          0
Finish reason unknown:                   0
Requests without usage frame:            0
Output token count source:               server-usage
Error breakdown:
```

| Line | 읽는 법 |
|---|---|
| `Truncated` | `output_len`에 걸려 잘린 요청 수. random 데이터셋에서는 전건이 정상입니다 |
| `Stopped at EOS` | 모델이 스스로 끝낸 요청 수. 위와 합쳐서 요청 수가 됩니다 |
| `Finish reason unknown` | 0이 아니면 컨테이너가 종료 이유를 알려 주지 않은 것입니다 |
| `Requests without usage frame` | 0이 아니면 출력 토큰 수가 추정치라, TPOT과 tok/s도 근사가 됩니다 |
| `Output token count source` | `server-usage`면 컨테이너가 센 값, `tokenizer`나 `fallback-1`이면 추정입니다 |
| `Error breakdown` | 실패가 있을 때 예외 종류별 건수. throttling과 타임아웃이 구분됩니다 |

특히 아래 둘을 먼저 봅니다. `Output token count source`가 `server-usage`가 아니면 TPOT은 참고값이고,
`Error breakdown`이 비어 있지 않으면 위 지표는 성공한 요청만의 값입니다.
