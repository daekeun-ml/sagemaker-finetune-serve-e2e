# 파이프라인 — 코스를 파이썬으로 한 번에 실행

!!! info "Scope"
    노트북 대신 **파이썬 스크립트로 코스를 실행하는 방법**을 다룹니다.

    - **여기서 다루는 것**: 검증된 코스를 다시 돌릴 때, CI, 무인 실행, 결과 재현
    - **여기서 다루지 않는 것**: 학습 내용 자체(LoRA·하이퍼파라미터)는
      [파인튜닝](03_finetuning.md), 서빙 엔진 선택은 [서빙 컨테이너](05_serving_containers.md)
    - **노트북이 맞는 경우**: 처음 배우는 중이거나 중간 산출물을 눈으로 보고 싶을 때.
      두 경로의 선택 기준은 [실행 runbook](RUN_E2E.md#두-가지-실행-방법)

노트북 대신 **파이썬을 그대로 돌려** 코스 하나를 끝까지 수행합니다. 노트북과 같은
`common/` 레이어를 쓰므로 결과는 같고, 진입점만 다릅니다.

```bash
# 전 구간
python pipelines/run_extraction.py --stages all

# 나눠서 — 학습만 돌려두고 나중에 배포
python pipelines/run_extraction.py --stages data,train
python pipelines/run_extraction.py --stages deploy,eval

# 정리 (endpoint 과금 중단)
python pipelines/run_extraction.py --stages cleanup

# 과금 없이 전 경로 점검
python pipelines/run_extraction.py --stages all --dry-run
```

## 코스와 스테이지

| 진입점 | 코스 | GRPO |
|---|---|---|
| `run_extraction.py` | 텍스트 → 구조화 JSON | ✅ |
| `run_classification.py` | intent 분류 | ✅ |
| `run_summarization.py` | 문서 요약 | ❌ |
| `run_domain_qa.py` | 도메인 QA | ❌ |
| `run_multimodal.py` | 이미지 → JSON (영수증) | ❌ |

스테이지는 `data → train → grpo → deploy → eval → cleanup` 순입니다. `--stages all`은
**grpo와 cleanup을 제외**합니다. cleanup은 실수로 방금 만든 endpoint를 지우지 않게, grpo는
SFT로 충분한 경우가 많고 GPU 시간이 한 번 더 들기 때문입니다.

GRPO까지 돌리려면 `--stages all+grpo`를 씁니다.

GRPO가 없는 코스에 `--stages grpo`를 주면 이유를 설명하고 거부합니다. 요약·QA는
reward를 프로그램으로 채점할 수 없어 rollout이 전부 만점이 되고 advantage가 0이 됩니다.

에이전트 단계(`05_agentic_strands`, `06_agentcore_deploy`)는 노트북에만 있습니다. 질의를
바꿔가며 응답을 보는 성격이라 스크립트로 만들 이득이 없습니다.

## 속도 측정 — `run_benchmark.py`

`eval`이 답이 맞는지를 본다면 벤치마크는 얼마나 빨리 오는지를 봅니다. 배포한 endpoint가 실제
트래픽에서 쓸 만한지는 정확도만으로 답할 수 없습니다.

!!! note "왜 스테이지가 아니라 별도 진입점인가"
    - `run_<course>.py`는 **모델을 만들어 배포하는 흐름**입니다. 순서가 있고 앞 단계가 뒤 단계의
      선행조건입니다.
    - 벤치마크는 **이미 있는 endpoint를 재는 일**입니다. 선행조건이 endpoint 하나뿐이고, 동시성을
      올려 가며 한계를 찾는 식으로 설정만 바꿔 몇 번씩 다시 돌립니다.
    - 한 파이프라인에 두면 "배포를 다시 해야 재나?"를 매번 되묻게 됩니다.

```bash
uv pip install -e '.[bench]'                                   # 최초 1회

python pipelines/run_benchmark.py --course extraction           # 상태 파일의 endpoint
python pipelines/run_benchmark.py --endpoint-name my-endpoint    # 이름을 직접
python pipelines/run_benchmark.py --course extraction --print-command   # 명령만 확인
```

### 측정은 sm-endpoint-bmt가 합니다

[sm-endpoint-bmt](https://github.com/daekeun-ml/sm-endpoint-bmt)를 그대로 씁니다. 지표 공식과
필드명, 출력 표를 `vllm bench serve`와 맞춘 도구이고, 로컬 vLLM에 같은 부하를 재생해 대조 검증한
이력이 있습니다(입력·출력 토큰 수가 정확히 일치, 처리량 0.6% 차이).

같은 지표를 kit과 그 도구 두 곳에서 구현하면, 숫자가 갈리는 순간 어느 쪽이 맞는지 판단할 근거가
없어집니다. 그래서 `run_benchmark.py`가 하는 일은 셋뿐입니다.

1. endpoint 이름을 찾습니다(`--endpoint-name` 또는 코스 상태 파일)
2. `config.yaml`의 `benchmark` 섹션을 그 도구의 CLI 인자로 옮깁니다
3. 그 도구를 부릅니다

`--print-command`로 2번의 결과를 눈으로 확인할 수 있습니다.

### 무엇이 나오나

| 지표 | 정의 |
|---|---|
| TTFT | 내용이 있는 첫 청크 도착 − 요청 전송 |
| TPOT | (전체 지연 − TTFT) / (출력 토큰 − 1) |
| ITL | 연속한 청크 사이의 간격 (TTFT는 포함하지 않습니다) |
| E2EL | 요청 전송 → 마지막 청크 |

지표마다 mean, median, 그리고 `benchmark.percentiles`(기본 `"50,95,99"`)가 함께 나옵니다.
평균만 보면 꼬리 지연을 놓칩니다 — 평균이 좋아도 p99가 나쁘면 일부 요청은 늘 느립니다.

```
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.55
Median TPOT (ms):                        15.57
P50 TPOT (ms):                           15.57
P95 TPOT (ms):                           15.57
P99 TPOT (ms):                           15.57
```

vLLM의 표에 더해 **SageMaker Specifics** 절이 AWS 경계에서 생기는 것을 보고합니다.
`finish_reason=length`로 잘린 요청 수, EOS로 끝난 수, usage 프레임이 없던 수, 예외별 오류 분포입니다.

### 설정과 덮어쓰기

`config.yaml`의 `benchmark` 섹션이 건수·동시성·부하율·백분위·저장 여부를 정합니다. `--` 뒤에 쓴
인자는 그 도구에 그대로 전달되며 설정을 덮습니다.

```bash
python pipelines/run_benchmark.py --course extraction -- --num-prompts 500 --max-concurrency 32
python pipelines/run_benchmark.py --course extraction -- --goodput ttft:200 tpot:50
python pipelines/run_benchmark.py --course extraction -- \
  --ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 20
```

goodput(SLO 만족 비율), ramp-up으로 한계점 찾기, ShareGPT/HuggingFace 데이터셋, 토큰 수를 정확히
제어한 프롬프트, CloudWatch `ModelLatency` 대조가 모두 그 도구에 있습니다. 전체 옵션은
`python -m sagemaker_benchmark --help`로 봅니다.

!!! warning "측정도 과금입니다"
    벤치마크는 리소스를 만들지 않지만, real-time endpoint는 요청이 없어도 삭제 전까지 시간당
    과금됩니다. 측정이 끝나면 `python pipelines/run_<course>.py --stages cleanup`을 실행하세요.

## 단계 사이 상태 전달

노트북은 셀 사이에서 `%store`로 값을 넘기지만 IPython 전용이라 여기서는 쓸 수 없습니다.
각 스테이지가 만든 것을 JSON 파일에 적고, 다음 스테이지가 읽습니다.

```
.pipeline_state/extraction.json
  { "bucket": ..., "role": ..., "model_data": "s3://.../model.tar.gz",
    "endpoint_name": "gemma-extraction-vllm", "stages": {...} }
```

- **코스별로 파일이 따로**입니다. `%store`는 전역이라 요약 코스가 멀티모달 endpoint를
  호출하는 사고가 있었는데(`maximum context length is 2048`), 파일을 나누면 구조적으로 막힙니다.
- 이미 만들어진 산출물이 있으면 스테이지를 **건너뜁니다**. 다시 하려면 `--force`.
- `--show-state`로 현재 상태만 볼 수 있습니다.
- 이 디렉토리는 gitignore 대상입니다(endpoint 이름과 S3 URI가 들어갑니다).

## 중간에 끊겼을 때 — 다시 실행하면 이어집니다

**Ctrl+C를 눌러도 학습 Job과 endpoint는 멈추지 않습니다.** AWS에서 계속 돌고 계속 과금됩니다.
그래서 다시 실행할 때 같은 Job을 또 제출하면 GPU가 두 대 돌아갑니다.

`--stages all`을 다시 실행하면 됩니다. 상태 파일에 남은 Job·endpoint 이름으로 실제 상태를
조회해서 판단합니다.

| 이전 실행이 남긴 것 | 다시 실행하면 |
|---|---|
| 학습 Job이 `InProgress` | 새로 제출하지 않고 **이어서 대기** |
| 학습 Job이 `Completed` | 재학습 없이 **산출물만 회수** |
| 학습 Job이 `Failed`/`Stopped` | 원인을 보여주고 중단 |
| endpoint가 `Creating` | 새로 만들지 않고 `InService`까지 대기 |
| endpoint가 `InService` | 그 endpoint를 그대로 사용 |
| 콘솔에서 지운 리소스 | 없어진 것을 확인하고 새로 만듦 |

진짜로 멈추려면 AWS 쪽에서 멈춰야 합니다.

```bash
aws sagemaker stop-training-job --training-job-name <state의 training_job> --region <리전>
```

!!! warning "--force는 진행 중인 것을 새로 만들지 않습니다"
    `--force`는 "이미 만든 산출물이 있어도 다시 한다"는 뜻입니다. 다만 **돌고 있는 Job이나
    endpoint는 `--force`로도 새로 만들지 않습니다.** 두 개가 동시에 과금되기 때문입니다.
    의도적으로 멈춘 Job을 같은 설정으로 다시 제출할 때만 필요합니다.

## 진행 로그

스테이지 진행은 logger로 나옵니다(기본 `INFO`). 학습이 수십 분 도는 동안 타임스탬프가 없으면
멈춘 것인지 느린 것인지 알 수 없습니다.

```
2026-08-02 06:53:35 | INFO | gemma_e2e.pipelines | [train] gemma-extraction-train-... 제출
2026-08-02 06:53:41 | INFO | gemma_e2e.pipelines |   training job: InProgress / Training
2026-08-02 06:53:41 | INFO | gemma_e2e.pipelines |   │ Starting training...
2026-08-02 06:55:12 | INFO | gemma_e2e.pipelines |   │ {'loss': '1.813', 'grad_norm': '0.6953', ...}
```

Job이 `Training` 단계에 들어가면 **CloudWatch 로그를 함께 흘립니다.** 상태 문자열만 보면
학습이 진행되는지 OOM으로 멈춰 있는지 구분할 수 없습니다. 로그 그룹은 Training 단계 진입 후에
생기므로 그전에는 상태만 나옵니다.

`--quiet`을 주면 `WARNING` 이상만 나옵니다(CI용). 실행 시작 헤더와 과금 확인 화면은 로그가 아니라
사람이 읽는 출력이므로 `--quiet`에서도 그대로 보입니다.

## 설정

설정은 [`config.yaml`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/config.yaml), 시크릿은 환경변수입니다.

| | 어디에 |
|---|---|
| 모델 크기·인스턴스·엔진·이미지 태그·샘플 수·epoch | `config.yaml` (커밋됨) |
| `HF_TOKEN` · `SAGEMAKER_ROLE_ARN` · `AWS_REGION` | env / `.env` (커밋 안 됨) |

우선순위는 **셸·`.env`의 기존 env > `config.yaml` > `common/config.py` 기본값**입니다.
한 번만 바꿀 때는 셸에서 넘기면 됩니다.

```bash
MODEL_SIZE=31B python pipelines/run_extraction.py --stages train
```

`config.yaml`이 없어도 동작합니다. `_config.py`의 기본값이 같은 값을 씁니다.

## --dry-run이 보장하는 것

과금이 발생하는 것을 **하나도 만들지 않습니다**. 학습 Job·endpoint뿐 아니라
**Bedrock 호출도 하지 않습니다.** Bedrock은 토큰당 과금이라 합성 100건이면 생성 10회 +
critique 약 100회가 실제로 청구됩니다. dry-run은 시드를 복제해 형식만 검증하고,
GRPO 프롬프트도 무료 경로(holdout)로 대체합니다.

그래서 dry-run은 몇 초에 끝나고, AWS 자격증명이 없는 기계에서도 전 경로를 밟습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `_config.py` | `config.yaml` 로더. 값을 `os.environ`으로 옮겨 `common/config.py`가 해석하게 합니다 |
| `_common.py` | 스테이지 구현 + 상태 저장 + 실행 드라이버 |
| `run_*.py` | 코스별 진입점. 코스 특이값만 선언하고 드라이버에 넘깁니다 |

`tracks/*/scripts/`와 혼동하지 마세요. 그쪽은 **Amazon SageMaker AI 컨테이너 안에서** 도는
`train.py`이고, 여기는 그것을 **제출하는** 쪽입니다.

## 이어서 볼 문서

- [실행 runbook](RUN_E2E.md): 노트북 경로의 단계별 핸드오프와 비용 가드
- [전체 지도](00_overview.md): 문서·노트북 대응 관계
- [파인튜닝](03_finetuning.md) · [SageMaker AI 추론](04_sagemaker_inference.md)
