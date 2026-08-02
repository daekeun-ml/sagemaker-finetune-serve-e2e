# pipelines — 코스를 파이썬으로 실행

노트북 대신 **파이썬을 그대로 돌려** 코스 하나를 끝까지 수행합니다. 노트북과 같은 `common/`
레이어를 쓰므로 결과는 같고, 진입점만 다릅니다.

```bash
# 과금 없이 전 경로 점검 (몇 초, AWS 자격증명 없이도 됩니다)
python pipelines/run_extraction.py --stages all --dry-run

# 전 구간
python pipelines/run_extraction.py --stages all

# 나눠서 — 학습만 돌려두고 나중에 배포
python pipelines/run_extraction.py --stages data,train
python pipelines/run_extraction.py --stages deploy,eval

# 끝나면 반드시 (endpoint 과금 중단)
python pipelines/run_extraction.py --stages cleanup
```

## 코스와 스테이지

| 진입점 | 코스 | GRPO |
|---|---|---|
| `run_extraction.py` | 텍스트 → 구조화 JSON | ✅ |
| `run_classification.py` | intent 분류 | ✅ |
| `run_summarization.py` | 문서 요약 | ❌ |
| `run_domain_qa.py` | 도메인 QA | ❌ |
| `run_multimodal.py` | 이미지 → JSON (영수증) | ❌ |

스테이지 순서는 `data → train → grpo → deploy → eval → cleanup`입니다. `--stages all`은
**grpo와 cleanup을 제외**합니다. cleanup은 방금 만든 endpoint를 실수로 지우지 않게, grpo는
SFT로 충분한 경우가 많고 GPU 시간이 한 번 더 들기 때문입니다.

GRPO까지 돌리려면 `--stages all+grpo`를 씁니다.

GRPO가 없는 코스에 `--stages grpo`를 주면 이유를 설명하고 거부합니다. 요약·QA는 reward를
프로그램으로 채점할 수 없어 rollout이 전부 만점이 되고 advantage가 0이 됩니다.

에이전트 단계(`05_agentic_strands`, `06_agentcore_deploy`)는 노트북에만 있습니다. 질의를
바꿔가며 응답을 보는 성격이라 스크립트로 만들 이득이 없습니다.

## 속도 측정 — `run_benchmark.py`

속도는 파이프라인 스테이지가 아니라 **별도 진입점**입니다. `run_<course>.py`는 모델을 만들어
배포하는 흐름이고(앞 단계가 뒤 단계의 선행조건), 벤치마크는 이미 있는 endpoint를 설정만 바꿔
몇 번씩 다시 재는 일입니다. 한 파이프라인에 두면 "배포를 다시 해야 재나?"를 매번 되묻게 됩니다.

```bash
uv pip install -e '.[bench]'                                  # 최초 1회

python pipelines/run_benchmark.py --course extraction          # 상태 파일의 endpoint
python pipelines/run_benchmark.py --endpoint-name my-endpoint   # 이름을 직접
python pipelines/run_benchmark.py --course extraction --print-command   # 명령만 확인
```

측정은 [sm-endpoint-bmt](https://github.com/daekeun-ml/sm-endpoint-bmt)가 합니다. 지표 공식과
필드명, 출력 표를 `vllm bench serve`와 맞춘 도구이고, 로컬 vLLM에 같은 부하를 재생해 대조
검증한 이력이 있습니다. 같은 지표를 두 곳에서 구현하면 숫자가 갈리는 순간 어느 쪽이 맞는지
판단할 근거가 없어지므로 여기서 다시 만들지 않았습니다.

| 지표 | 정의 |
|---|---|
| TTFT | 내용이 있는 첫 청크 도착 − 요청 전송 |
| TPOT | (전체 지연 − TTFT) / (출력 토큰 − 1) |
| ITL | 연속한 청크 사이의 간격 (TTFT는 포함하지 않습니다) |
| E2EL | 요청 전송 → 마지막 청크 |

지표마다 mean, median, 그리고 `benchmark.percentiles`(기본 `"50,95,99"`)가 함께 나옵니다.

```
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.55
Median TPOT (ms):                        15.57
P50 TPOT (ms):                           15.57
P95 TPOT (ms):                           15.57
P99 TPOT (ms):                           15.57
```

`config.yaml`의 `benchmark` 섹션이 건수·동시성·부하율·백분위·저장 여부를 정합니다. `--` 뒤에
쓴 인자는 그 도구에 그대로 전달되며 설정을 덮습니다.

```bash
python pipelines/run_benchmark.py --course extraction -- --num-prompts 500 --max-concurrency 32
python pipelines/run_benchmark.py --course extraction -- --goodput ttft:200 tpot:50
python pipelines/run_benchmark.py --course extraction -- \
  --ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 20
```

goodput, ramp-up, ShareGPT/HuggingFace 데이터셋, CloudWatch 대조가 그 도구에 있습니다.
전체 옵션은 `python -m sagemaker_benchmark --help`로 봅니다.

## 상태는 파일로 넘깁니다

노트북은 셀 사이에서 `%store`로 값을 넘기지만 IPython 전용이라 여기서는 쓸 수 없습니다.
각 스테이지가 만든 것을 JSON 파일에 적고, 다음 스테이지가 읽습니다.

```
.pipeline_state/extraction.json
  { "bucket": ..., "role": ..., "model_data": "s3://.../model.tar.gz",
    "endpoint_name": "gemma-extraction-vllm-...", "training_job": "gemma-extraction-train-...",
    "stages": { "data": "...", "train": "..." } }
```

- **코스별로 파일이 따로**입니다. `%store`는 전역이라 요약 코스가 멀티모달 endpoint를 호출하는
  사고가 있었는데(`maximum context length is 2048`), 파일을 나누면 구조적으로 막힙니다.
- 이미 만들어진 산출물이 있으면 스테이지를 건너뜁니다. 다시 하려면 `--force`.
- `--show-state`로 현재 상태만 볼 수 있습니다.
- 이 디렉터리는 gitignore 대상입니다(endpoint 이름과 S3 URI가 들어갑니다).

## 중간에 끊겼을 때 — 다시 실행하면 이어집니다

**Ctrl+C를 눌러도 학습 Job과 endpoint는 멈추지 않습니다.** AWS에서 계속 돌고 계속 과금됩니다.
그래서 다시 실행할 때 같은 Job을 또 제출하면 GPU가 두 대 돌아갑니다.

**같은 명령을 그대로 다시 실행하면 됩니다.** 상태 파일에 남은 Job·endpoint 이름으로 실제 상태를
조회해서 판단합니다.

```bash
python pipelines/run_extraction.py --stages all
```

```
[data]  skip — train.jsonl 이 이미 있습니다(400건).
[train] 🔴 이전 실행이 제출한 Job 이 아직 돌고 있습니다:
        gemma-extraction-train-20260802064111 (Training, 8분 경과)
        새로 제출하지 않고 이어서 대기합니다.
  training job: InProgress / Training
```

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

> [!WARNING]
> `--force`는 "이미 만든 산출물이 있어도 다시 한다"는 뜻입니다. 다만 **돌고 있는 Job이나
> endpoint는 `--force`로도 새로 만들지 않습니다.** 두 개가 동시에 과금되기 때문입니다.
> 의도적으로 멈춘 Job을 같은 설정으로 다시 제출할 때만 필요합니다.

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

## 설정은 파일, 시크릿은 env

| | 어디에 |
|---|---|
| 모델 크기 · 인스턴스 · 서빙 엔진 · 이미지 태그 · 샘플 수 · epoch | [`../config.yaml`](../config.yaml) (커밋됨) |
| `HF_TOKEN` · `SAGEMAKER_ROLE_ARN` · `AWS_REGION` | env / `.env` (커밋 안 됨) |

`config.yaml`은 값의 **이유를 주석으로** 담고 있어, 왜 `max_num_seqs`가 32인지 파일 안에서 알 수
있습니다. 우선순위는 **셸·`.env`의 기존 env > `config.yaml` > `common/config.py` 기본값**이라,
한 번만 바꿀 때는 셸에서 넘기면 됩니다.

```bash
MODEL_SIZE=31B python pipelines/run_extraction.py --stages train
```

`config.yaml`이 없어도 동작합니다. `_config.py`의 기본값이 같은 값을 씁니다.

## --dry-run이 보장하는 것

과금이 발생하는 것을 **하나도 만들지 않습니다.** 학습 Job·endpoint뿐 아니라 **Bedrock 호출도
하지 않습니다.** Bedrock은 토큰당 과금이라 합성 100건이면 생성 10회 + critique 약 100회가 실제로
청구됩니다. dry-run은 시드를 복제해 형식만 검증하고, GRPO 프롬프트도 무료 경로(holdout)로
대체합니다.

그래서 몇 초에 끝나고, AWS 자격증명이 없는 기계에서도 전 경로를 밟습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `_config.py` | `config.yaml` 로더. 값을 `os.environ`으로 옮겨 `common/config.py`가 해석하게 합니다 |
| `_common.py` | 스테이지 구현 + 상태 저장 + 이어받기 + 실행 드라이버 |
| `run_*.py` | 코스별 진입점. 코스 특이값만 선언하고 드라이버에 넘깁니다 |

`tracks/*/scripts/`와 혼동하지 마세요. 그쪽은 **SageMaker AI 컨테이너 안에서** 도는 `train.py`이고,
여기는 그것을 **제출하는** 쪽입니다.
