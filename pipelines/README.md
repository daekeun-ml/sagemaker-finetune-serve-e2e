# pipelines — 코스를 파이썬으로 한 번에 실행

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

스테이지는 `data → train → grpo → deploy → eval → cleanup` 순이고, `--stages all`은
**cleanup을 제외**합니다(실수로 방금 만든 endpoint를 지우지 않게).

GRPO가 없는 코스에 `--stages grpo`를 주면 이유를 설명하고 거부합니다 — 요약·QA는
reward를 프로그램으로 채점할 수 없어 rollout이 전부 만점이 되고 advantage가 0이 됩니다.

에이전트 단계(`05_agentic_strands`, `06_agentcore_deploy`)는 노트북에만 있습니다. 질의를
바꿔가며 응답을 보는 성격이라 스크립트로 만들 이득이 없습니다.

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

## 설정

설정은 [`../config.yaml`](../config.yaml), 시크릿은 환경변수입니다.

| | 어디에 |
|---|---|
| 모델 크기·인스턴스·엔진·이미지 태그·샘플 수·epoch | `config.yaml` (커밋됨) |
| `HF_TOKEN` · `SAGEMAKER_ROLE_ARN` · `AWS_REGION` | env / `.env` (커밋 안 됨) |

우선순위는 **셸·`.env`의 기존 env > `config.yaml` > `common/config.py` 기본값**입니다.
한 번만 바꿀 때는 셸에서 넘기면 됩니다.

```bash
MODEL_SIZE=31B python pipelines/run_extraction.py --stages train
```

`config.yaml`이 없어도 동작합니다 — `_config.py`의 기본값이 같은 값을 씁니다.

## --dry-run이 보장하는 것

과금이 발생하는 것을 **하나도 만들지 않습니다**. 학습 잡·endpoint뿐 아니라
**Bedrock 호출도 하지 않습니다** — Bedrock은 토큰당 과금이라 합성 100건이면 생성 10회 +
critique 약 100회가 실제로 청구됩니다. dry-run은 시드를 복제해 형식만 검증하고,
GRPO 프롬프트도 무료 경로(holdout)로 대체합니다.

그래서 dry-run은 몇 초에 끝나고, AWS 자격증명이 없는 기계에서도 전 경로를 밟습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `_config.py` | `config.yaml` 로더. 값을 `os.environ`으로 옮겨 `common/config.py`가 해석하게 합니다 |
| `_common.py` | 스테이지 구현 + 상태 저장 + 실행 드라이버 |
| `run_*.py` | 코스별 진입점. 코스 특이값만 선언하고 드라이버에 넘깁니다 |

`tracks/*/scripts/`와 혼동하지 마세요 — 그쪽은 **SageMaker 컨테이너 안에서** 도는
`train.py`이고, 여기는 그것을 **제출하는** 쪽입니다.
