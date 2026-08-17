# Python 스크립트 실행법

!!! info "Scope"
    노트북 없이 Python 스크립트로 코스의 데이터 준비, 학습, 배포, 평가와 정리를 실행하는 방법을 설명합니다.

    처음 실행하거나 중간 결과를 확인하려면 [노트북 실행법](run_notebook.md)을 사용하세요.

## 빠른 시작

전체 실행 전에 dry-run으로 단계와 설정을 확인합니다.

```bash
python pipelines/run_extraction.py --stages all --dry-run
```

실제 데이터 준비, 학습, 배포와 평가는 다음 명령으로 실행합니다.

```bash
python pipelines/run_extraction.py --stages all
```

평가가 끝나면 Endpoint를 정리합니다.

```bash
python pipelines/run_extraction.py --stages cleanup
```

`all`에는 GRPO와 cleanup이 포함되지 않습니다. 추출이나 분류 코스에서 GRPO까지 실행하려면 `all+grpo`를 사용합니다.

```bash
python pipelines/run_extraction.py --stages all+grpo
```

## 코스별 진입점

| 유스케이스 | 진입점 | GRPO | 별도 평가 단계 |
|---|---|---|---|
| 텍스트 구조화 추출 | `pipelines/run_extraction.py` | 지원 | 있음 |
| 의도 분류 | `pipelines/run_classification.py` | 지원 | 있음 |
| 문서 요약 | `pipelines/run_summarization.py` | 없음 | 있음 |
| 도메인 질의응답 | `pipelines/run_domain_qa.py` | 없음 | 있음 |
| 이미지 구조화 추출 | `pipelines/run_multimodal.py` | 없음 | 없음 |

멀티모달 코스의 `all`은 해당 코스가 지원하는 데이터, 학습과 배포 단계만 실행합니다.

## 스테이지

| 스테이지 | 동작 | 상태 파일에 저장하는 값 |
|---|---|---|
| `data` | 시드 변환, 합성 데이터와 held-out 데이터 준비 | S3 입력 위치와 로컬 데이터 경로 |
| `train` | SFT Training Job 제출과 완료 대기 | Training Job 이름과 `model_data` |
| `grpo` | SFT 결과를 사용한 GRPO 추가 학습 | GRPO Job 이름과 갱신된 `model_data` |
| `deploy` | Real-time Endpoint 배포와 테스트 호출 | `endpoint_name` |
| `eval` | held-out 데이터 평가 | 평가 결과 |
| `cleanup` | Endpoint, EndpointConfig와 Model 삭제 | 정리 완료 상태 |

필요한 단계만 나눠 실행할 수 있습니다.

```bash
python pipelines/run_extraction.py --stages data,train
python pipelines/run_extraction.py --stages deploy,eval
```

`deploy`와 `eval`은 앞 실행이 상태 파일에 저장한 `model_data`와 `endpoint_name`을 사용합니다.

## dry-run

`--dry-run`은 실제 Training Job, Endpoint와 Bedrock 호출을 만들지 않습니다. AWS 자격증명이 없는 환경에서도 전체 단계와 상태 전달을 확인할 수 있습니다.

dry-run 상태는 실제 실행과 별도 파일에 저장됩니다.

```text
.pipeline_state/extraction.dryrun.json
```

노트북의 `DRY_RUN=1`은 데이터와 학습 규모를 줄이는 설정이며 AWS 리소스를 비활성화하지 않습니다. 두 검증 방식의 차이는 [노트북 DRY_RUN](run_notebook.md#두-가지-검증-모드-구분)에서 확인할 수 있습니다.

## 설정

| 위치 | 용도 |
|---|---|
| `config.yaml` | 모델, 학습, 서빙, 데이터, 평가와 벤치마크 설정 |
| 셸과 `.env` | 리전, 실행 역할, 이미지 URI와 기능 스위치 |
| `common/config.py` | 공통 코드 기본값 |

환경변수가 `config.yaml`보다 우선하고, `config.yaml`이 코드 기본값보다 우선합니다.

한 번만 값을 바꿀 때는 명령 앞에 환경변수를 지정합니다.

```bash
TRAIN_INSTANCE_TYPE=ml.g6e.2xlarge \
  python pipelines/run_extraction.py --stages train
```

HF 토큰과 역할 ARN 같은 계정별 값은 `config.yaml`에 저장하지 않습니다.

다른 설정 파일을 사용하려면 `--config`로 지정합니다.

```bash
python pipelines/run_extraction.py --config my-config.yaml --stages all
```

## 상태 파일

각 스테이지는 결과를 코스별 JSON 파일에 저장합니다.

```text
.pipeline_state/extraction.json
.pipeline_state/classification.json
.pipeline_state/summarization.json
.pipeline_state/domain_qa.json
.pipeline_state/mm_extraction.json
```

상태 파일에는 Training Job, 모델 S3 URI, Endpoint 이름과 완료한 스테이지가 기록됩니다. 이 디렉터리는 Git에 포함되지 않습니다.

현재 상태만 확인할 수 있습니다.

```bash
python pipelines/run_extraction.py --show-state
```

기본 디렉터리를 바꾸려면 `--state-dir`, 파일 하나를 직접 지정하려면 `--state`를 사용합니다.

## 중단 후 재개

터미널을 종료하거나 `Ctrl+C`를 눌러도 AWS의 Training Job과 Endpoint는 계속 실행될 수 있습니다. 같은 명령을 다시 실행하면 상태 파일과 AWS 리소스를 확인해 기존 작업을 이어받습니다.

| 기존 상태 | 재실행 동작 |
|---|---|
| Training Job이 `InProgress` | 새로 제출하지 않고 완료까지 대기 |
| Training Job이 `Completed` | 기존 모델 아티팩트 사용 |
| Training Job이 `Failed` 또는 `Stopped` | 원인을 출력하고 중단 |
| Endpoint가 `Creating` | 기존 Endpoint가 준비될 때까지 대기 |
| Endpoint가 `InService` | 기존 Endpoint 사용 |
| 상태 파일의 리소스가 삭제됨 | 필요한 단계에서 다시 생성 |

AWS 작업 자체를 멈추려면 해당 리소스를 AWS에서 중지하거나 삭제해야 합니다.

## `--force`

`--force`는 완료된 산출물이 있어도 선택한 스테이지를 다시 실행합니다.

```bash
python pipelines/run_extraction.py --stages train --force
```

진행 중인 Training Job이나 Endpoint는 `--force`를 사용해도 중복 생성하지 않습니다. 중복 리소스와 비용을 막기 위한 동작입니다.

## 외부 Endpoint 평가

이 저장소 밖에서 만든 Endpoint를 평가하려면 이름을 직접 지정합니다.

```bash
python pipelines/run_extraction.py \
  --stages eval \
  --endpoint-name my-endpoint
```

지정한 이름은 상태 파일에도 저장되어 이후 평가나 cleanup에서 같은 Endpoint를 사용합니다.

## MLflow 실험 추적

Python 파이프라인은 선택적으로 MLflow에 설정, 단계 상태, 학습 지표와 평가 결과를 기록합니다.

```bash
USE_MLFLOW=0 python pipelines/run_extraction.py --stages all
USE_MLFLOW=1 MLFLOW_TRACKING_URI=local python pipelines/run_extraction.py --stages all
USE_MLFLOW=1 python pipelines/run_extraction.py --stages all
```

첫 번째 명령은 기록을 끄고, 두 번째는 로컬 SQLite를 사용하며, 세 번째는 같은 리전의 SageMaker MLflow App을 찾습니다.

설정과 기록 범위는 [SageMaker Managed MLflow](../experiments/mlflow.md)에서 설명합니다.

## 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| 이전 결과를 재사용해 스테이지가 건너뛰어짐 | `--show-state`로 확인하고 필요하면 `--force` 사용 |
| Training Job이 시작 직후 실패 | 실행 역할 권한, 이미지 URI와 CloudWatch 로그 확인 |
| Training Job이 `Stopped` | 실행 시간 제한과 모델 병합 로그 확인 |
| `InsufficientInstanceCapacity` | 리전 또는 인스턴스 유형 변경 |
| Endpoint가 `Failed` | Endpoint CloudWatch 로그에서 CUDA OOM과 모델 로딩 오류 확인 |
| 상태 파일과 AWS 리소스가 다름 | 삭제된 리소스인지 확인한 뒤 같은 명령 재실행 |
| 비용이 계속 발생 | `cleanup` 실행 후 Endpoint와 AgentCore Runtime 확인 |

학습 문제는 [파인튜닝](../guides/02_finetuning.md), 배포 문제는 [SageMaker AI 추론](../guides/03_sagemaker_inference.md), 엔진별 문제는 [서빙 컨테이너](../guides/04_serving_containers.md)에서 확인하세요.

## cleanup

파이프라인의 `all`에는 cleanup이 포함되지 않습니다. 평가나 데모가 끝나면 별도로 실행합니다.

```bash
python pipelines/run_extraction.py --stages cleanup
```

cleanup은 상태 파일의 Endpoint를 기준으로 Endpoint, EndpointConfig와 Model을 삭제합니다. 여러 번 배포했거나 다른 리전을 사용했다면 AWS 콘솔에서 남은 Endpoint도 확인하세요.

AgentCore Runtime은 파이프라인 스테이지가 아니므로 별도 스크립트로 정리합니다.

```bash
bash agentcore/cleanup_agent.sh --aws
```

## 속도 측정

배포된 Endpoint의 TTFT, TPOT, ITL과 E2EL은 별도 명령으로 측정합니다.

```bash
python pipelines/run_benchmark.py --course extraction
```

사용법과 지표 정의는 [속도 측정](../experiments/benchmark.md)에서 확인할 수 있습니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `pipelines/_config.py` | `config.yaml` 로드와 검증 |
| `pipelines/_common.py` | 스테이지 실행, 상태 저장과 재개 |
| `pipelines/run_*.py` | 코스별 진입점 |
| `config.yaml` | 기본 실행 설정 |
| `.pipeline_state/` | 코스별 실행 상태 |
