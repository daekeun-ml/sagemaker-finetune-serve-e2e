# SageMaker AI Fine-tuning & Serving E2E

[Quick start](#quick-start) | [Courses](#courses) | [Why this kit](#why-this-kit) | [Setup](#setup) | [Docs](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/) | [Cost & cleanup](#cost--cleanup)

Gemma 4를 Amazon SageMaker AI에서 **합성 데이터 생성부터 파인튜닝, 서빙, 평가, 에이전트 연계까지** 다루는 한국어 실습 자료입니다.
**태스크별 실습 코스** 5개가 각각 독립된 E2E로 동작하므로, 필요한 태스크 하나만 골라 처음부터 끝까지 돌릴 수 있습니다.

### [가이드북 바로가기](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/)

**코드를 따라 치는 대신, 왜 그렇게 하는지 이해하고 넘어가려는 분을 위한 가이드북입니다.**
SageMaker AI가 처음이어도 읽을 수 있도록 개념부터 시작하고, 각 설정과 구조를 선택한 이유를 설명합니다.
학습 Job이 왜 머지 도중 죽었는지, 24GB GPU에서 서빙이 왜 안 떴는지 같은 것들이 원인과 함께 정리돼 있습니다.

---

## Quick start

### 1) 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv 미설치 시
git clone https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e.git
cd sagemaker-finetune-serve-e2e
uv sync && source .venv/bin/activate
```

AWS 자격증명과 리전이 설정돼 있어야 합니다(`aws configure` 또는 환경변수). Gemma 4는 ungated라 HF 토큰이 필요 없습니다.

### 2) 과금 없이 먼저 확인

```bash
python pipelines/run_extraction.py --stages all --dry-run
```

몇 초 안에 끝나며 **과금 리소스나 유료 API 호출을 만들지 않습니다.** 학습 작업, 엔드포인트, Bedrock 호출을 생략하므로 AWS 자격증명이 없어도 전 경로를 확인할 수 있습니다.

### 3) 실제로 돌리기: 두 가지 방법

**스크립트 (Python)**: 검증된 코스를 다시 돌릴 때, CI, 무인 실행

```bash
python pipelines/run_extraction.py --stages all       # 데이터, 학습, 배포, 평가
python pipelines/run_extraction.py --stages all+grpo  # GRPO 추가 학습 포함 (선택)

python pipelines/run_extraction.py --stages data,train    # 학습만 하고
python pipelines/run_extraction.py --stages deploy,eval   # 나중에 배포

python pipelines/run_extraction.py --stages cleanup   # 끝나면 반드시
```

코스별 지원 단계, 중단 후 재개, 상태 파일과 `--force` 사용법은 [`pipelines/README.md`](pipelines/README.md)에서 확인할 수 있습니다.

**노트북 (JupyterLab)**: 처음 배울 때, 중간 산출물을 눈으로 볼 때

```bash
jupyter lab
```

`tracks/01_extraction_to_json/`을 열어 `00_setup.ipynb`부터 번호 순서대로 실행합니다. 단계별 핸드오프와 비용 가드는 [`docs/RUN_E2E.md`](docs/RUN_E2E.md)에 있습니다.

> [!WARNING]
> **실시간 엔드포인트는 삭제할 때까지 시간당 과금됩니다.** 요청이 없어도 인스턴스 비용이 발생합니다.
> 실습을 마치면 `cleanup` 단계나 `99_cleanup.ipynb`를 **반드시** 실행하세요.

**더 자세한 안내**

- **처음이라면:** [`docs/getting_started.md`](docs/getting_started.md)에서 설치, 스모크 테스트, 로컬 dry-run, 노트북 실행 순서를 확인하세요.
- **설정을 바꾸려면:** [`config.yaml`](config.yaml)에서 모델 크기, 인스턴스, 서빙 엔진을 변경하세요.
- **SDK v2에서 옮겨 왔다면:** [SageMaker Python SDK V3](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/sdk_v3/)에서 차이를 확인하세요.

---

## Who is this for

**Good fit**

- 자체 데이터로 SLM을 파인튜닝해 **엔드포인트까지 배포하고 싶은 분**. 데이터 준비부터 정리까지 한 코스면 끝납니다.
- 관리형 레시피가 **아직 지원하지 않는 모델과 기법**을 써야 하는 분 (아래 [Why this kit](#why-this-kit) 참고).
- 학습 코드를 **직접 수정해야 하는 분**. `scripts/train.py`가 독립적으로 구성돼 있어 TRL/PEFT 설정을 그대로 읽고 고칠 수 있습니다.
- SLM을 **에이전트의 도구로 연결하는 구조**를 보고 싶은 분. Claude가 추론하고 SLM이 특화 작업을 처리합니다.
- SageMaker Python SDK **v3** 기준 코드가 필요한 분 (`ModelTrainer` / `ModelBuilder`. v2의 `Estimator`와 `HuggingFaceModel`은 제거됐습니다).

**Not a good fit**

- 파인튜닝 없이 **바로 배포만** 하려는 경우에는 SageMaker JumpStart가 더 빠릅니다.
- 관리형 레시피가 **이미 지원하는 조합**이라면 SageMaker AI model customization 또는 SageMaker Recipes가 운영 부담이 적습니다.
- 멀티노드 대규모 사전학습은 SageMaker HyperPod 영역입니다. 이 자료는 단일 GPU LoRA/QLoRA를 기준으로 합니다.

전제 지식은 Python과 Jupyter 사용 경험 정도면 충분합니다. SageMaker AI와 Bedrock 개념은 노트북에서 설명합니다.

## Requirements

<details>
<summary><b>준비 사항 8가지 펼쳐 보기</b></summary>

| Item | Notes |
|---|---|
| AWS 계정 | SageMaker AI 학습 작업과 엔드포인트를 만들 수 있는 계정 (과금 발생) |
| IAM 실행 역할 | `AmazonSageMaker-ExecutionRole-*` 또는 동급 권한. 없으면 코드가 IAM에서 자동 탐지 |
| 서비스 쿼터 | 학습과 추론용 GPU 인스턴스 (기본 `ml.g6.2xlarge`). 신규 계정은 **쿼터가 0일 수 있어 미리 증설 신청**이 필요합니다 |
| 리전 | `.env`의 `AWS_REGION`. GPU 용량과 Bedrock 모델 가용성이 리전마다 다릅니다. 리전을 바꾸면 이미지 URI의 리전도 함께 바꾸세요 |
| Bedrock 모델 접근 | 합성 데이터 생성과 LLM-as-judge 평가에 Claude를 사용하므로 콘솔에서 모델 접근 권한을 설정해야 합니다 |
| Python | 3.10 이상 (로컬은 노트북 실행용. 학습은 컨테이너 안에서 돕니다) |
| 로컬 GPU | 선택. 있으면 학습 스크립트를 `--dry_run`으로 실행해 클라우드 제출 전에 검증할 수 있습니다 |
| HF 토큰 | Gemma 4는 ungated라 불필요. gated 모델(Gemma 3 등)을 쓸 때만 `hf auth login` |

</details>

## Why this kit

**"SageMaker AI가 Gemma 4를 지원하지 않나?"** 지원합니다. 다만 **안 되는 조합이 꽤 있습니다.**

AWS 관리형 방식(JumpStart, model customization, Recipes)은 지원 모델, 기법, 리전이 정해져 있고 같은 모델도 크기에 따라 지원 범위가 다릅니다. 이 자료는 해당 목록에 없는 조합을 다룹니다.

| Path | Fine-tuning | Notes |
|---|---|---|
| **SageMaker JumpStart** | 불가 (배포만) | 몇 단계만으로 엔드포인트 배포. Gemma 4 5종 모두 `training_supported=False`로 확인 |
| **SageMaker AI model customization** | 가능 | 관리형 SFT/DPO/RFT. 지원 모델, 기법, 리전이 한정 |
| **SageMaker Recipes** (학습 작업 / HyperPod) | 가능 | 검증된 레시피로 SFT/DPO/GRPO. 지원 모델 목록에 의존 |
| **이 자료 (BYOC, DLC + 커스텀 스크립트)** | 가능 | 모델, 기법, 하이퍼파라미터를 **직접 제어**. 대신 코드를 관리해야 함 |

<details>
<summary><b>이 방식을 선택한 이유와 주의점</b></summary>

1. **지원 목록에 없는 조합을 쓸 수 있습니다.** AWS 관리형 방식은 지원 모델, 기법, 리전이 정해져 있고 자주 바뀝니다. 원하는 조합이 빠져 있으면 기다리는 수밖에 없습니다.
2. **학습 코드가 열려 있습니다.** LoRA 대상, 채팅 템플릿, 패킹, 보상 함수까지 읽고 고칠 수 있습니다. 관리형은 노출된 하이퍼파라미터 범위 안에서만 조정됩니다.
3. **운영 제약이 코드에 반영돼 있습니다.** 컨테이너를 직접 관리하면 AWS가 처리하던 문제도 직접 해결해야 합니다. 이 자료에는 실제 실행에서 확인한 문제와 해결 방법이 포함돼 있습니다.

실행 과정에서 확인한 대표적인 문제는 **`save_pretrained`로 저장한 Gemma 4 E2B/E4B 체크포인트가 vLLM, SGLang과 LMI에서 로드되지 않는 것**입니다. KV-sharing 레이어의 가중치가 저장 과정에서 빠지기 때문입니다. `scripts/train.py`가 저장 직전에 이를 복원하므로 학습 결과를 vLLM으로 서빙할 수 있습니다. 자세한 내용은 [`docs/05_serving_containers.md`](docs/05_serving_containers.md)에서 확인할 수 있습니다.

그 외에도 학습이 끝난 뒤 머지 단계에서 Job이 잘리는 문제, 24GB GPU에서의 서빙 OOM, 응답이 조용히 절단되는 문제 등을 `docs/`에 원인과 함께 정리해 두었습니다.

</details>

> 지원 범위와 리전 가용성은 빠르게 바뀝니다. 위 표는 방향을 잡기 위한 것이고, **실제 선택 전에 최신 문서와 콘솔에서 다시 확인**하세요. AWS 관리형 방식이 이미 지원하는 조합이라면 그쪽이 운영 부담이 적습니다.

---

## What it does

| Stage | What happens |
|---|---|
| 데이터 | 공개 permissive 시드 + Bedrock Converse로 grounded 합성 (critique/refine 루프) |
| 학습 | PyTorch DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA. 추출과 분류 코스는 SFT 후 GRPO 추가 학습 선택 가능 |
| 서빙 | SageMaker AI 실시간 엔드포인트. vLLM(기본), SGLang, DJL LMI 중 선택 (모두 OpenAI 호환) |
| 평가 | 학습에 쓰지 않은 데이터로 엔드포인트를 직접 호출해 코스별 지표 산출 |
| 에이전트 | Strands Agent에서 Bedrock Claude가 추론하고 파인튜닝한 SLM을 도구로 호출. AgentCore Runtime 배포까지 |
| 실험 추적 | 선택 기능. 파이프라인 실행의 설정, 평가 결과와 학습 지표를 MLflow에 기록 |

기본 모델은 `google/gemma-4-E4B-it`(apache-2.0, ungated)입니다. `MODEL_SIZE` 환경변수로 `E2B` / `E4B` / `12B` / `26B-A4B` / `31B`를 고르거나, `MODEL_ID`로 임의 모델을 지정할 수 있습니다.

학습 스크립트(`scripts/train.py`)는 독립적으로 구성돼 있으며 **로컬 GPU `--dry_run`과 SageMaker AI 학습 작업에서 같은 파일**을 씁니다. 클라우드에 제출하기 전에 로컬에서 학습 코드를 먼저 검증할 수 있습니다.

## Courses

| Course | Task | Seed dataset (license) |
|---|---|---|
| `01_extraction_to_json` | 텍스트에서 구조화 JSON 추출 | `glaiveai/glaive-function-calling-v2` (apache-2.0) |
| `02_classification` | 의도 분류 | `mteb/banking77` (mit) |
| `03_summarization` | 문서 요약 | `FiscalNote/billsum` (cc0-1.0) |
| `04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) |
| `05_multimodal_extraction` | 영수증 이미지에서 구조화 JSON 추출 | `naver-clova-ix/cord-v2` (cc-by-4.0) |

추출과 분류 코스는 GRPO 추가 학습을 선택할 수 있고, 멀티모달 코스는 합성 데이터와 에이전트 단계가 없습니다. 전체 노트북 순서와 선택 단계는 [`docs/RUN_E2E.md`](docs/RUN_E2E.md)에서 확인할 수 있습니다.

## 노트북 vs 스크립트

두 방식은 `common/` 모듈을 공유하지만 단계 구성과 상태 전달 방식이 다릅니다. 기본 실행 명령은 위 [Quick start](#quick-start)에서 확인할 수 있습니다.

| | 노트북 (`tracks/`) | 스크립트 (`pipelines/`) |
|---|---|---|
| 적합 | 처음 배울 때, 중간 산출물을 볼 때, 질의를 바꿔가며 볼 때 | 검증된 코스 재실행, CI, 무인 실행, 결과 재현 |
| 단계 전달 | `%store` (IPython 전용이고 **전역**) | 코스별 JSON 파일 (`.pipeline_state/`) |
| 설정 | 노트북 셀 상수 + `.env` | `config.yaml` + 환경변수 |
| 에이전트 단계 | 있음 (05, 06) | 없음 (노트북에만) |
| MLflow 실험 추적 | 자동 기록하지 않음 | 선택적으로 기록 |

에이전트 단계를 스크립트로 옮기지 않은 이유는 질의를 바꿔가며 응답을 확인하는 작업에 자동 실행이 적합하지 않기 때문입니다.

## Evaluation

`04_evaluate`가 학습에 쓰지 않은 데이터로 엔드포인트를 호출해 코스별 지표를 계산합니다.

| Course | Metrics |
|---|---|
| 추출 | arg F1 + valid JSON rate |
| 분류 | macro-F1 + accuracy |
| 요약 | ROUGE-L + LLM-as-judge |
| 도메인 QA | LLM-as-judge + ROUGE-L |

합성 데이터나 학습에 쓴 데이터로 평가하면 점수가 부풀려집니다. 평가 데이터는 학습 슬라이스 뒤쪽에서 따로 분리합니다.

## Repository layout

각 코스는 노트북, 시드 데이터 어댑터인 `track_data.py`, 학습과 로컬 서빙 코드를 담은 `scripts/`로 구성됩니다. 코드에서는 코스를 `track`이라는 이름으로 표현합니다.

```
sagemaker-finetune-serve-e2e/
├── common/     노트북이 공통으로 import 하는 얇은 레이어
├── tracks/     5개 코스 (각각 독립 E2E, 노트북)
├── pipelines/  같은 코스를 파이썬으로 한 번에 실행 (CI와 재현용)
├── docs/       파인튜닝, 서빙 컨테이너, 합성 데이터, 에이전트 가이드
├── tools/      노트북 셀 출력 정리, 노트북 빌더
├── agentcore/  Strands를 AgentCore Runtime에 배포하는 ARM64 컨테이너 스캐폴드
└── mlflow_setup.ipynb   모든 코스가 공유하는 MLflow App 설정
```

<details>
<summary><b>common/</b>: 공통 레이어</summary>

| File | Role |
|---|---|
| `config.py` | 모델 프리셋, 리전, 역할, 버킷, `DRY_RUN` (환경변수로 덮어쓸 수 있음) |
| `gemma_format.py` | messages 어댑터 (`apply_chat_template`에 위임) |
| `aws_utils.py` | 엔드포인트 호출(스트리밍 포함), Bedrock Converse, CloudWatch 링크 |
| `dlc.py` | DLC 이미지 해석과 엔진별(vLLM/SGLang/LMI) 서빙 환경변수 조립 |
| `display_utils.py` | 노트북 추론 결과 렌더링 |
| `eval_utils.py` | 코스별 평가 지표 |
| `grpo_data.py` | GRPO 프롬프트 소스 3종 (holdout / synth / failures) |
| `model_inspect.py` | 체크포인트가 vLLM으로 서빙 가능한지 판정 |
| `mlflow_utils.py` | 로컬 SQLite와 SageMaker Managed MLflow 연결, run 기록과 UI 안내 |
| `synth/` | grounded 합성 (Bedrock Converse + critique/refine) |

</details>

<details>
<summary><b>tracks/</b>: 코스별 대표 코드</summary>

**텍스트 코스 (01~04)**: 노트북 구성은 위 [Courses](#courses) 참고.

| File | Role |
|---|---|
| `track_data.py` | 시드 데이터셋을 로드하고 `{input, output}`을 messages로 변환 (코스마다 다름) |
| `scripts/train.py` | SFT. 로컬 `--dry_run`과 SageMaker AI 학습 작업에서 같은 파일을 씁니다 |
| `scripts/train_grpo.py` | SFT 후 GRPO 추가 학습. 보상을 프로그램으로 채점하는 추출과 분류 코스에서 사용 |
| `scripts/serve_local_vllm.sh` | 배포 전 로컬 vLLM으로 모델 로드 확인 |
| `scripts/bench_local_vllm.sh` | 로컬 처리량과 지연 측정 |
| `scripts/cleanup_local.sh` | 로컬에 받아 둔 모델과 벤치 산출물 정리 |
| `scripts/requirements.txt` | 학습 컨테이너 안에서 설치할 패키지 (로컬 환경과 별개) |

`train.py` / `train_grpo.py` / `*.sh`는 네 코스에서 **내용이 동일**합니다. 코스 차이는 `track_data.py`와 노트북의 하이퍼파라미터에 있습니다.

**멀티모달 코스 (05)**

| File | Role |
|---|---|
| `track_data.py` | cord-v2 로더 (images + messages) |
| `scripts/train_mm.py` | 멀티모달 SFT. vision tower freeze + language LoRA |
| `samples/` | 배포 검증용 영수증 2장과 정답 JSON (평가용 분리 데이터, 즉시 로드) |

</details>

## Setup

### 1) 의존성 설치

기본 설치는 위 [Quick start](#quick-start)에 있습니다. 그 외에 알아 둘 것:

```bash
uv lock --upgrade-package transformers    # 특정 패키지만 최신으로
```

pip만 쓸 경우: `pip install -r requirements.txt` (같은 floor 핀).

로컬 `transformers` 버전과 SageMaker AI 컨테이너 안의 버전은 **별개**입니다. 컨테이너 쪽은 `tracks/*/scripts/requirements.txt`가 설치하고, 이미지 자체는 `.env`의 `DLC_IMAGE_URI`가 결정합니다.

### 2) 설정과 시크릿

**설정**은 파일에, **시크릿**은 환경변수에 둡니다.

| | 어디에 | 예 |
|---|---|---|
| 모델 크기, 인스턴스, 서빙 엔진, 이미지 태그, 샘플 수, epoch | [`config.yaml`](config.yaml) (커밋됨) | `model.size: E4B` |
| 같은 값들 (노트북 경로) | `.env` (커밋됨, 시크릿 없음) | `TRAIN_INSTANCE_TYPE=...` |
| HF 토큰, 역할 ARN, 리전 | 환경변수 / 셸 | 아래 참고 |

`config.yaml`에는 각 기본값을 선택한 이유가 주석으로 적혀 있습니다. 반복해서 사용할 설정은 이 파일에서 변경하고, 한 번만 다른 값을 사용하려면 명령 앞에 환경변수를 지정합니다. 같은 설정이 여러 곳에 있으면 셸 환경변수, `.env`, `config.yaml`, 코드 기본값 순서로 적용됩니다.

```bash
MODEL_SIZE=31B python pipelines/run_extraction.py --stages train
```

시크릿은 어느 파일에도 넣지 마세요. `config.yaml`에 시크릿 키가 있으면 로더가 경고하고 무시합니다.

```bash
# HF 토큰: gated 모델(gemma-3/2 등)을 쓸 때만 필요. gemma-4 계열은 불필요
hf auth login

# SageMaker AI 실행 역할: 비워 두면 IAM에서 자동 탐지
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>

# Bedrock 모델 ID (inference-profile prefix 필수). 기본값은 common/config.py 참고
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5

export DRY_RUN=1     # 노트북에서 데이터와 평가 규모 축소
```

### 3) 실험 추적 (선택)

한 번 실행하고 결과만 확인한다면 기본값인 `USE_MLFLOW=0`을 유지하면 됩니다. 설정을 바꾸며 여러 번 실행하거나 팀에서 결과를 비교하려면 MLflow를 켭니다.

```bash
# MLflow 없이 실행
USE_MLFLOW=0 python pipelines/run_extraction.py --stages all

# 로컬 SQLite에 기록
USE_MLFLOW=1 MLFLOW_TRACKING_URI=local \
  python pipelines/run_extraction.py --stages all

# 같은 리전의 MLflow App을 이름으로 찾아 기록
USE_MLFLOW=1 python pipelines/run_extraction.py --stages all
```

관리형 환경이 필요하면 `mlflow_setup.ipynb`에서 MLflow App을 준비합니다. App을 찾지 못하면 로컬 SQLite로 전환되며, 특정 App이나 기존 Tracking Server를 사용하려면 `MLFLOW_TRACKING_URI`에 ARN을 지정합니다.

추적은 `pipelines/` 실행에만 적용됩니다. 코스 노트북은 반복 실행할 때 불필요한 run이 쌓이지 않도록 자동 기록하지 않습니다. 사용 시점, 오픈소스 MLflow와의 차이, 관리형 환경 설정은 [`docs/mlflow.md`](docs/mlflow.md)에서 확인할 수 있습니다.

## License

이 리포의 코드와 문서는 [MIT](LICENSE)입니다. 아래는 **그와 별개로** 확인해야 하는 모델과 데이터셋 라이선스입니다.

- **Gemma 4** = apache-2.0 + ungated. 토큰 없이 받을 수 있고 use-restriction이 없습니다.
- **Gemma 3 / 2 / 3n** = 커스텀 Gemma Terms + gated. HF 토큰과 약관 수락이 필요하고, 파인튜닝과 서빙 산출물까지 use-restriction이 전파됩니다.
- 시드 데이터셋은 permissive만 선별했습니다 (NC/ND/라이선스 미선언 제외). dolly 등 share-alike 데이터의 파생물은 배포 시 조건을 확인하세요.

모델 카드와 데이터셋 라이선스는 재배포나 서빙 전에 다시 확인하는 것이 안전합니다. 모델 ID, SDK 버전, 리전 지원 여부는 자주 바뀝니다.

## How this was built

기본 설계와 의사 코드는 직접 작성하고, 생성 도구를 사용해 노트북과 스크립트의 반복 구조를 구현했습니다. 생성 결과는 사람이 검토하고 실제 실행 결과를 다시 규칙과 코드에 반영했습니다.

실행 검증에서 발견한 대표적인 문제는 다음과 같습니다.

- **학습 작업이 병합 도중 중단되는 문제**: SDK가 `StoppingCondition`을 생략하면 1시간을 설정하며 이 제한에는 후처리도 포함됩니다. 상태는 `Stopped`이고 `FailureReason`은 비어 있어 실제 실행으로 확인해야 했습니다.
- **`save_pretrained`가 저장한 체크포인트를 vLLM이 못 읽는 문제**: KV-shared 레이어의 텐서 54개가 소실됩니다. 배포까지 가 봐야 나오는 문제였습니다.
- **두 dry-run의 의미가 달랐던 문제**: 파이프라인의 `--dry-run`은 과금 없이 전체 흐름을 검증하지만, 노트북의 `DRY_RUN=1`은 데이터와 평가 규모만 줄입니다. 현재 문서는 두 동작을 명확히 구분합니다.

## Disclaimer

이 가이드는 저자의 개인 견해와 실측 경험을 정리한 것으로, 저자가 재직 중인 회사의 공식 문서나 입장을 대변하지 않습니다. 내용이 공식 문서와 다를 경우 **공식 문서가 우선합니다.**
