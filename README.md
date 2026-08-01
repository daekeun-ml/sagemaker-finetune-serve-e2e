# SageMaker Fine-tuning & Serving E2E

[Quick start](#quick-start) | [Courses](#courses) | [Why this kit](#why-this-kit) | [Setup](#setup) | [Docs](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/) | [Cost & cleanup](#cost--cleanup)

Gemma 4를 Amazon SageMaker에서 **합성 데이터 → 파인튜닝 → 서빙 → 평가 → agentic loop**까지 잇는 한국어 핸즈온 kit입니다.
**태스크별 실습 코스** 5개가 각각 독립된 E2E로 동작하므로, 필요한 태스크 하나만 골라 처음부터 끝까지 돌릴 수 있습니다.

### 📘 [가이드북 바로가기 →](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/)

**코드를 따라 치는 대신, 왜 그렇게 하는지 이해하고 넘어가려는 분을 위한 가이드북입니다.**
SageMaker가 처음이어도 읽을 수 있게 개념부터 시작하고, 각 절이 **어디서 막히는지(pain point) → 왜 그런지(why) → 그래서 이 값·이 구조인지**로 이어집니다.
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

몇 초 안에 끝나고 **과금되는 것을 하나도 만들지 않습니다.** 학습 Job·endpoint는 물론 Bedrock 호출도 하지 않으므로, AWS 자격증명이 없어도 전 경로를 밟아 볼 수 있습니다.

### 3) 실제로 돌리기 — 두 가지 방법

**스크립트 (Python)** — 검증된 코스를 다시 돌릴 때, CI, 무인 실행

```bash
python pipelines/run_extraction.py --stages all       # 전 구간

python pipelines/run_extraction.py --stages data,train    # 학습만 하고
python pipelines/run_extraction.py --stages deploy,eval   # 나중에 배포

python pipelines/run_extraction.py --stages cleanup   # 끝나면 반드시
```

자세한 내용은 [`pipelines/README.md`](pipelines/README.md).

**노트북 (JupyterLab)** — 처음 배울 때, 중간 산출물을 눈으로 볼 때

```bash
jupyter lab
```

`tracks/01_extraction_to_json/`을 열어 `00_setup.ipynb`부터 번호 순서대로 실행합니다. 단계별 핸드오프와 비용 가드는 [`docs/RUN_E2E.md`](docs/RUN_E2E.md)에 있습니다.

> [!WARNING]
> **real-time endpoint는 지울 때까지 시간당 과금됩니다** — 요청이 없어도 인스턴스 비용이 발생합니다.
> 실습을 마치면 `cleanup` 단계나 `99_cleanup.ipynb`를 **반드시** 실행하세요.

**더 자세한 안내**

- **처음이라면** → [`docs/getting_started.md`](docs/getting_started.md) (설치 → 스모크 → 로컬 dry-run → 노트북)
- **설정을 바꾸려면** → [`config.yaml`](config.yaml) (모델 크기·인스턴스·서빙 엔진)
- **SDK v2에서 옮겨 왔다면** → [SageMaker Python SDK V3](https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/sdk_v3/) (V2와 무엇이 다른지)

---

## Who is this for

**Good fit**

- 자체 데이터로 SLM을 파인튜닝해 **엔드포인트까지 올려 보고 싶은 분**. 데이터 준비부터 정리까지 한 코스면 끝납니다.
- 관리형 레시피가 **아직 지원하지 않는 모델·기법**을 써야 하는 분 (아래 [Why this kit](#why-this-kit) 참고).
- 학습 코드를 **직접 손대야 하는 분**. `scripts/train.py`가 self-contained라 TRL/PEFT 설정을 그대로 읽고 고칠 수 있습니다.
- SLM을 **에이전트의 tool로 붙이는 구조**를 보고 싶은 분 (reasoning은 Claude, 특화 추출은 SLM).
- SageMaker Python SDK **v3** 기준 코드가 필요한 분 (`ModelTrainer` / `ModelBuilder` — v2의 `Estimator`·`HuggingFaceModel`은 제거됨).

**Not a good fit**

- 파인튜닝 없이 **바로 배포만** 하려는 경우 → SageMaker JumpStart가 더 빠릅니다.
- 관리형 레시피가 **이미 지원하는 조합**이라면 → SageMaker AI model customization / SageMaker Recipes가 운영 부담이 적습니다.
- 멀티노드 대규모 사전학습 → SageMaker HyperPod 영역입니다. 이 kit은 단일 GPU LoRA/QLoRA 기준입니다.

전제 지식은 Python과 Jupyter 사용 경험 정도면 충분합니다. SageMaker·Bedrock 개념은 노트북에서 설명합니다.

## Requirements

<details>
<summary><b>필요한 것 8가지</b> — AWS 계정 · IAM role · GPU 쿼터 · 리전 · Bedrock 접근 · Python · (선택) 로컬 GPU · (선택) HF 토큰</summary>

| Item | Notes |
|---|---|
| AWS 계정 | SageMaker 학습 잡·엔드포인트를 만들 수 있는 계정 (과금 발생) |
| IAM 실행 role | `AmazonSageMaker-ExecutionRole-*` 또는 동급 권한. 없으면 코드가 IAM에서 자동 탐지 |
| 서비스 쿼터 | 학습·추론용 GPU 인스턴스 (기본 `ml.g6.2xlarge`). 신규 계정은 **쿼터가 0일 수 있어 미리 증설 신청**이 필요합니다 |
| 리전 | 기본 `us-west-2`. GPU 용량과 Bedrock 모델 가용성이 리전마다 달라, 막히면 리전을 바꾸는 것이 가장 빠릅니다 |
| Bedrock 모델 접근 | 합성 데이터 생성과 LLM-as-judge 평가에 Claude 사용 → 콘솔에서 모델 접근 권한을 켜 두세요 |
| Python | 3.10 이상 (로컬은 노트북 실행용. 학습은 컨테이너 안에서 돕니다) |
| 로컬 GPU | 선택. 있으면 `--dry_run`으로 클라우드 제출 전에 파이프라인을 검증할 수 있습니다 |
| HF 토큰 | Gemma 4는 ungated라 불필요. gated 모델(Gemma 3 등)을 쓸 때만 `hf auth login` |

</details>

**비용 감각** — 기본값(200건 x 2 epoch)이면 학습 1회가 20분대이고, 엔드포인트는 시간당 과금입니다. 코스 하나를 학습부터 정리까지 도는 데 대략 몇 달러 수준입니다(리전·인스턴스에 따라 다름). **엔드포인트를 지우지 않으면 계속 과금되므로** `99_cleanup.ipynb`를 반드시 실행하세요.

## Why this kit

SageMaker에서 Gemma 4를 다루는 길은 여러 개입니다. 관리형 경로도 있고, 컨테이너를 직접 가져가는 길도 있습니다.
문제는 **길마다 되는 것과 안 되는 것이 다르고, 같은 길에서도 모델 사이즈에 따라 갈린다**는 점입니다. 이 kit은 관리형 경로가 아직 닿지 않는 조합을 맡습니다.

| Path | Fine-tuning | Notes |
|---|---|---|
| **SageMaker JumpStart** | 불가 (배포만) | 클릭 몇 번으로 엔드포인트. gemma-4 5종 모두 `training_supported=False`로 확인 |
| **SageMaker AI model customization** | 가능 | 관리형 SFT/DPO/RFT. 지원 모델·기법·리전이 한정 |
| **SageMaker Recipes** (Training Job / HyperPod) | 가능 | 검증된 레시피로 SFT/DPO/GRPO. 지원 모델 목록에 의존 |
| **이 kit (BYOC · DLC + 커스텀 스크립트)** | 가능 | 모델·기법·하이퍼파라미터를 **직접 제어**. 대신 코드를 관리해야 함 |

<details>
<summary><b>직접 경로를 택한 이유 3가지와, 그래서 만난 함정</b></summary>

1. **지원 목록에 없는 조합을 쓸 수 있습니다.** 관리형 경로는 지원 모델·기법·리전이 정해져 있고 자주 바뀝니다. 원하는 조합이 빠져 있으면 기다리는 수밖에 없습니다.
2. **학습 코드가 열려 있습니다.** LoRA target, chat template, packing, reward 함수까지 읽고 고칠 수 있습니다. 관리형은 노출된 하이퍼파라미터 범위 안에서만 조정됩니다.
3. **함정이 코드에 반영돼 있습니다.** 직접 경로에서는 관리형이 대신 처리해 주던 문제를 내가 맡게 됩니다 — 이 kit은 실제로 겪고 고친 것들을 담고 있습니다.

가장 대표적인 함정: **`save_pretrained`로 저장한 gemma-4 E2B/E4B 체크포인트는 vLLM·SGLang·LMI에서 로드가 실패합니다.** KV-sharing 레이어의 가중치가 저장 과정에서 빠지기 때문입니다. `scripts/train.py`가 저장 직전에 이를 복원하므로, 학습 결과를 그대로 vLLM으로 서빙할 수 있습니다 (상세: [`docs/05_serving_containers.md`](docs/05_serving_containers.md)).

그 외에도 학습이 끝난 뒤 머지 단계에서 잡이 잘리는 문제, 24GB GPU에서의 서빙 OOM, 응답이 조용히 절단되는 문제 등을 `docs/`에 원인과 함께 정리해 두었습니다.

</details>

> 지원 범위와 리전 가용성은 빠르게 바뀝니다. 위 표는 방향을 잡기 위한 것이고, **실제 선택 전에 최신 문서와 콘솔에서 다시 확인**하세요. 관리형 경로가 이미 지원하는 조합이라면 그쪽이 운영 부담이 적습니다.

---

## What it does

| Stage | What happens |
|---|---|
| 데이터 | 공개 permissive 시드 + Bedrock Converse로 grounded 합성 (critique/refine 루프) |
| 학습 | PyTorch DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA. 추출·분류 코스는 SFT→GRPO 정련 선택 가능 |
| 서빙 | SageMaker real-time endpoint — vLLM(기본) / SGLang / DJL LMI 중 선택 (셋 다 OpenAI 호환) |
| 평가 | held-out 세트로 endpoint를 직접 호출해 코스별 지표 산출 (합성·학습셋 사용 금지) |
| Agent | Strands Agent — reasoning은 Bedrock Claude, 파인튜닝한 SLM은 tool로 호출. AgentCore Runtime 배포까지 |

기본 모델은 `google/gemma-4-E4B-it`(apache-2.0, ungated)입니다. `MODEL_SIZE` 환경변수로 `E2B` / `E4B` / `12B` / `26B-A4B` / `31B`를 고르거나, `MODEL_ID`로 임의 모델을 지정할 수 있습니다.

학습 스크립트(`scripts/train.py`)는 self-contained이며 **로컬 GPU `--dry_run`과 SageMaker 학습 잡에서 같은 파일**을 씁니다. 클라우드에 제출하기 전에 로컬에서 파이프라인을 먼저 검증할 수 있습니다.

## Courses

| Course | Task | Seed dataset (license) |
|---|---|---|
| `01_extraction_to_json` | 텍스트 → 구조화 JSON 추출 | `glaiveai/glaive-function-calling-v2` (apache-2.0) |
| `02_classification` | 의도 분류 | `mteb/banking77` (mit) |
| `03_summarization` | 문서 요약 | `FiscalNote/billsum` (cc0-1.0) |
| `04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) |
| `05_multimodal_extraction` | 이미지 → 구조화 JSON (영수증) | `naver-clova-ix/cord-v2` (cc-by-4.0) |

**텍스트 코스(01~04) 노트북 순서**

```
00_setup → 01_data_and_synthetic → 02_train_sft_sagemaker → 03_deploy_endpoint
        → 04_evaluate → 05_agentic_strands → 06_agentcore_deploy → 99_cleanup
```

- `02a_train_grpo_sagemaker` (선택): 추출·분류 코스만. reward를 프로그램으로 채점할 수 있는 태스크에 적용합니다.
- `02b_local_serve` (선택): 배포 전에 로컬 vLLM으로 모델이 뜨는지 확인합니다.

**멀티모달 코스(05)** 는 이미지 입력이라 구조가 다릅니다 — 합성 단계가 없고, endpoint가 이미지를 받습니다.

```
00_setup → 01_data_explore → 02_train_mm_sagemaker → 03_deploy_mm_endpoint → 99_cleanup
```

## 노트북 vs 스크립트

둘은 같은 `common/` 레이어를 쓰므로 결과가 같고, 쓰는 상황이 다릅니다. 실행 명령은 위 [Quick start](#quick-start) 참고.

| | 노트북 (`tracks/`) | 스크립트 (`pipelines/`) |
|---|---|---|
| 적합 | 처음 배울 때, 중간 산출물을 볼 때, 질의를 바꿔가며 볼 때 | 검증된 코스 재실행, CI, 무인 실행, 결과 재현 |
| 단계 전달 | `%store` (IPython 전용이고 **전역**) | 코스별 JSON 파일 (`.pipeline_state/`) |
| 설정 | 노트북 셀 상수 + `.env` | `config.yaml` + env(시크릿만) |
| agentic 단계 | 있음 (05, 06) | 없음 — 노트북에만 |

agentic 단계를 스크립트로 옮기지 않은 이유는, 질의를 바꿔가며 응답을 보는 성격이라 스크립트로 얻는 것이 없기 때문입니다.

## Evaluation

`04_evaluate`가 held-out 세트로 endpoint를 호출해 코스별 지표를 계산합니다.

| Course | Metrics |
|---|---|
| 추출 | arg F1 + valid JSON rate |
| 분류 | macro-F1 + accuracy |
| 요약 | ROUGE-L + LLM-as-judge |
| 도메인 QA | LLM-as-judge + ROUGE-L |

합성 데이터나 학습에 쓴 데이터로 평가하면 점수가 부풀려집니다. held-out은 학습 슬라이스 뒤쪽에서 따로 떼어 씁니다.

## Repository layout

각 코스는 노트북 + `track_data.py`(시드 어댑터) + `scripts/`(학습·서빙 코드)로 같은 구조를 갖습니다. 디렉터리 이름 `tracks/`와 `track_data.py`·`TRACKS` 같은 코드 식별자는 초기 명칭을 그대로 유지하고 있으므로, 본문의 "코스"와 코드의 `track`은 같은 것을 가리킵니다.

```
sagemaker-finetune-serve-e2e/
├── common/     노트북이 공통으로 import 하는 얇은 레이어
├── tracks/     5개 코스 (각각 독립 E2E, 노트북)
├── pipelines/  같은 코스를 파이썬으로 한 번에 실행 (CI·재현용)
├── docs/       파인튜닝 · 서빙 컨테이너 · 합성 데이터 · agentic 가이드
├── tools/      노트북 셀 출력 정리
└── agentcore/  ARM64 컨테이너 스캐폴드 (Strands → AgentCore Runtime)
```

<details>
<summary><b>common/</b> — 공통 레이어</summary>

| File | Role |
|---|---|
| `config.py` | 모델 프리셋, 리전, role/bucket, DRY_RUN (전부 env 오버라이드) |
| `gemma_format.py` | messages 어댑터 (`apply_chat_template`에 위임) |
| `aws_utils.py` | endpoint 호출(스트리밍 포함) · Bedrock Converse · CloudWatch 링크 |
| `dlc.py` | DLC 이미지 해석 + 엔진별(vLLM/SGLang/LMI) 서빙 env 조립 |
| `display_utils.py` | 노트북 추론 결과 렌더링 |
| `eval_utils.py` | 코스별 평가 지표 |
| `grpo_data.py` | GRPO 프롬프트 소스 3종 (holdout / synth / failures) |
| `model_inspect.py` | 체크포인트가 vLLM으로 서빙 가능한지 판정 |
| `synth/` | grounded 합성 (Bedrock Converse + critique/refine) |

</details>

<details>
<summary><b>tracks/</b> — 코스별 대표 코드</summary>

**텍스트 코스 (01~04)** — 노트북 구성은 위 [Courses](#courses) 참고.

| File | Role |
|---|---|
| `track_data.py` | 시드 데이터셋 로드 + `{input, output}` → messages 변환 (코스마다 다름) |
| `scripts/train.py` | SFT. 로컬 `--dry_run`과 SageMaker 학습 잡에서 같은 파일을 씁니다 |
| `scripts/train_grpo.py` | SFT→GRPO 정련. reward를 프로그램으로 채점하는 추출·분류 코스에서 사용 |
| `scripts/serve_local_vllm.sh` | 배포 전 로컬 vLLM으로 모델 로드 확인 |
| `scripts/bench_local_vllm.sh` | 로컬 처리량·지연 측정 |
| `scripts/cleanup_local.sh` | 로컬에 받아 둔 모델·벤치 산출물 정리 |
| `scripts/requirements.txt` | 학습 컨테이너 안에서 설치할 패키지 (로컬 환경과 별개) |

`train.py` / `train_grpo.py` / `*.sh`는 네 코스에서 **내용이 동일**합니다. 코스 차이는 `track_data.py`와 노트북의 하이퍼파라미터에 있습니다.

**멀티모달 코스 (05)**

| File | Role |
|---|---|
| `track_data.py` | cord-v2 로더 (images + messages) |
| `scripts/train_mm.py` | 멀티모달 SFT — vision tower freeze + language LoRA |
| `samples/` | 배포 검증용 영수증 2장 + 정답 JSON (held-out, 즉시 로드) |

</details>

## Setup

### 1) 의존성 설치

기본 설치는 위 [Quick start](#quick-start)에 있습니다. 그 외에 알아 둘 것:

```bash
uv lock --upgrade-package transformers    # 특정 패키지만 최신으로
```

pip만 쓸 경우: `pip install -r requirements.txt` (같은 floor 핀).

로컬 `transformers` 버전과 SageMaker 컨테이너 안의 버전은 **별개**입니다. 컨테이너 쪽은 `tracks/*/scripts/requirements.txt`가 설치하고, 이미지 자체는 `.env`의 `DLC_IMAGE_URI`가 결정합니다.

### 2) 설정과 시크릿

**설정**은 파일에, **시크릿**은 환경변수에 둡니다.

| | 어디에 | 예 |
|---|---|---|
| 모델 크기 · 인스턴스 · 서빙 엔진 · 이미지 태그 · 샘플 수 · epoch | [`config.yaml`](config.yaml) (커밋됨) | `model.size: E4B` |
| 같은 값들 (노트북 경로) | `.env` (커밋됨, 시크릿 없음) | `TRAIN_INSTANCE_TYPE=...` |
| HF 토큰 · role ARN · 리전 | 환경변수 / 셸 | 아래 참고 |

`config.yaml`은 값의 **이유를 주석으로** 담고 있어서, 왜 `max_num_seqs`가 32인지 파일 안에서 알 수 있습니다. 우선순위는 **셸·`.env`의 기존 env > `config.yaml` > `common/config.py` 기본값**이라, 한 번만 바꿀 때는 셸에서 넘기면 됩니다.

```bash
MODEL_SIZE=31B python pipelines/run_extraction.py --stages train
```

시크릿은 어느 파일에도 넣지 마세요. `config.yaml`에 시크릿 키가 있으면 로더가 경고하고 무시합니다.

```bash
# HF 토큰 — gated 모델(gemma-3/2 등)을 쓸 때만 필요. gemma-4 계열은 불필요
hf auth login

# SageMaker 실행 role — 비워 두면 IAM에서 자동 탐지
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>

# Bedrock 모델 ID (inference-profile prefix 필수). 기본값은 common/config.py 참고
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5

export DRY_RUN=1     # 먼저 파이프라인만 검증
```

## Cost & cleanup

- **real-time endpoint는 삭제할 때까지 시간당 과금됩니다.** 실습을 마치면 반드시 `99_cleanup.ipynb`를 실행하세요.
- 학습 잡은 종료 시 과금이 멈춥니다. 노트북이 `stopping_condition`으로 상한을 걸어 두므로 잡이 무한히 도는 일은 없습니다.
- Bedrock Converse는 토큰 단위 과금입니다 (대량 합성 시 비용 발생). AgentCore Runtime도 배포해 두면 과금됩니다.
- 각 노트북은 학습·배포 직후 CloudWatch 다이렉트 링크를 출력합니다.

## License notes

- **Gemma 4** = apache-2.0 + ungated. 토큰 없이 받을 수 있고 use-restriction이 없습니다.
- **Gemma 3 / 2 / 3n** = 커스텀 Gemma Terms + gated. HF 토큰과 약관 수락이 필요하고, 파인튜닝·서빙 산출물까지 use-restriction이 전파됩니다.
- 시드 데이터셋은 permissive만 선별했습니다 (NC/ND/라이선스 미선언 제외). dolly 등 share-alike 데이터의 파생물은 배포 시 조건을 확인하세요.

모델 카드와 데이터셋 라이선스는 재배포·서빙 전에 다시 확인하는 것이 안전합니다. 모델 ID·SDK 버전·리전 지원 여부는 자주 바뀝니다.

## How this was built

기본 설계와 의사 코드(pseudo code)는 직접 작성했고, 이를 바탕으로 Claude Code로 노트북과 스크립트를 생성했습니다.

**백지에서 시작한 것이 아닙니다.** 그동안 직접 만들어 온 SageMaker 실습 에셋들에서 얻은 판단 기준 — 어떤 순서로 가르쳐야 이해되는지, 어디서 사람들이 막히는지, 어떤 값을 기본으로 둬야 안전한지 — 을 컨텍스트로 넣었습니다. 그 축적이 없으면 "돌아가는 코드"는 나오지만 "실습에 쓸 수 있는 코드"는 나오지 않습니다.

**그리고 한 번 생성해서 끝난 것도 아닙니다.** human-in-the-loop으로 돌렸습니다 — 생성 → 실제 실행 → 문제 발견 → 지시 수정 → 재생성을 반복했고, 그 과정에서 나온 판단 기준은 다시 규칙으로 정리해 다음 생성에 반영했습니다.

실제로 이 루프에서 잡아낸 것들입니다.

- **학습 Job이 머지 도중 죽는 문제** — SDK가 `StoppingCondition`을 생략하면 1시간을 넣는데, 그 창이 후처리까지 덮습니다. 상태는 `Stopped`이고 `FailureReason`은 비어 있어서, 실행해 보지 않으면 드러나지 않았습니다.
- **`save_pretrained`가 저장한 체크포인트를 vLLM이 못 읽는 문제** — KV-shared 레이어의 텐서 54개가 소실됩니다. 배포까지 가 봐야 나오는 문제였습니다.
- **`--dry-run`인데 실제로 과금되던 문제** — 건수만 줄이고 Bedrock 호출은 그대로여서, 합성 100건에 약 110회가 청구됐습니다. 실행 로그를 읽다 발견했습니다.
- **문서의 어색한 표현** — `재-export` 같은 조어, "근육기억"(muscle memory 직역), 문단 전체를 감싼 볼드 안의 볼드. 사람이 읽어 보고 지적한 뒤에야 고쳐졌습니다.

그래서 이 kit의 코드에는 **한 번에 맞힌 것보다 틀렸다가 고친 것이 더 많이 담겨 있습니다.** 주석에 "왜 이 값인가"가 자주 붙어 있는 이유도 그것입니다.

## Disclaimer

이 가이드는 저자의 개인 견해와 실측 경험을 정리한 것으로, 저자가 재직 중인 회사의 공식 문서나
입장을 대변하지 않습니다. 내용이 공식 문서와 다를 경우 **공식 문서가 우선합니다.**
