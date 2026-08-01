# SageMaker Fine-tuning & Serving E2E

Gemma 4를 Amazon SageMaker에서 **파인튜닝 → 서빙 → 평가 → agentic loop**까지 잇는 한국어 핸즈온 자산입니다.
5개 트랙이 각각 독립된 E2E로 동작하므로, 필요한 태스크 하나만 골라 처음부터 끝까지 돌릴 수 있습니다.

- **가이드 사이트** → https://daekeun-ml.github.io/sagemaker-finetune-serve-e2e/ (검색 가능한 문서)
- **처음이라면** → [`GETTING_STARTED.md`](GETTING_STARTED.md) (설치 → 스모크 → 로컬 dry-run → 노트북)
- **전체를 한 번에 돌린다면** → [`docs/RUN_E2E.md`](docs/RUN_E2E.md) (E2E 실행 런북)

---

## Who is this for

**Good fit**

- 자체 데이터로 SLM을 파인튜닝해 **엔드포인트까지 올려 보고 싶은 분**. 데이터 준비부터 정리까지 한 트랙이면 끝납니다.
- 관리형 레시피가 **아직 지원하지 않는 모델·기법**을 써야 하는 분 (아래 [Why this asset](#why-this-asset) 참고).
- 학습 코드를 **직접 손대야 하는 분**. `scripts/train.py`가 self-contained라 TRL/PEFT 설정을 그대로 읽고 고칠 수 있습니다.
- SLM을 **에이전트의 tool로 붙이는 구조**를 보고 싶은 분 (reasoning은 Claude, 특화 추출은 SLM).
- SageMaker Python SDK **v3** 기준 코드가 필요한 분 (`ModelTrainer` / `ModelBuilder` — v2의 `Estimator`·`HuggingFaceModel`은 제거됨).

**Not a good fit**

- 파인튜닝 없이 **바로 배포만** 하려는 경우 → JumpStart가 더 빠릅니다.
- 관리형 레시피가 **이미 지원하는 조합**이라면 → SageMaker AI model customization / SageMaker Recipes가 운영 부담이 적습니다.
- 멀티노드 대규모 사전학습 → HyperPod 영역입니다. 이 자산은 단일 GPU LoRA/QLoRA 기준입니다.

전제 지식은 Python과 Jupyter 사용 경험 정도면 충분합니다. SageMaker·Bedrock 개념은 노트북에서 설명합니다.

## Requirements

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

**비용 감각** — 기본값(200건 x 2 epoch)이면 학습 1회가 20분대이고, 엔드포인트는 시간당 과금입니다. 트랙 하나를 학습부터 정리까지 도는 데 대략 몇 달러 수준입니다(리전·인스턴스에 따라 다름). **엔드포인트를 지우지 않으면 계속 과금되므로** `99_cleanup.ipynb`를 반드시 실행하세요.

## Why this asset

Gemma 4는 SageMaker의 관리형 경로에서도 쓸 수 있습니다. 그런데 **경로마다 지원 범위가 다르고, 모델 사이즈별로도 갈립니다.** 이 자산은 그 틈을 메웁니다.

| Path | Fine-tuning | Notes |
|---|---|---|
| **JumpStart** | 불가 (배포만) | 클릭 몇 번으로 엔드포인트. gemma-4 5종 모두 `training_supported=False`로 확인 |
| **SageMaker AI model customization** | 가능 | 관리형 SFT/DPO/RFT. 지원 모델·기법·리전이 한정 |
| **SageMaker Recipes** (Training Job / HyperPod) | 가능 | 검증된 레시피로 SFT/DPO/GRPO. 지원 모델 목록에 의존 |
| **이 자산 (BYOS · DLC + 커스텀 스크립트)** | 가능 | 모델·기법·하이퍼파라미터를 **직접 제어**. 대신 코드를 관리해야 함 |

직접 경로를 택하는 이유는 셋입니다.

1. **지원 목록에 없는 조합을 쓸 수 있습니다.** 관리형 경로는 지원 모델·기법·리전이 정해져 있고 자주 바뀝니다. 원하는 조합이 빠져 있으면 기다리는 수밖에 없습니다.
2. **학습 코드가 열려 있습니다.** LoRA target, chat template, packing, reward 함수까지 읽고 고칠 수 있습니다. 관리형은 노출된 하이퍼파라미터 범위 안에서만 조정됩니다.
3. **함정이 코드에 반영돼 있습니다.** 직접 경로에는 관리형이 대신 처리해 주던 문제가 남습니다 — 이 자산은 실제로 겪고 고친 것들을 담고 있습니다.

가장 대표적인 함정: **`save_pretrained`로 저장한 gemma-4 E2B/E4B 체크포인트는 vLLM·SGLang·LMI에서 로드가 실패합니다.** KV-sharing 레이어의 가중치가 저장 과정에서 빠지기 때문입니다. `scripts/train.py`가 저장 직전에 이를 복원하므로, 학습 결과를 그대로 vLLM으로 서빙할 수 있습니다 (상세: [`docs/05_serving_containers.md`](docs/05_serving_containers.md)).

그 외에도 학습이 끝난 뒤 머지 단계에서 잡이 잘리는 문제, 24GB GPU에서의 서빙 OOM, 응답이 조용히 절단되는 문제 등을 `docs/`에 원인과 함께 정리해 두었습니다.

> 지원 범위와 리전 가용성은 빠르게 바뀝니다. 위 표는 방향을 잡기 위한 것이고, **실제 선택 전에 최신 문서와 콘솔에서 다시 확인**하세요. 관리형 경로가 이미 지원하는 조합이라면 그쪽이 운영 부담이 적습니다.

---

## What it does

| Stage | What happens |
|---|---|
| 데이터 | 공개 permissive 시드 + Bedrock Converse로 grounded 합성 (critique/refine 루프) |
| 학습 | PyTorch DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA. 추출·분류 트랙은 SFT→GRPO 정련 선택 가능 |
| 서빙 | SageMaker real-time endpoint — vLLM(기본) / SGLang / DJL LMI 중 선택 (셋 다 OpenAI 호환) |
| 평가 | held-out 세트로 endpoint를 직접 호출해 트랙별 지표 산출 (합성·학습셋 사용 금지) |
| Agent | Strands Agent — reasoning은 Bedrock Claude, 파인튜닝한 SLM은 tool로 호출. AgentCore Runtime 배포까지 |

기본 모델은 `google/gemma-4-E4B-it`(apache-2.0, ungated)입니다. `MODEL_SIZE` 환경변수로 `E4B` / `12B` / `26B-A4B`를 고르거나, `MODEL_ID`로 임의 모델을 지정할 수 있습니다.

학습 스크립트(`scripts/train.py`)는 self-contained이며 **로컬 GPU `--dry_run`과 SageMaker 학습 잡에서 같은 파일**을 씁니다. 클라우드에 제출하기 전에 로컬에서 파이프라인을 먼저 검증할 수 있습니다.

## Tracks

| Track | Task | Seed dataset (license) |
|---|---|---|
| `01_extraction_to_json` | 텍스트 → 구조화 JSON 추출 | `glaiveai/glaive-function-calling-v2` (apache-2.0) |
| `02_classification` | 의도 분류 | `mteb/banking77` (mit) |
| `03_summarization` | 문서 요약 | `FiscalNote/billsum` (cc0-1.0) |
| `04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) |
| `05_multimodal_extraction` | 이미지 → 구조화 JSON (영수증) | `naver-clova-ix/cord-v2` (cc-by-4.0) |

**텍스트 트랙(01~04) 노트북 순서**

```
00_setup → 01_data_and_synthetic → 02_train_sft_sagemaker → 03_deploy_endpoint
        → 04_evaluate → 05_agentic_strands → 06_agentcore_deploy → 99_cleanup
```

- `02a_train_grpo_sagemaker` (선택): 추출·분류 트랙만. reward를 프로그램으로 채점할 수 있는 태스크에 적용합니다.
- `02b_local_serve` (선택): 배포 전에 로컬 vLLM으로 모델이 뜨는지 확인합니다.

**멀티모달 트랙(05)** 은 이미지 입력이라 구조가 다릅니다 — 합성 단계가 없고, endpoint가 이미지를 받습니다.

```
00_setup → 01_data_explore → 02_train_mm_sagemaker → 03_deploy_mm_endpoint → 99_cleanup
```

## Evaluation

`04_evaluate`가 held-out 세트로 endpoint를 호출해 트랙별 지표를 계산합니다.

| Track | Metrics |
|---|---|
| 추출 | arg F1 + valid JSON rate |
| 분류 | macro-F1 + accuracy |
| 요약 | ROUGE-L + LLM-as-judge |
| 도메인 QA | LLM-as-judge + ROUGE-L |

합성 데이터나 학습에 쓴 데이터로 평가하면 점수가 부풀려집니다. held-out은 학습 슬라이스 뒤쪽에서 따로 떼어 씁니다.

## Repository layout

각 트랙은 노트북 + `track_data.py`(시드 어댑터) + `scripts/`(학습·서빙 코드)로 같은 구조를 갖습니다.

```
sagemaker-finetune-serve-e2e/
├── common/     노트북이 공통으로 import 하는 얇은 레이어
├── tracks/     5개 트랙 (각각 독립 E2E)
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
| `eval_utils.py` | 트랙별 평가 지표 |
| `grpo_data.py` | GRPO 프롬프트 소스 3종 (holdout / synth / failures) |
| `model_inspect.py` | 체크포인트가 vLLM으로 서빙 가능한지 판정 |
| `synth/` | grounded 합성 (Bedrock Converse + critique/refine) |

</details>

<details>
<summary><b>tracks/</b> — 트랙별 대표 코드</summary>

**텍스트 트랙 (01~04)** — 노트북 구성은 위 [Tracks](#tracks) 참고.

| File | Role |
|---|---|
| `track_data.py` | 시드 데이터셋 로드 + `{input, output}` → messages 변환 (트랙마다 다름) |
| `scripts/train.py` | SFT. 로컬 `--dry_run`과 SageMaker 학습 잡에서 같은 파일을 씁니다 |
| `scripts/train_grpo.py` | SFT→GRPO 정련. reward를 프로그램으로 채점하는 추출·분류 트랙에서 사용 |
| `scripts/serve_local_vllm.sh` | 배포 전 로컬 vLLM으로 모델 로드 확인 |
| `scripts/bench_local_vllm.sh` | 로컬 처리량·지연 측정 |
| `scripts/cleanup_local.sh` | 로컬에 받아 둔 모델·벤치 산출물 정리 |
| `scripts/requirements.txt` | 학습 컨테이너 안에서 설치할 패키지 (로컬 환경과 별개) |

`train.py` / `train_grpo.py` / `*.sh`는 네 트랙에서 **내용이 동일**합니다. 트랙 차이는 `track_data.py`와 노트북의 하이퍼파라미터에 있습니다.

**멀티모달 트랙 (05)**

| File | Role |
|---|---|
| `track_data.py` | cord-v2 로더 (images + messages) |
| `scripts/train_mm.py` | 멀티모달 SFT — vision tower freeze + language LoRA |
| `samples/` | 배포 검증용 영수증 2장 + 정답 JSON (held-out, 즉시 로드) |

</details>

## Setup

### 1) 의존성 설치

```bash
# uv 미설치 시
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/sagemaker-finetune-serve-e2e
uv sync                     # uv.lock 기반 재현 설치 (권장)
source .venv/bin/activate

# 최신 버전으로 올리려면
uv lock --upgrade-package transformers
```

pip만 쓸 경우: `pip install -r requirements.txt` (같은 floor 핀).

로컬 `transformers` 버전과 SageMaker 컨테이너 안의 버전은 **별개**입니다. 컨테이너 쪽은 `tracks/*/scripts/requirements.txt`가 설치하고, 이미지 자체는 `.env`의 `DLC_IMAGE_URI`가 결정합니다.

### 2) 환경변수

`.env`에 설정값이 들어 있습니다 (인스턴스 타입, DLC 이미지 URI, 리전, 합성 건수). VS Code에서 이 폴더를 워크스페이스로 열면 자동 로드됩니다.

시크릿은 `.env`에 넣지 마세요.

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

기본 설계와 의사 코드(pseudo code)는 직접 작성했고, 이를 바탕으로 Claude Code를 활용해 노트북과 스크립트를 생성했습니다.
생성된 코드는 모두 직접 검수했으며, 학습·배포·추론을 실제로 실행해 동작을 확인한 뒤 반영했습니다.
