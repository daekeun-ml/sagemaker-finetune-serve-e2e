# E2E 실행 가이드: 한 코스를 처음부터 끝까지 완주하기

!!! info "문서 범위"
    한 코스를 **클라우드에서 한 번에 완주**하려는 분을 위한 실행 문서입니다.
    - **선행 조건**: AWS 계정, Amazon SageMaker AI 실행 역할, Bedrock 모델 접근 권한이 필요합니다. 설치와 스모크 테스트는 [시작하기](getting_started.md), 기본 개념은 [SageMaker AI 기초](01_sagemaker_basics.md)에서 확인할 수 있습니다.
    - **다루는 내용**: 실행 순서, 단계별 준비물, 단계 간 결과 전달, 비용과 다음 단계로 넘어가기 전 확인 사항입니다.
    - **다루지 않는 내용**: 개념 배경, 하이퍼파라미터 튜닝과 컨테이너 내부 구현입니다.

이 문서는 **실행 순서와 확인 지점**만 담습니다. "왜 이렇게 하는가"는 [전체 지도](00_overview.md)와 각 주제 가이드에 있습니다.

!!! warning "빠르게 바뀌는 값"
    모델 ID, DLC 이미지 태그, SDK API, 인스턴스 가용성과 리전별 GPU 용량은 바뀔 수 있습니다. 문서의 예시값은 실행 전에 공식 문서와 AWS 콘솔에서 다시 확인하고, 계정 ID, 역할 ARN과 HF 토큰은 문서나 노트북에 저장하지 마십시오.

---

## TL;DR

**태스크별 코스 하나의 노트북을 `00_setup`부터 `99_cleanup`까지 실행하면 한 번의 E2E가 완성됩니다. 스크립트 경로는 먼저 `--dry-run`으로 과금 없이 검증하고, 노트북 경로는 `DRY_RUN=1`로 데이터와 평가 규모를 줄여 확인하십시오.**

폴더 이름과 코드 식별자(`tracks/`, `TRACKS`, `track_data.py`)는 초기 이름인 `track`을 그대로 쓰고 있습니다. 이 문서에서 말하는 **코스**와 같은 것을 가리킵니다.

정리하면 다음과 같습니다.

1. **코스 하나가 완결된 E2E입니다.** 5개 코스는 서로 독립이므로 한 코스를 끝내고 정리한 뒤 다음 코스로 옮기세요([5개 코스를 모두 돌리려면](#5개-코스를-모두-돌리려면)).
2. **핸드오프는 `%store`와 코스 로컬 파일로 이어집니다.** 어느 노트북이 무엇을 만들어 넘기는지는 [단계별 실행과 데이터 핸드오프](#단계별-실행과-데이터-핸드오프)의 표에 정리했습니다.
3. **두 dry-run의 의미가 다릅니다.** 파이프라인의 `--dry-run`은 과금 리소스를 만들지 않지만, 노트북의 `DRY_RUN=1`은 일부 데이터 규모만 줄입니다([두 가지 검증 모드 구분](#두-가지-검증-모드-구분)).
4. **막히는 지점은 거의 정해져 있습니다.** `%store` 오염, `MaxRuntimeExceeded`, 24GB GPU CUDA OOM과 Bedrock 모델 ID 형식이 대표적이며, 해결 방법은 [E2E 흐름에서 자주 막히는 곳](#e2e-흐름에서-자주-막히는-곳)에 정리돼 있습니다.
5. **실시간 엔드포인트는 삭제 전까지 시간당 과금됩니다.** 중간에 멈추더라도 엔드포인트가 실행 중이면 `99_cleanup`을 먼저 실행해야 합니다([비용과 정리](#비용과-정리)).

---

## 완주할 때 자주 막히는 지점

완주를 시도한 사람이 실제로 자주 겪는 것들입니다.

- "노트북이 10개인데 **어디서 시작해서 어디서 끝나는 건가요?**": 코스마다 선택 단계가 섞여 있어 필수 경로가 안 보입니다.
- "`02`에서 `train_path`가 없다고 합니다.": 앞 노트북을 건너뛰었거나, `%store` 값이 **다른 코스 것**입니다.
- "학습 Job이 `Completed`인데 **배포가 안 됩니다.**": 머지 단계가 시간 제한에 잘려 서빙용 모델이 아티팩트에 없습니다.
- "설정을 건드리지 않았는데 **endpoint가 `Failed`로 끝났어요.**": 24GB GPU에서 엔진 기본값이 메모리를 넘겼습니다.
- "**얼마 나올지 모르겠어서** 시작이 무섭습니다.": 무엇이 상시 과금이고 무엇이 호출당 과금인지 구분이 안 됩니다.
- "테스트만 했는데 **다음 날 청구서가 왔습니다.**": endpoint를 지우지 않았습니다.

이 가이드는 위 문제를 실행 순서에 맞춰 확인할 수 있도록 구성했습니다.

---

## 두 가지 검증 모드 구분

이 저장소에는 이름이 비슷하지만 동작이 다른 두 검증 방식이 있습니다.

| 실행 경로 | 설정 | 동작 | 과금 |
|---|---|---|---|
| 파이프라인 | `python pipelines/run_extraction.py --stages all --dry-run` | AWS 작업을 제출하지 않고 전체 단계와 상태 전달을 검증 | 없음 |
| 노트북 | 셸 또는 `.env`에서 `DRY_RUN=1` | 시드, 합성, 평가와 로컬 학습 규모를 축소 | Bedrock 호출, SageMaker 학습 작업과 엔드포인트는 실행한 만큼 발생 |

노트북의 `DRY_RUN=1`은 시드를 8건, 합성을 6건, 평가를 20건으로 줄이고 로컬 `train.py --dry_run`을 1 epoch와 최대 32행으로 제한합니다. 하지만 `02_train_sft_sagemaker`의 학습 규모는 노트북의 `MAX_TRAIN_SAMPLES`와 `EPOCHS`가 결정하며, `03_deploy_endpoint`를 실행하면 실제 GPU 엔드포인트가 생성됩니다.

처음에는 파이프라인 `--dry-run`으로 설정과 단계 연결을 확인하고, 노트북을 사용할 때는 `DRY_RUN=1`과 작은 학습 설정으로 한 번 실행한 뒤 실제 규모로 늘리는 순서를 권장합니다.

---

## 두 가지 실행 방법

같은 코스를 **노트북**과 **파이썬 스크립트**로 실행할 수 있습니다. 두 경로는 `common/` 모듈을 공유하지만 단계 구성과 상태 전달 방식이 다릅니다.

| | 노트북 (`tracks/`) | 스크립트 (`pipelines/`) |
|---|---|---|
| 적합 | 처음 배울 때, 중간 산출물을 눈으로 볼 때, 질의를 바꿔가며 볼 때 | 검증된 코스를 다시 돌릴 때, CI, 무인 실행, 결과 재현 |
| 실행 | JupyterLab에서 셀 순서대로 | `python pipelines/run_extraction.py --stages all` |
| 단계 전달 | `%store` (IPython 전용, **전역**) | 코스별 JSON 파일 (`.pipeline_state/`) |
| 설정 | 노트북 셀 상수 + `.env` | `config.yaml` + 환경변수 |
| 에이전트 단계 | 있음 (05, 06) | 없음: 노트북에만 |
| MLflow 실험 추적 | 자동 기록하지 않음 | 선택적으로 기록 |

```bash
# 나눠서 실행: 학습만 돌려두고 나중에 배포
python pipelines/run_extraction.py --stages data,train
python pipelines/run_extraction.py --stages deploy,eval
```

!!! tip "먼저 --dry-run"
    파이프라인의 `--dry-run`은 학습 작업, 엔드포인트와 Bedrock 호출을 만들지 않으므로 AWS 자격증명 없이 전체 단계를 검증할 수 있습니다.

### 파이프라인 실험 추적

MLflow는 파이프라인 경로에서만 자동 기록됩니다. 한 번 실행하고 끝내면 `USE_MLFLOW=0`을 유지하고, 반복 실험을 비교하려면 로컬 SQLite나 SageMaker Managed MLflow를 사용합니다.

```bash
# 기록하지 않음
USE_MLFLOW=0 python pipelines/run_extraction.py --stages all

# 로컬 SQLite에 기록
USE_MLFLOW=1 MLFLOW_TRACKING_URI=local \
  python pipelines/run_extraction.py --stages all

# 같은 리전의 MLflow App을 자동 탐색
USE_MLFLOW=1 python pipelines/run_extraction.py --stages all
```

관리형 환경은 루트의 `mlflow_setup.ipynb`에서 준비합니다. App을 찾지 못하면 로컬 SQLite로 전환되며, 설정 우선순위와 기록 범위는 [MLflow로 파인튜닝 실험 비교하기](mlflow.md)에 정리돼 있습니다.

파이프라인의 자세한 사용법은 [`pipelines/README.md`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/pipelines/README.md)에 있습니다. 아래 절은 **노트북 경로**를 기준으로 설명합니다.

## 파이프라인 한눈에

각 노트북은 결과를 `%store`로 저장하고 다음 노트북이 `%store -r`로 읽습니다. 다만 `train.jsonl`처럼 **코스 간 오염이 치명적인 값은 코스 로컬 파일로 고정**해 두었습니다([단계별 실행과 데이터 핸드오프](#단계별-실행과-데이터-핸드오프) 표 참고).

### 텍스트 코스 (01~04) 파이프라인

```
00_setup
01_data_and_synthetic
02_train_sft_sagemaker
  02a_train_grpo_sagemaker  # 선택, 추출과 분류
  02b_local_serve           # 선택, 로컬 서빙 확인
03_deploy_endpoint
04_evaluate
05_agentic_strands
06_agentcore_deploy
99_cleanup
```

- **(선택) `02a_train_grpo_sagemaker`**는 SFT 후 GRPO 추가 학습을 실행하며, 보상을 프로그램으로 채점할 수 있는 **추출과 분류 코스에만** 있습니다.
- **(선택) `02b_local_serve`**(배포 전 로컬 vLLM 프리플라이트)는 4개 텍스트 코스 모두에 있습니다.
- 평가는 `04_evaluate`(로컬 메트릭, 빠름, 저렴) 한 경로입니다.

??? info "관리형 evaluator를 쓰지 않는 이유"
    SDK v3의 관리형 evaluator(`BenchMarkEvaluator`/`CustomScorerEvaluator`/`LLMAsJudgeEvaluator`)는 **SageMaker Public Hub에 평가 레시피가 등록된 모델(Amazon Nova, 일부 JumpStart)** 전용입니다.
    gemma-4 커스텀 파인튜닝 산출물(S3 체크포인트)에는 쓸 수 없습니다. 실측하면 `DescribeHubContent ... does not exist`로 떨어집니다.

### 멀티모달 코스 (05) 파이프라인

멀티모달 코스는 **노트북 구성이 다르고 더 짧습니다**(이미지 입력, 합성과 에이전트 단계 없음).

```
00_setup
01_data_explore
02_train_mm_sagemaker
03_deploy_mm_endpoint
99_cleanup
```

- **시드**: `naver-clova-ix/cord-v2`(cc-by-4.0, ungated). 합성 데이터 단계가 없어 이미지+JSON 시드를 그대로 씁니다.
- **학습**: `scripts/train_mm.py`에서 `AutoModelForImageTextToText`와 `AutoProcessor`를 사용하고, vision/audio 파라미터를 동결한 뒤 language 계층만 [LoRA](https://huggingface.co/docs/peft/index)로 학습합니다.
- **서빙**: 이미지 입력을 받는 멀티모달 엔드포인트이며, 텍스트 전용 모델로 다시 내보내지 않습니다.

---

## 사전 준비 체크리스트

한 번만 하면 되는 준비입니다.

- [ ] **설치 완료**: 저장소 루트에서 `uv sync`를 실행합니다. 자세한 절차와 대체 설치 방법은 [시작하기](getting_started.md), 현재 의존성 하한은 `pyproject.toml`에서 확인할 수 있습니다.
- [ ] **AWS 자격증명**: `aws sts get-caller-identity`가 계정을 반환하는지 확인합니다.
- [ ] **SageMaker AI 실행 역할**: `SAGEMAKER_ROLE_ARN`에 SageMaker AI, S3와 ECR 권한이 있는지 확인합니다. 역할 자동 탐지에 성공해도 필요한 권한이 모두 있다는 뜻은 아니며, 권한 부족은 학습 작업이 시작된 뒤 드러날 수 있습니다([실행 역할이 매개하는 것](01_sagemaker_basics.md#실행-role로-무엇을-하는가-s3와-ecr-접근)).
- [ ] **Bedrock 모델 접근**: 콘솔에서 Claude 모델 접근을 활성화하고, [교차 리전 추론](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)에 사용하는 inference profile ID를 `BEDROCK_CLAUDE_MODEL_ID`에 설정합니다.
- [ ] **모델 선택**: 기본값은 `google/gemma-4-E4B-it`이며, `MODEL_SIZE`나 `MODEL_ID`로 변경할 수 있습니다.
- [ ] **HF 토큰**: Gemma 4에는 필요하지 않습니다. gated 모델을 사용할 때만 약관을 수락하고 `hf auth login`으로 인증합니다.
- [ ] **리전 정합성**: SageMaker AI, Bedrock, S3와 `.env`의 DLC 이미지 URI가 `AWS_REGION`과 일치하는지 확인합니다.
- [ ] **MLflow 사용 여부**: 실험 추적이 필요 없으면 `USE_MLFLOW=0`을 유지합니다. 관리형 MLflow를 사용하려면 `mlflow_setup.ipynb`와 [MLflow 가이드](mlflow.md)를 먼저 확인합니다.
- [ ] **비용 인지**: 실시간 엔드포인트는 삭제 전까지 시간당 과금되므로, 실습이 끝나면 `99_cleanup`을 반드시 실행합니다.

```bash
export AWS_REGION=us-west-2                                  # .env의 DLC URI 리전과 일치해야 합니다
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5   # 콘솔에서 현행 ID 재확인
# export MODEL_IS_GATED=1 && export HF_TOKEN=hf_...          # gated 모델을 쓸 때만
export DRY_RUN=1             # 노트북에서 데이터와 평가 규모 축소
```

VS Code로 이 리포 폴더를 워크스페이스로 열면 `.env`가 커널 env로 자동 주입됩니다(인스턴스 타입과 DLC 이미지 URI, 리전, 합성 건수 등 설정값). 시크릿은 `.env`에 넣지 말고 셸 export나 `hf auth login`을 쓰세요.

!!! tip "LiteLLM이 필요하면 별도 환경에"
    에이전트 단계는 Bedrock을 boto3로 직접 호출하므로 **LiteLLM 없이 완결됩니다**. `common/llm_gateway.py`를 사용해야 한다면 별도 환경에 설치하세요(`uv pip install -e '.[litellm]'`, 현재 하한 `litellm>=1.75.9`).
    코어에서 뺀 이유는 litellm이 요구하는 `importlib-metadata>=8`이 sagemaker의 `<7`과 하드 충돌하기 때문입니다.

---

## 단계별 실행과 데이터 핸드오프

`jupyter lab`을 실행하고 코스 폴더(`tracks/<track>/`)에서 번호 순서대로 진행합니다. 아래 표는 각 단계의 작업, 전달 결과와 완료 조건을 정리합니다.

| # | 노트북 | 하는 일 | 다음으로 넘기는 것 | 완료 확인 |
|---|---|---|---|---|
| 1 | `00_setup` | 설치, 자격증명, 역할과 버킷 확인 | `%store`: `role`, `bucket` | 계정 ID, 역할과 버킷이 정상적으로 출력됨 |
| 2 | `01_data_and_synthetic` | 시드 로드, grounded 합성, EDA | 코스 로컬 파일 `data/train.jsonl` | JSONL 형식과 토큰 길이가 정상임 |
| 3 | `02_train_sft_sagemaker` | 선택적 로컬 dry-run 후 [TRL `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) 기반 SageMaker AI 학습 | `%store`: `model_data`, `md_<track_key>` | 학습 작업이 `Completed`이고 CloudWatch 링크가 출력됨 |
| 3-a | `02a_train_grpo_sagemaker` | 선택 단계. 추출과 분류 코스에서 SFT 모델을 GRPO로 추가 학습 | 갱신된 `model_data` | 학습 작업이 `Completed` |
| 3-b | `02b_local_serve` | 선택 단계. 배포 전 로컬 vLLM 확인 | 없음 | 로컬 호출이 정상 응답 |
| 4 | `03_deploy_endpoint` | [실시간 엔드포인트](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) 배포와 호출 테스트 | `%store`: `endpoint_name`, `ep_<track_key>` | `InService` 상태와 정상 응답 확인 |
| 5 | `04_evaluate` | 학습에서 분리한 평가 데이터로 지표 계산 | 없음 | `arg_f1`, `macro_f1`, ROUGE-L 또는 judge 점수 출력 |
| 6 | `05_agentic_strands` | [Strands](https://github.com/strands-agents/sdk-python)에서 SLM과 Bedrock Claude 연계 | 없음 | 에이전트 응답 정상 |
| 7 | `06_agentcore_deploy` | [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html) 배포 | 없음 | 선택적으로 Runtime 호출 확인 |
| 8 | `99_cleanup` | 엔드포인트, 설정, 모델과 Runtime 삭제 | 없음 | 해당 코스의 엔드포인트가 남지 않음 |

학습 작업과 엔드포인트 생성은 **SageMaker AI에서 진행되므로 커널이나 세션이 끊겨도 계속됩니다.** 각 노트북의 재접속 셀에서 작업 이름이나 엔드포인트 이름으로 다시 연결할 수 있습니다.

### 단계별 주의

- **② 합성**: `NUM_SYNTHETIC`가 Bedrock 호출량, 즉 비용을 좌우합니다. 설정 기본값은 200이고, 이 저장소의 `.env`는 요약 코스 지연 때문에 100으로 낮춰 두었습니다.
    합성 전에 토큰 길이 EDA를 꼭 보세요. 학습이 자르는 단위는 문자가 아니라 토큰이고, 한국어와 JSON은 문자당 토큰 수가 영어의 몇 배입니다.
- **③ 학습**: `stopping_condition`을 **반드시 명시**하세요. 생략하면 SDK 기본 1시간이 붙습니다.
    [StoppingCondition API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)가 적는 API 기본값 1일과는 다릅니다.
    학습 이미지는 `.env`의 `DLC_IMAGE_URI`를 우선 사용합니다. 이 값이 없으면 `DLC_REPOSITORY`와 `DLC_TAG`를 조합하고, 그래도 결정할 수 없으면 라이브러리 버전 조합으로 해석합니다.
    태그는 자주 갱신되므로 실행 직전에 [DLC available images](https://aws.github.io/deep-learning-containers/reference/available_images/)에서 현행 태그를 확인하세요.
    첫 실행은 용량 대기(Pending)와 이미지 pull(Downloading) 때문에 시작이 느립니다(실측 각 6분과 3분).
- **④ 배포**: 기본 경로는 **vLLM DLC(`SERVING_ENGINE=vllm`)** 이고, `sglang` 또는 `lmi`(`OPTION_*` 환경변수)로 전환할 수 있습니다.
    셋 다 연속 배칭 + OpenAI 호환 `messages` 스키마라 호출 코드가 동일합니다.
    **한 번에 하나만** 배포하세요. 여러 엔드포인트를 동시에 실행하면 비용이 중복되며, 엔드포인트 기동에는 5~15분이 걸립니다.
    참고할 곳: 엔진 선택 기준은 [서빙 엔진 선택: SERVING_ENGINE](05_serving_containers.md#서빙-엔진-선택-serving_engine), 메모리 예산은 [메모리 예산: L4 22.9GB 실측](05_serving_containers.md#메모리-예산-l4-229gb-실측), 호출 스키마는 [SageMaker AI 추론](04_sagemaker_inference.md#invoke_endpoint-호출-스키마).
- **⑤ 평가**: 평가 데이터는 학습에 쓴 앞 구간(`NUM_SEED_SAMPLES`, 기본 300건)을 **명시적으로 건너뛴 뒤** 분리합니다.
    `pool[-N:]` 방식은 위험합니다. 예를 들어 `N_EVAL=50`인데 150건만 로드하면 평가 데이터가 학습 구간 안에 포함될 수 있습니다.
- **⑥/⑦ 에이전트**: 엔드포인트와 Bedrock이 **각각 과금**됩니다.
    엔드포인트는 `sagemaker-runtime`, Bedrock은 `bedrock-runtime`의 [`converse`](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)를 사용하며 서로 별개 서비스입니다.
    AgentCore는 GA 상태와 리전을 재확인하세요([production 배포](06_agentic.md#production-배포-agentcore-runtime)).
- **⑧ 정리**: 중간에 멈추더라도 endpoint가 떠 있으면 `99_cleanup`이 먼저입니다. 그러지 않으면 계속 과금됩니다.

??? info "②에서 `NUM_SYNTHETIC`를 100으로 낮춘 근거"
    요약 시드는 중앙값이 1건당 1,651자로, 추출 코스의 475자보다 길어서 배치 프롬프트가 약 10,900자까지 늘어납니다.
    출력은 2,554토큰으로 `max_tokens` 4,500 안이라 절단은 없습니다. 즉 문제는 품질이 아니라 순수 지연입니다.

!!! danger "평가셋은 합성으로 만들지 마세요"
    합성 데이터나 학습셋으로 평가하면 점수가 조용히 부풀려집니다. teacher 모델을 얼마나 모방했는지를 재는 데 그칩니다.
    `04_evaluate`는 반드시 **합성 증강 이전의 시드에서 결정론적으로 분리한 평가 데이터**만 사용합니다. 자세한 배경은 [평가 데이터 분리 원칙](02_synthetic_data.md#held-out-규율-합성으로-평가-금지)에 있습니다.

위 단계들의 근거를 원본 소스에서 확인하려면 다음을 보세요.

??? info "더 읽을 거리"
    - [aws/deep-learning-containers](https://github.com/aws/deep-learning-containers): 학습과 서빙 DLC의 Dockerfile과 `sagemaker_entrypoint.sh`가 환경변수를 CLI 플래그로 변환하는 규칙을 확인할 때.
    - [SageMaker 모델 배포 옵션 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html): 이 가이드가 ④에서 실시간 추론을 선택한 배경을 확인할 때.
    - [InvokeEndpoint API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html): ④ 이후 호출에 걸리는 파라미터와 payload 한도의 원문.
    - ④에서 고르는 세 엔진의 저장소와 문서: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [DJL LMI](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html). 지원 모델과 플래그는 문서보다 저장소가 빠릅니다.

---

## 5개 코스를 모두 돌리려면

코스는 **독립**입니다. 한 코스를 완주하고 정리한 뒤, 다른 코스 폴더에서 같은 순서를 반복하면 됩니다.

```
tracks/01_extraction_to_json/    텍스트에서 JSON 추출, 처음 시작할 코스
tracks/02_classification/        (intent 분류)
tracks/03_summarization/         (문서 요약)
tracks/04_domain_qa/             (도메인 QA)
tracks/05_multimodal_extraction/ 이미지에서 JSON 추출, 별도 구조
```

- 텍스트 코스(01~04)는 위 [텍스트 코스 파이프라인](#텍스트-코스-0104-파이프라인)의 `00_setup`부터 `99_cleanup`까지 순서대로 실행합니다.
- 멀티모달 코스(05)는 [멀티모달 코스 파이프라인](#멀티모달-코스-05-파이프라인)의 5단계 세트를 씁니다.
- 코스마다 **별도 엔드포인트**가 생성되므로 각 코스의 `99_cleanup`을 각각 실행해야 합니다.
- 공통 로직은 `common/`이 공유하므로, 텍스트 코스 간 차이는 데이터 어댑터(`tracks/*/track_data.py`)와 `config.TRACKS` 레지스트리뿐입니다.
- 여러 코스를 동시에 띄우면 GPU 인스턴스 비용이 코스 수만큼 늘어납니다. **한 코스씩 완주하고 정리하는 방식**을 권장합니다.

??? question "오해: “코스를 옮기면 `%store` 값도 알아서 바뀌겠지?”"
    **그렇지 않습니다.** `%store`는 IPython 프로필 단위라 **코스를 넘어 공유**됩니다. 전역 `endpoint_name`/`model_data`는 마지막에 실행한 코스 값이 남아, 엉뚱한 endpoint를 호출하거나 다른 코스 모델을 배포하게 됩니다.
    그래서 노트북은 코스 전용 키(`ep_<track_key>`, `md_<track_key>`)를 먼저 읽고, `train_path`는 아예 코스 로컬 파일(`data/train.jsonl`)로 고정합니다.

---

## E2E 흐름에서 자주 막히는 곳

| 증상 | 원인과 해결 |
|---|---|
| `02`에서 `train_path`/`data/train.jsonl` 없음 | `01`을 실행하지 않았습니다. 코스 내 노트북은 **순서대로** 실행하세요 |
| 다른 코스 엔드포인트를 호출하거나 옛 모델이 배포됨 | `%store` 전역 키가 오염된 경우입니다. 코스 전용 키(`ep_<track_key>`, `md_<track_key>`)를 사용하고, 리전을 바꿨다면 `aws_utils.ensure_model_data_in_region()`으로 이전 리전 아티팩트를 걸러냅니다 |
| 학습 작업이 시작 직후 실패 | IAM 역할 권한이나 DLC 태그를 확인합니다. CloudWatch 로그, `.env`의 `DLC_IMAGE_URI` 리전과 `aws ecr describe-images` 결과를 차례로 확인하십시오 |
| 학습이 끝났지만 작업이 `Stopped`이고 병합 모델이 없음 | `stopping_condition`의 실행 상한에 병합 단계가 포함되지 못한 경우입니다. `MAX_RUNTIME_HOURS`를 충분히 늘리고 자세한 원인은 [파인튜닝](03_finetuning.md)에서 확인합니다 |
| `InsufficientInstanceCapacity`로 학습 작업이 시작되지 않음 | 리전별 GPU 용량 문제입니다. `AWS_REGION`을 바꾸고 `.env`의 DLC 이미지 URI 리전도 함께 변경합니다 |
| 엔드포인트가 `Failed`이고 `did not pass the ping health check`만 표시됨 | CUDA OOM인 경우가 많습니다. `serving_env()`의 `max_num_seqs=32`, `gpu_memory_utilization=0.90` 기본값을 유지하고 CloudWatch 엔드포인트 로그를 확인합니다([24GB GPU CUDA OOM](04_sagemaker_inference.md#24gb-gpu-cuda-oom-max_num_seqs-기본값)) |
| gated 모델 다운로드 401 | HF 약관과 토큰을 확인하고 `MODEL_IS_GATED=1`과 `HF_TOKEN` 또는 `hf auth login`을 사용합니다. 토큰이 필요 없는 `google/gemma-4-E4B-it`으로 바꿀 수도 있습니다 |
| Bedrock `converse` 400 | inference profile 접두사가 있는 모델 ID를 사용하고 콘솔에서 모델 접근 권한을 확인합니다 |
| 엔드포인트 응답이 반복되거나 품질이 낮음 | 원문 문자열 대신 `messages` 스키마를 보내 서버가 chat template을 적용하게 합니다(`aws_utils.invoke_sagemaker_chat`) |
| 응답이 중간에 끊김 | `max_tokens`와 `finish_reason`을 확인합니다([max_tokens 절단과 finish_reason](05_serving_containers.md#max_tokens-절단과-finish_reason)) |
| speculative decoding이 켜지지 않음 | 이 kit 노트북에는 배선돼 있지 않습니다. 컨테이너 설정 키는 vLLM DLC `SM_VLLM_SPECULATIVE_CONFIG` / LMI `OPTION_SPECULATIVE_CONFIG`이며, **Gemma용 P-EAGLE head는 AWS가 공개하지 않았고** 커뮤니티 EAGLE3 head는 fine-tuned target과의 정합성을 직접 실측해야 합니다([Speculative decoding (EAGLE3 / P-EAGLE)](05_serving_containers.md#speculative-decoding-eagle3--p-eagle)) |
| `litellm` import 오류 | sagemaker와 `importlib-metadata` 요구 버전이 충돌하므로 별도 환경에 설치합니다 |
| 비용이 계속 나감 | 엔드포인트나 AgentCore Runtime이 남아 있는지 확인하고 `99_cleanup`과 콘솔에서 정리합니다 |

---

## 완료 기준

한 코스 E2E가 "됐다"고 말할 수 있는 조건입니다.

- [ ] `02` 학습 Job이 `Completed`이고, 아티팩트 루트에 **머지된 서빙용 모델**이 있음(어댑터만 있으면 배포되지 않습니다)
- [ ] `03` invoke 스모크가 의미 있는 출력을 반환
- [ ] `04_evaluate` 지표가 나옴 (가능하면 파인튜닝 전 baseline과 비교해 개선 폭 확인)
- [ ] (선택) `05_agentic_strands`에서 Claude가 SLM 엔드포인트를 도구로 호출하는 왕복이 성공
- [ ] (프로덕션 목표 시) `06_agentcore_deploy`로 Runtime 배포 확인
- [ ] (MLflow 사용 시) 파이프라인 상위 run과 학습 하위 run이 의도한 실험에 기록됨
- [ ] `99_cleanup`을 실행한 뒤 해당 코스 접두사의 엔드포인트가 콘솔에 남아 있지 않음

---

## 비용과 정리

!!! danger "비용과 정리"
    **실시간 엔드포인트는 삭제하기 전까지 GPU 인스턴스 요금이 계속 부과됩니다.** 호출이 없어도 과금되므로 실습이 끝나면 모든 코스의 `99_cleanup`을 실행하고 콘솔에서 남은 엔드포인트가 없는지 확인하십시오. 여러 번 배포했다면 `%store`에는 마지막 `endpoint_name`만 남으므로 코스 접두사와 사용했던 모든 리전을 기준으로 잔여 리소스를 확인해야 합니다.

삭제 순서는 **엔드포인트, 엔드포인트 설정, 모델**입니다. 앞 단계의 리소스를 먼저 삭제하지 않으면 다음 리소스가 사용 중인 상태로 남습니다.

모델 이름은 `ModelBuilder`가 `model-42c30d1e`처럼 자동 생성하므로 `endpoint_name`만으로 찾을 수 없습니다. `99_cleanup`은 엔드포인트 설정에서 실제 `ModelName`을 조회한 뒤 삭제합니다.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker AI 실시간 엔드포인트 | 인스턴스 시간당, 삭제 전까지 계속 | `99_cleanup`에서 엔드포인트, 엔드포인트 설정, 모델 순서로 삭제 |
| SageMaker AI 학습 작업 | 작업 실행 시간만 과금 | 작업 종료 시 과금 중단 |
| Bedrock Converse (합성, 에이전트, 평가) | 호출 토큰량 기준, 상주 리소스 없음 | 합성 건수와 평가 샘플 수로 조절 |
| AgentCore Runtime | 배포한 경우 Runtime 리소스 과금 | `bash agentcore/cleanup_agent.sh --aws`(Runtime + ECR) |
| 로컬 `local_model/`, vLLM 프로세스 | 과금 없음(디스크 약 15GB, GPU 점유) | `bash scripts/cleanup_local.sh --yes` |

각 노트북은 학습과 배포 직후 **CloudWatch 링크**를 출력합니다(`common/aws_utils.cw_links()`). 학습 로그, 엔드포인트 기동, OOM과 Bedrock 호출량을 여기서 확인할 수 있습니다.

??? question "오해: “엔드포인트를 호출하지 않으면 요금도 안 나오죠?”"
    **아닙니다.** 실시간 엔드포인트는 호출 여부와 관계없이 프로비저닝된 인스턴스에 대해 과금됩니다. 사용하지 않으면 삭제해야 하며, 자세한 내용은 [비용과 정리](04_sagemaker_inference.md#비용과-cleanup)에서 확인할 수 있습니다.

---

## 관련 리포지토리 파일

설정과 공통 유틸:

- `common/config.py`: 전역 설정 로더. `MODEL_SIZE` 프리셋, `SERVING_ENGINE`, `is_dry_run()`, `TRACKS` 레지스트리
- `common/dlc.py`: DLC 이미지 URI 해석과 서빙 환경변수 생성(`serving_env`)
- `common/aws_utils.py`: 엔드포인트 호출(`invoke_sagemaker_chat`), CloudWatch 링크, 리전 정합성 검사와 비용 경고
- `common/mlflow_utils.py`: MLflow 대상 선택, run 기록, 학습 컨테이너 환경변수와 UI 안내
- `.env`: 인스턴스 타입과 DLC 이미지 URI, 리전, 합성 건수 등 비시크릿 설정값
- `mlflow_setup.ipynb`: SageMaker Managed MLflow App 생성과 연결 확인

학습 스크립트(코스 폴더에 자족적으로 들어 있음):

- `tracks/*/scripts/train.py`: SFT와 LoRA/QLoRA 학습, 병합 후 텍스트 모델 저장
- `tracks/*/scripts/train_grpo.py`: SFT 산출물을 보상 함수로 추가 학습하는 GRPO 스크립트(추출과 분류 코스만)
- `tracks/05_multimodal_extraction/scripts/train_mm.py`: 멀티모달 SFT(`AutoProcessor`, vision 동결, 텍스트 모델 변환 없음)

평가와 정리:

- `common/eval_utils.py`: 코스별 지표(`arg_f1`/`macro_f1`/ROUGE-L/LLM-judge)
- `tracks/*/99_cleanup.ipynb`: 엔드포인트, 엔드포인트 설정과 모델 삭제
- `agentcore/cleanup_agent.sh`: AgentCore Runtime + ECR 정리(`--aws`)
