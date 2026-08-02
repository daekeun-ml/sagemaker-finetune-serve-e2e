# E2E 실행 runbook — 한 코스를 처음부터 끝까지 완주하기

!!! info "Scope"
    한 코스를 **클라우드에서 한 번에 완주**하려는 분을 위한 실행 문서입니다.

    - **선행 조건**: AWS 계정 + Amazon SageMaker AI 실행 role + Bedrock 모델 액세스.
      설치 절차와 스모크/로컬 dry-run은 [시작하기](getting_started.md).
      Training Job·Endpoint가 낯설면 [SageMaker AI 기초](01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: 실행 순서 · 단계별 준비물 · 단계 간 핸드오프 · 비용 ·
      다음으로 넘어가기 전 확인 지점
    - **여기서 다루지 않는 것**: 개념 배경(각 주제 가이드) · 학습 하이퍼파라미터 튜닝 ·
      컨테이너 내부 구조

이 문서는 **실행 순서와 확인 지점**만 담습니다. "왜 이렇게 하는가"는 [전체 지도](00_overview.md)와 각 주제 가이드에 있습니다.

!!! warning "빠르게 바뀌는 값"
    모델 ID·DLC 이미지 태그·SDK v3 API 이름·인스턴스 타입 가용성·리전 GPU 용량·AgentCore GA 상태는 분기마다 바뀝니다.
    이 문서의 태그·ID·인스턴스 타입은 전부 **실행 직전 재확인** 대상입니다(코드에도 `# TODO verify`로 표기).
    확인처는 각 절에 인라인으로 링크한 공식 문서입니다. 이 문서의 수치는 실측 스냅샷이므로 원문 쪽이 항상 최신입니다.
    실제 값은 `.env`와 셸 env로 주입합니다. 계정 ID·role ARN·HF 토큰은 문서에도 노트북에도 하드코딩하지 않습니다.

---

## TL;DR

**태스크별 실습 코스 하나(`tracks/<track>/` 폴더)의 노트북을 `00 → 99` 순서로 실행하면 1개 E2E가 완성됩니다. 첫 완주는 `DRY_RUN=1`로 소량·저비용으로 돌려 파이프라인을 검증하고, 두 번째 완주에서만 본격 과금하세요.**

폴더 이름과 코드 식별자(`tracks/`, `TRACKS`, `track_data.py`)는 초기 이름인 `track`을 그대로 쓰고 있습니다. 이 문서에서 말하는 **코스**와 같은 것을 가리킵니다.

정리하면 다음과 같습니다.

1. **코스 하나가 완결된 E2E입니다.** 5개 코스는 서로 독립이므로 한 코스를 끝내고 정리한 뒤 다음 코스로 옮기세요([5개 코스를 모두 돌리려면](#5개-코스를-모두-돌리려면)).
2. **핸드오프는 `%store`와 코스 로컬 파일로 이어집니다.** 어느 노트북이 무엇을 만들어 넘기는지는 [단계별 실행과 데이터 핸드오프](#단계별-실행과-데이터-핸드오프)의 표에 정리했습니다.
3. **두 번 완주가 정석입니다.** 1차는 `DRY_RUN=1`, 2차는 실제 규모입니다([왜 두 번 완주하는가](#왜-두-번-완주하는가--dry_run-우선)).
4. **막히는 지점은 거의 정해져 있습니다.** `%store` 오염, `MaxRuntimeExceeded`, 24GB GPU CUDA OOM, Bedrock 모델 ID 형식이 전부입니다.
    증상별 정리는 [E2E 흐름에서 자주 막히는 곳](#e2e-흐름에서-자주-막히는-곳)에 있습니다.
5. **real-time endpoint는 삭제 전까지 시간당 과금됩니다.** 중간에 멈추더라도 endpoint가 떠 있으면 `99_cleanup`이 먼저입니다([비용과 cleanup](#비용과-cleanup)).

---

## 기존 Pain Point

완주를 시도한 사람이 실제로 자주 겪는 것들입니다.

- "노트북이 10개인데 **어디서 시작해서 어디서 끝나는 건가요?**": 코스마다 선택 단계가 섞여 있어 필수 경로가 안 보입니다.
- "`02`에서 `train_path`가 없다고 합니다.": 앞 노트북을 건너뛰었거나, `%store` 값이 **다른 코스 것**입니다.
- "학습 Job이 `Completed`인데 **배포가 안 됩니다.**": 머지 단계가 시간 제한에 잘려 서빙용 모델이 아티팩트에 없습니다.
- "설정을 건드리지 않았는데 **endpoint가 `Failed`로 끝났어요.**": 24GB GPU에서 엔진 기본값이 메모리를 넘겼습니다.
- "**얼마 나올지 모르겠어서** 시작이 무섭습니다.": 무엇이 상시 과금이고 무엇이 호출당 과금인지 구분이 안 됩니다.
- "테스트만 했는데 **다음 날 청구서가 왔습니다.**": endpoint를 지우지 않았습니다.

이 runbook은 위 여섯 가지를 실행 순서 안에서 하나씩 막습니다.

---

## 왜 두 번 완주하는가 — DRY_RUN 우선

!!! abstract "쉽게 말하면"
    새 레시피로 손님상을 차리기 전에 **재료를 조금만 써서 한 번 만들어 보는 것**과 같습니다.
    1차 완주는 "불이 켜지나, 순서가 맞나"만 보고, 2차 완주에서 제대로 된 양을 씁니다.
    파이프라인이 깨지는 지점은 대부분 데이터 양과 무관한 곳(권한·리전·핸드오프·스키마)이라, 소량으로도 거의 다 걸립니다.

| 축 | 1차 완주 (`DRY_RUN=1`) | 2차 완주 (`DRY_RUN=0`) |
|---|---|---|
| 목적 | 자격증명·권한·핸드오프·스키마 검증 | 실제 성능 확보 |
| 시드·합성 건수 | 시드 8건 / 합성 6건 | 시드 300건 / 합성 100~200건 |
| 학습 | 로컬 `train.py --dry_run`(1 epoch·32행) | SageMaker AI 학습 잡(`MAX_TRAIN_SAMPLES`·`EPOCHS`) |
| 평가 | `N_EVAL=20` | `N_EVAL` 기본 50 (env로 조절) |
| endpoint | 작게 띄우고 바로 `99_cleanup` | 평가·agentic까지 쓰고 정리 |
| 비용 | Bedrock 호출 소량 + endpoint 수십 분 | 학습 Job + endpoint 실사용 시간 |

### DRY_RUN이 바꾸는 것 3가지

1. **합성·시드 건수** — `01_data_and_synthetic`이 시드 8건, 합성 6건으로 줄입니다(`config.is_dry_run()`).
    평상시 값은 `NUM_SEED_SAMPLES=300`, `NUM_SYNTHETIC`(config 기본 200, 이 리포의 `.env`는 100)입니다.
2. **평가 건수** — `04_evaluate`가 `N_EVAL=20`으로 축소합니다.
3. **로컬 학습 규모** — `train.py --dry_run`이 epoch 1, `max_seq_length ≤ 512`, 최대 32행으로 자릅니다.

!!! warning "SageMaker AI 학습 Job 규모는 DRY_RUN이 줄이지 않습니다"
    `02_train_sft_sagemaker`가 제출하는 클라우드 학습 Job의 크기는 노트북 안의 `MAX_TRAIN_SAMPLES`(핸즈온 기본 200)와 `EPOCHS`(기본 2)가 결정합니다.
    `DRY_RUN`은 데이터 준비·평가·로컬 dry-run에만 걸리므로, 클라우드 학습 비용을 줄이려면 이 두 값을 직접 낮추세요.

그래서 1차 완주도 "무료 리허설"은 아닙니다. 무엇이 실제로 과금되는지는 다음 착각에서 갈립니다.

??? question "오개념 — “DRY_RUN=1이면 과금이 아예 없는 거죠?”"
    **아닙니다.** `DRY_RUN=1`은 **양을 줄이는 스위치**일 뿐입니다. Bedrock 합성 호출은 건수만큼 과금되고, `03_deploy_endpoint`를 실행하면 GPU endpoint가 실제로 떠서 시간당 과금됩니다.
    비용이 0인 검증은 `tests/test_smoke.py`(순수 로직)와 로컬 GPU dry-run뿐입니다.

---

## 두 가지 실행 방법

같은 코스를 **노트북**으로도, **파이썬 스크립트**로도 돌릴 수 있습니다. 둘은 같은 `common/`
레이어를 쓰므로 결과가 같고, 쓰는 상황이 다릅니다.

| | 노트북 (`tracks/`) | 스크립트 (`pipelines/`) |
|---|---|---|
| 적합 | 처음 배울 때, 중간 산출물을 눈으로 볼 때, 질의를 바꿔가며 볼 때 | 검증된 코스를 다시 돌릴 때, CI, 무인 실행, 결과 재현 |
| 실행 | JupyterLab에서 셀 순서대로 | `python pipelines/run_extraction.py --stages all` |
| 단계 전달 | `%store` (IPython 전용, **전역**) | 코스별 JSON 파일 (`.pipeline_state/`) |
| 설정 | 노트북 셀 상수 + `.env` | `config.yaml` + env(시크릿만) |
| 에이전트 단계 | 있음 (05, 06) | 없음 — 노트북에만 |

```bash
# 나눠서 실행 — 학습만 돌려두고 나중에 배포
python pipelines/run_extraction.py --stages data,train
python pipelines/run_extraction.py --stages deploy,eval
```

!!! tip "먼저 --dry-run"
    `--dry-run`은 **과금되는 것을 하나도 만들지 않습니다.** 학습 Job·endpoint는 물론 Bedrock 호출도
    하지 않습니다. Bedrock은 토큰당 과금이라 합성 100건이면 생성 10회 + critique 약 100회가 실제로
    청구됩니다. 그래서 몇 초에 끝나고 AWS 자격증명 없이도 전 경로를 밟습니다.

자세한 사용법은 [`pipelines/README.md`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/pipelines/README.md)에 있습니다.
아래 절은 **노트북 경로**를 기준으로 설명합니다.

## 파이프라인 한눈에

각 노트북은 결과를 `%store`로 저장하고 다음 노트북이 `%store -r`로 읽습니다.
다만 `train.jsonl`처럼 **코스 간 오염이 치명적인 값은 코스 로컬 파일로 고정**해 두었습니다([단계별 실행과 데이터 핸드오프](#단계별-실행과-데이터-핸드오프) 표 참고).

### 텍스트 코스 (01~04) 파이프라인

```
00_setup ──▶ 01_data_and_synthetic ──▶ 02_train_sft_sagemaker ──▶ 03_deploy_endpoint
  (role,        (data/train.jsonl)         (model_data)              (endpoint_name)
   bucket)                                       │                         │
                                        (선택) 02a_train_grpo              ▼
                                        (선택) 02b_local_serve
                        04_evaluate ──▶ 05_agentic_strands ──▶ 06_agentcore_deploy
                        (held-out 점수)   (SLM+Claude 루프)       (프로덕션 배포)
                                                                          │
                                                                          ▼
                                                                     99_cleanup
```

- **(선택) `02a_train_grpo_sagemaker`**(SFT→GRPO 정련)는 리워드를 프로그램으로 채점할 수 있는 **추출·분류 코스에만** 있습니다.
- **(선택) `02b_local_serve`**(배포 전 로컬 vLLM 프리플라이트)는 4개 텍스트 코스 모두에 있습니다.
- 평가는 `04_evaluate`(로컬 메트릭·빠름·저렴) 한 경로입니다.

??? info "관리형 evaluator를 쓰지 않는 이유"
    SDK v3의 관리형 evaluator(`BenchMarkEvaluator`/`CustomScorerEvaluator`/`LLMAsJudgeEvaluator`)는 **SageMaker Public Hub에 평가 레시피가 등록된 모델(Amazon Nova·일부 JumpStart)** 전용입니다.
    gemma-4 커스텀 파인튜닝 산출물(S3 체크포인트)에는 쓸 수 없습니다. 실측하면 `DescribeHubContent ... does not exist`로 떨어집니다.

### 멀티모달 코스 (05) 파이프라인

멀티모달 코스는 **노트북 세트가 다르고 더 짧습니다**(이미지 입력, 합성·agentic 단계 없음).

```
00_setup ──▶ 01_data_explore ──▶ 02_train_mm_sagemaker ──▶ 03_deploy_mm_endpoint ──▶ 99_cleanup
            (cord-v2 이미지+JSON)  (vision 동결+language LoRA)  (멀티모달 endpoint)
```

- **시드**: `naver-clova-ix/cord-v2`(cc-by-4.0, ungated). 합성 데이터 단계가 없어 이미지+JSON 시드를 그대로 씁니다.
- **학습**: `scripts/train_mm.py`(`AutoModelForImageTextToText` + `AutoProcessor`).
    vision/audio 파라미터를 동결하고 language 쪽만 [LoRA](https://huggingface.co/docs/peft/index)로 학습합니다.
- **서빙**: 이미지 입력을 받는 멀티모달 endpoint입니다. 텍스트 전용 re-export를 하지 않습니다.

---

## 사전 준비 체크리스트

한 번만 하면 되는 준비입니다.

- [ ] **설치 완료**: `uv venv --python 3.12` → `uv pip install -r pyproject.toml`(자세한 절차는 [시작하기](getting_started.md)).
    코어 의존성은 `>=` floor로만 고정되어 있습니다(`sagemaker>=3.16.0`, `transformers>=5.14.1`, `trl>=1.8.0`, `peft>=0.19.1`). 현행 값은 `pyproject.toml`이 원본입니다.
    SDK v3는 클래스 이름이 v2와 다르므로, 노트북 코드를 손볼 때는 [SageMaker Python SDK 저장소](https://github.com/aws/sagemaker-python-sdk)의 현행 API를 기준으로 보세요.
- [ ] **AWS 자격증명**: `aws sts get-caller-identity`가 계정을 반환하는지 확인합니다.
- [ ] **SageMaker AI 실행 role**: `SAGEMAKER_ROLE_ARN`에 SageMaker AI·S3·ECR 권한이 있어야 합니다.
    Studio/노트북 인스턴스에서는 `config.resolve_sagemaker_role()`이 `get_execution_role()`로 자동 획득하고, IAM user로 로컬 실행하면 IAM에서 실행 role을 자동 탐지합니다.
    **role이 잡히는 것과 그 role에 필요한 권한이 붙어 있는 것은 다릅니다.** 첫 완주 전에 S3·ECR 권한을 한 번 열어 확인하세요. 권한 부족은 제출 시점이 아니라 Job이 뜬 뒤에 드러납니다([실행 role이 매개하는 것](01_sagemaker_basics.md#실행-role로-무엇을-하는가--s3와-ecr-접근)).
- [ ] **Bedrock 모델 액세스**: 콘솔에서 사용할 Claude 모델의 액세스를 활성화합니다.
    그리고 AWS가 [cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)로 문서화하는 **inference-profile ID**(`us.`/`eu.`/`apac.`/`global.` 접두사 필수)를 확보해 `BEDROCK_CLAUDE_MODEL_ID`에 설정합니다. 기본값은 `global.anthropic.claude-sonnet-5`입니다.
- [ ] **모델 선택**: 기본은 `google/gemma-4-E4B-it`(apache-2.0·ungated, `MODEL_SIZE=E4B`)입니다.
    `MODEL_SIZE`로 `E2B`/`12B`/`26B-A4B`/`31B`를, `MODEL_ID`로 임의 모델을 지정할 수 있습니다.
- [ ] **(gated 모델을 쓸 때만) HF 토큰**: gemma-4 전 사이즈는 ungated라 토큰이 필요 없습니다.
    gemma-3 계열 등을 쓸 때만 HF 약관을 수락하고 `MODEL_IS_GATED=1` + 토큰을 설정하세요.
    토큰은 `hf auth login`으로 저장해 두면 config가 파일에서 읽습니다(커스텀 캐시를 쓰면 `HF_HOME`도 같이 맞춰야 합니다).
- [ ] **리전 정합성**: SageMaker AI·Bedrock·S3가 같은 리전(`AWS_REGION`, 기본 `us-west-2`)을 쓰는지 확인합니다.
    리전을 옮기면 `.env`의 DLC 이미지 URI 리전도 함께 바꿔야 합니다(학습 이미지는 리전별 private ECR에서만 pull됩니다).
- [ ] **비용 인지**: real-time endpoint는 삭제 전까지 시간당 과금되므로, 실습이 끝나면 `99_cleanup`을 반드시 실행합니다.

```bash
export AWS_REGION=us-west-2                                  # .env의 DLC URI 리전과 일치해야 합니다
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5   # 콘솔에서 현행 ID 재확인
# export MODEL_IS_GATED=1 && export HF_TOKEN=hf_...          # gated 모델을 쓸 때만
export DRY_RUN=1             # 첫 완주는 1로 (저비용 파이프라인 검증)
```

VS Code로 이 리포 폴더를 워크스페이스로 열면 `.env`가 커널 env로 자동 주입됩니다(인스턴스 타입·DLC 이미지 URI·리전·합성 건수 등 설정값). 시크릿은 `.env`에 넣지 말고 셸 export나 `hf auth login`을 쓰세요.

!!! tip "LiteLLM이 필요하면 별도 환경에"
    agentic 단계는 Bedrock을 boto3로 직접 호출하므로 **LiteLLM 없이 완결됩니다**. `common/llm_gateway.py`(LiteLLM 경유)가 필요하면 별도 환경에 설치하세요(`uv pip install -e '.[litellm]'`, 현재 하한 `litellm>=1.75.9`).
    코어에서 뺀 이유는 litellm이 요구하는 `importlib-metadata>=8`이 sagemaker의 `<7`과 하드 충돌하기 때문입니다.

---

## 단계별 실행과 데이터 핸드오프

`jupyter lab`을 띄우고 코스 폴더(`tracks/<track>/`)에서 번호 순서대로 실행합니다. 아래는 각 단계가 **무엇을 하고 / 무엇을 넘기고 / 무엇을 확인**하는지입니다.

| # | 노트북 | 하는 일 | 다음으로 넘기는 것 | 완료 확인 |
|---|---|---|---|---|
| ① | `00_setup` | 설치·자격증명·role/bucket 해석 | `%store`: `role`, `bucket` | account id 출력, role/bucket 정상 |
| ② | `01_data_and_synthetic` | 시드 로드 + grounded 합성 + EDA | 코스 로컬 파일 `data/train.jsonl` | JSONL 생성, 포맷·토큰 길이 미리보기 정상 |
| ③ | `02_train_sft_sagemaker` | (선택 로컬 dry-run →) [TRL `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) 기반 SageMaker AI 학습 Job | `%store`: `model_data`, `md_<track_key>` | Job `Completed`, CloudWatch 링크 |
| ③-a | (선택) `02a_train_grpo_sagemaker` | SFT→GRPO 정련 — **추출·분류 코스만** | `model_data` 갱신 | Job `Completed` |
| ③-b | (선택) `02b_local_serve` | 배포 전 로컬 vLLM 프리플라이트 | (없음) | 로컬 invoke 응답 정상 |
| ④ | `03_deploy_endpoint` | [real-time endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) 배포 + invoke 스모크 | `%store`: `endpoint_name`, `ep_<track_key>` | `InService` 도달, invoke 응답 정상 |
| ⑤ | `04_evaluate` | held-out 세트로 성공기준 수치화 | (없음) | 지표 출력(`arg_f1`/`macro_f1`/ROUGE-L/judge) |
| ⑥ | `05_agentic_strands` | SLM(tool) + Bedrock Claude 루프 ([Strands](https://github.com/strands-agents/sdk-python)) | (없음) | 에이전트 응답 정상 |
| ⑦ | `06_agentcore_deploy` | [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html) 배포(프로덕션) | (없음) | (선택) Runtime 호출 성공 |
| ⑧ | `99_cleanup` | endpoint·config·model·Runtime 삭제 | (없음) | 이 코스 endpoint 목록이 비어 있음 |

학습 Job과 endpoint 생성은 **SageMaker AI 서버에서 진행되므로 커널이나 세션이 끊겨도 계속됩니다.**
각 노트북에 재접속 셀이 있어 Job 이름(`TrainingJob.get(name)`)이나 endpoint 이름(`Endpoint.get(name)`)으로 다시 붙을 수 있습니다.

### 단계별 주의

- **② 합성**: `NUM_SYNTHETIC`가 Bedrock 호출량, 즉 비용을 좌우합니다. config 기본값은 200이고, 이 리포의 `.env`는 요약 코스 지연 때문에 100으로 낮춰 두었습니다.
    합성 전에 토큰 길이 EDA를 꼭 보세요. 학습이 자르는 단위는 문자가 아니라 토큰이고, 한국어·JSON은 문자당 토큰 수가 영어의 몇 배입니다.
- **③ 학습**: `stopping_condition`을 **반드시 명시**하세요. 생략하면 SDK 기본 1시간이 붙습니다.
    [StoppingCondition API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)가 적는 API 기본값 1일과는 다릅니다.
    학습 이미지는 `.env`의 `DLC_IMAGE_URI`(완전 URI)를 `common/dlc.py`가 그대로 씁니다. env가 없으면 `DLC_REPOSITORY`+`DLC_TAG` 조립 → 라이브러리 버전 조합 순으로 폴백합니다.
    태그는 자주 갱신되므로 실행 직전에 [DLC available images](https://aws.github.io/deep-learning-containers/reference/available_images/)에서 현행 태그를 확인하세요.
    첫 실행은 용량 대기(Pending)와 이미지 pull(Downloading) 때문에 시작이 느립니다(실측 각 6분·3분).
- **④ 배포**: 기본 경로는 **vLLM DLC(`SERVING_ENGINE=vllm`)** 이고, `sglang`(같은 셀에서 처리) 또는 `lmi`(`OPTION_*` env)로 전환할 수 있습니다.
    셋 다 연속 배칭 + OpenAI 호환 `messages` 스키마라 호출 코드가 동일합니다.
    **한 번에 하나만** 배포하세요. 둘을 띄우면 endpoint가 두 개가 되어 과금이 중복됩니다. endpoint 기동에는 5~15분이 걸립니다.
    참고할 곳: 엔진 선택 기준은 [서빙 엔진 선택 — SERVING_ENGINE](05_serving_containers.md#서빙-엔진-선택--serving_engine), 메모리 예산은 [메모리 예산 — L4 22.9GB 실측](05_serving_containers.md#메모리-예산--l4-229gb-실측), 호출 스키마는 [SageMaker AI 추론](04_sagemaker_inference.md#invoke_endpoint-호출-스키마).
- **⑤ 평가**: held-out은 학습에 쓴 앞 구간(`NUM_SEED_SAMPLES`, 기본 300건)을 **명시적으로 건너뛴 뒤** 잘라 씁니다.
    `pool[-N:]` 방식은 위험합니다(예: `N_EVAL=50`이면 150건만 로드되어 held-out이 학습 구간 안쪽에 통째로 들어갑니다).
- **⑥/⑦ agentic**: endpoint와 Bedrock이 **이중으로 과금**됩니다.
    endpoint는 `sagemaker-runtime`, Bedrock은 `bedrock-runtime`의 [`converse`](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)로 SageMaker AI와 **별개 서비스**입니다.
    AgentCore는 GA 상태와 리전을 재확인하세요([프로덕션 배포](06_agentic.md#프로덕션-배포--agentcore-runtime)).
- **⑧ 정리**: 중간에 멈추더라도 endpoint가 떠 있으면 `99_cleanup`이 먼저입니다. 그러지 않으면 계속 과금됩니다.

??? info "②에서 `NUM_SYNTHETIC`를 100으로 낮춘 근거"
    요약 시드는 1건 중앙 1,651자로, 추출 코스 475자보다 훨씬 깁니다 → 배치 프롬프트가 약 10,900자까지 늘어납니다.
    출력은 2,554토큰으로 `max_tokens` 4,500 안이라 절단은 없습니다. 즉 문제는 품질이 아니라 순수 지연입니다.

!!! danger "평가셋은 합성으로 만들지 마세요"
    합성 데이터나 학습셋으로 평가하면 점수가 조용히 부풀려집니다. teacher 모델을 얼마나 모방했는지를 재는 데 그칩니다.
    `04_evaluate`는 반드시 **합성 증강 이전의 시드에서 결정론적으로 분리한 held-out**만 사용합니다. 규율의 배경은 [held-out 규율](02_synthetic_data.md#held-out-규율--합성으로-평가-금지)에 있습니다.

위 단계들의 근거를 원본 소스에서 확인하려면 다음을 보세요.

??? info "더 읽을 거리"
    - [aws/deep-learning-containers](https://github.com/aws/deep-learning-containers): 학습·서빙 DLC의 Dockerfile과 `sagemaker_entrypoint.sh`(env → CLI 플래그 변환 규칙)를 소스에서 확인할 때.
    - [SageMaker 모델 배포 옵션 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html): 이 runbook이 ④에서 real-time을 고르는 배경(4옵션 정의와 과금 모델).
    - [InvokeEndpoint API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html): ④ 이후 호출에 걸리는 파라미터·payload 한도의 원문.
    - ④에서 고르는 세 엔진의 저장소·문서: [vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang) · [DJL LMI](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html). 지원 모델·플래그는 문서보다 저장소가 빠릅니다.

---

## 5개 코스를 모두 돌리려면

코스는 **독립**입니다. 한 코스를 완주하고 정리한 뒤, 다른 코스 폴더에서 같은 순서를 반복하면 됩니다.

```
tracks/01_extraction_to_json/    (텍스트→JSON 추출)   ← 플래그십, 여기부터
tracks/02_classification/        (intent 분류)
tracks/03_summarization/         (문서 요약)
tracks/04_domain_qa/             (도메인 QA)
tracks/05_multimodal_extraction/ (이미지→JSON 추출, 영수증·gemma-4 vision)  ← 별도 구조
```

- 텍스트 코스(01~04)는 위 [텍스트 코스 파이프라인](#텍스트-코스-0104-파이프라인)의 `00 → 99` 순서를 따릅니다.
- 멀티모달 코스(05)는 [멀티모달 코스 파이프라인](#멀티모달-코스-05-파이프라인)의 5단계 세트를 씁니다.
- 코스마다 **별도 endpoint**가 뜹니다 → 각 코스의 `99_cleanup`을 각각 실행하세요.
- 공통 로직은 `common/`이 공유하므로, 텍스트 코스 간 차이는 데이터 어댑터(`tracks/*/track_data.py`)와 `config.TRACKS` 레지스트리뿐입니다.
- 여러 코스를 동시에 띄우면 GPU 인스턴스 비용이 코스 수만큼 늘어납니다. **한 코스씩 완주하고 정리하는 방식**을 권장합니다.

??? question "오개념 — “코스를 옮기면 `%store` 값도 알아서 바뀌겠지?”"
    **그렇지 않습니다.** `%store`는 IPython 프로필 단위라 **코스를 넘어 공유**됩니다. 전역 `endpoint_name`/`model_data`는 마지막에 실행한 코스 값이 남아, 엉뚱한 endpoint를 호출하거나 다른 코스 모델을 배포하게 됩니다.
    그래서 노트북은 코스 전용 키(`ep_<track_key>`, `md_<track_key>`)를 먼저 읽고, `train_path`는 아예 코스 로컬 파일(`data/train.jsonl`)로 고정합니다.

---

## E2E 흐름에서 자주 막히는 곳

| 증상 | 원인과 해결 |
|---|---|
| `02`에서 `train_path`/`data/train.jsonl` 없음 | `01`을 실행하지 않았습니다. 코스 내 노트북은 **순서대로** 실행하세요 |
| 다른 코스 endpoint를 호출하거나 옛 모델이 배포됨 | `%store` 전역 키 오염 → 코스 전용 키(`ep_<track_key>`, `md_<track_key>`)를 쓰고, 리전을 바꿨다면 `aws_utils.ensure_model_data_in_region()`이 옛 리전 아티팩트를 걸러 줍니다 |
| 학습 Job이 시작 직후 실패 | IAM role 권한(S3/ECR) 또는 DLC 태그 문제 → CloudWatch 로그 확인, `.env`의 `DLC_IMAGE_URI` 리전·태그 재확인(`aws ecr describe-images`). 권한이 제출 시점에 안 걸리고 여기서 터지는 구조는 [실행 role이 매개하는 것](01_sagemaker_basics.md#실행-role로-무엇을-하는가--s3와-ecr-접근) |
| 학습이 끝났는데 Job이 `Stopped`, 아티팩트에 머지 모델이 없음 | `stopping_condition` 생략 시 붙는 SDK 기본 1시간(`MaxRuntimeExceeded`)에 머지 단계가 잘렸습니다 → `MAX_RUNTIME_HOURS`를 명시(기본 4시간). 실측에서는 Pending 6분 + Downloading 3분 + Training 55분(189 step 전부 완료) 후 머지 도중 종료됐고, `FailureReason`은 비어 있습니다. 상세는 [파인튜닝](03_finetuning.md) |
| `InsufficientInstanceCapacity`로 Job이 안 뜸 | 리전별 GPU 용량 문제 → `AWS_REGION`을 바꿔 재시도(`.env`의 DLC URI 리전도 함께 변경) |
| endpoint가 `Failed`, 이유는 `did not pass the ping health check`뿐 | 대개 CUDA OOM입니다. 24GB GPU(L4)에서 vLLM 기본 `max_num_seqs=256`이 샘플러 logits 버퍼를 `256 × 262,144 × 4B = 256 MiB`로 잡아 터집니다 → `serving_env()` 기본값(`max_num_seqs=32`, `gpu_memory_utilization=0.90`)을 유지하고 CloudWatch endpoint 로그를 확인하세요([24GB GPU CUDA OOM](04_sagemaker_inference.md#24gb-gpu-cuda-oom--max_num_seqs-기본값)) |
| gated 모델 다운로드 401 | HF 약관 미수락 또는 토큰 없음 → `MODEL_IS_GATED=1` + `HF_TOKEN`(또는 `hf auth login`), 아니면 ungated `google/gemma-4-E4B-it`을 쓰세요 |
| Bedrock `converse` 400 | 모델 ID가 base(접두사 없음)이거나 액세스 미승인 → inference-profile ID(`us.`/`global.` 등) 사용 + 콘솔에서 액세스 활성화 |
| endpoint invoke 응답이 반복되거나 저품질 | raw 텍스트를 보내 chat template이 빠졌습니다 → `messages` 스키마로 보내 **서버가 template을 적용**하게 하세요(`aws_utils.invoke_sagemaker_chat`) |
| 응답이 중간에 끊김 | `max_tokens` 부족 → `finish_reason`을 확인하세요([max_tokens 절단과 finish_reason](05_serving_containers.md#max_tokens-절단과-finish_reason)) |
| speculative decoding이 켜지지 않음 | 이 kit 노트북에는 배선돼 있지 않습니다. 컨테이너 설정 키는 vLLM DLC `SM_VLLM_SPECULATIVE_CONFIG` / LMI `OPTION_SPECULATIVE_CONFIG`이며, **Gemma용 P-EAGLE head는 AWS가 공개하지 않았고** 커뮤니티 EAGLE3 head는 fine-tuned target과의 정합성을 직접 실측해야 합니다([Speculative decoding (EAGLE3 / P-EAGLE)](05_serving_containers.md#speculative-decoding-eagle3--p-eagle)) |
| `litellm` import 오류 | 코어에 미포함(sagemaker와 `importlib-metadata` 충돌) → 별도 환경에 설치 |
| 비용이 계속 나감 | endpoint 또는 AgentCore Runtime 미삭제 → `99_cleanup` 실행 + 콘솔에서 확인 |

---

## 완료 기준 — Definition of Done

한 코스 E2E가 "됐다"고 말할 수 있는 조건입니다.

- [ ] `02` 학습 Job이 `Completed`이고, 아티팩트 루트에 **머지된 서빙용 모델**이 있음(어댑터만 있으면 배포되지 않습니다)
- [ ] `03` invoke 스모크가 의미 있는 출력을 반환
- [ ] `04_evaluate` 지표가 나옴 (가능하면 파인튜닝 전 baseline과 비교해 개선 폭 확인)
- [ ] (선택) `05_agentic_strands`에서 Claude가 SLM endpoint를 tool로 호출하는 왕복이 성공
- [ ] (프로덕션 목표 시) `06_agentcore_deploy`로 Runtime 배포 확인
- [ ] `99_cleanup` 실행 → 이 코스 prefix의 endpoint 목록이 비어 있고, 콘솔에서도 0개

---

## 비용과 cleanup

!!! danger "비용과 cleanup"
    **real-time endpoint는 삭제하기 전까지 시간당(GPU 인스턴스) 요금이 계속 부과됩니다.** 호출이 0건이어도 켜져 있는 동안 과금됩니다.
    실습이 끝나면 **모든 코스의 `99_cleanup`을 실행하고 콘솔에서 endpoint 0개를 확인**하세요.
    여러 번 배포했다면 `%store`의 `endpoint_name`은 마지막 것만 가리킵니다. 코스 prefix(`gemma-extraction` 등)로 잔여 리소스를 훑어 정리하고, 다른 리전에도 띄운 적이 있으면 그 리전도 확인하세요.

삭제 순서는 **endpoint → endpoint-config → model**입니다. 앞을 지우지 않으면 뒤가 사용 중이라 거부됩니다.

model 이름은 `ModelBuilder`가 `model-42c30d1e`처럼 자동 생성하므로 `endpoint_name`으로는 찾을 수 없습니다. `99_cleanup`은 endpoint-config에서 실제 `ModelName`을 먼저 조회해 지웁니다.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker AI real-time endpoint | 인스턴스 시간당, 삭제 전까지 계속 | `99_cleanup` → `delete_endpoint` → `delete_endpoint_config` → `delete_model` |
| SageMaker AI 학습 Job | Job 실행 시간만(종료 시 과금 중단) | 자동 종료. Managed Spot 미사용 시 on-demand 요금 |
| Bedrock Converse (합성·agentic·judge) | 호출 토큰량 기준, 상주 리소스 없음 | teardown 불필요. 합성 건수·judge 샘플 수로 조절 |
| AgentCore Runtime | 배포한 경우 Runtime 리소스 과금 | `bash agentcore/cleanup_agent.sh --aws`(Runtime + ECR) |
| 로컬 `local_model/`·vLLM 프로세스 | 과금 없음(디스크 약 15GB·GPU 점유) | `bash scripts/cleanup_local.sh --yes` |

각 노트북은 학습·배포 직후 **CloudWatch 다이렉트 링크**를 출력합니다(`common/aws_utils.cw_links()`). Job 로그, endpoint 기동, OOM, Bedrock 호출량을 여기서 실시간으로 볼 수 있습니다.

??? question "오개념 — “endpoint를 호출하지 않으면 요금도 안 나오죠?”"
    **아닙니다.** real-time endpoint는 호출 여부와 무관하게 **provisioned 인스턴스가 시간당** 과금됩니다. 오토스케일도 통상 최소 1대는 유지합니다.
    쓰지 않는다면 삭제가 정답입니다. 비용 관점의 전체 비교는 [비용과 cleanup](04_sagemaker_inference.md#비용과-cleanup)에 있습니다.

---

## 관련 리포지토리 파일

설정과 공통 유틸:

- `common/config.py`: 전역 설정 로더. `MODEL_SIZE` 프리셋, `SERVING_ENGINE`, `is_dry_run()`, `TRACKS` 레지스트리
- `common/dlc.py`: DLC 이미지 URI 해석(`DLC_IMAGE_URI` → `DLC_REPOSITORY`+`DLC_TAG` → 버전 조합 폴백)과 서빙 env 생성(`serving_env`)
- `common/aws_utils.py`: endpoint 호출(`invoke_sagemaker_chat`), CloudWatch 링크(`cw_links`),
    리전 정합성 검사(`ensure_model_data_in_region`), 비용 경고(`COST_WARNING`)
- `.env`: 인스턴스 타입·DLC 이미지 URI·리전·합성 건수 등 비시크릿 설정값

학습 스크립트(코스 폴더에 자족적으로 들어 있음):

- `tracks/*/scripts/train.py`: SFT + LoRA/QLoRA 학습, 머지 후 텍스트 re-export
- `tracks/*/scripts/train_grpo.py`: SFT 산출물을 reward 함수로 정련하는 GRPO 학습(추출·분류 코스만)
- `tracks/05_multimodal_extraction/scripts/train_mm.py`: 멀티모달 SFT(`AutoProcessor` + vision 동결, 텍스트 re-export 없음)

평가와 정리:

- `common/eval_utils.py`: 코스별 지표(`arg_f1`/`macro_f1`/ROUGE-L/LLM-judge)
- `tracks/*/99_cleanup.ipynb`: endpoint → endpoint-config → model 삭제
- `agentcore/cleanup_agent.sh`: AgentCore Runtime + ECR 정리(`--aws`)
