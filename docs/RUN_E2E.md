# E2E 실행 런북 (Run End-to-End)

파이프라인을 **처음부터 끝까지 한 번에** 돌리는 사람을 위한 런북.
"어떤 순서로, 무엇을 준비하고, 각 단계가 무엇을 다음으로 넘기고, 얼마 드는지, 무엇을 확인하고 넘어가는지"를 한 곳에.

> 설치·개별 방식(스모크/dry-run)은 [`../GETTING_STARTED.md`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/GETTING_STARTED.md) 참고. 이 문서는 **클라우드 E2E 전체 흐름** 전용.
> 개념 배경은 [`00_overview.md`](00_overview.md) 및 각 주제 가이드.

---

## 0. 파이프라인 한눈에

```
00_setup ──▶ 01_data_and_synthetic ──▶ 02_train_sft_sagemaker ──▶ 03_deploy_endpoint
  (role,        (train_path)               (model_data)            (endpoint_name)
   bucket)                                       │                        │
                                        (선택) 02a_train_grpo             ▼
                                        (선택) 02b_local_serve
                        04_evaluate ──▶ 05_agentic_strands ──▶ 06_agentcore_deploy
                        (held-out 점수)   (SLM+Claude 루프)       (프로덕션 배포)
                        (선택) 04b_sagemaker_eval (managed 평가·🔴 별도 비용)
                                                                        │
                                                                        ▼
                                                                  99_cleanup 🔴
```
각 노트북은 결과를 `%store`로 저장하고 다음 노트북이 `%store -r`로 읽습니다 (아래 "데이터 핸드오프" 표).
**한 트랙**(예 `tracks/01_extraction_to_json/`)을 00→99 순서로 실행하면 1개 E2E가 완성됩니다.
- 위 다이어그램은 **텍스트 트랙(01~04)** 기준입니다. **(선택) `02a_train_grpo_sagemaker`**(SFT→GRPO 정련)는 추출·분류 트랙에만, **(선택) `02b_local_serve`**(배포 전 로컬 vLLM 검증)는 모든 텍스트 트랙에서 제공됩니다. 평가는 **(선택) `04b_sagemaker_eval`**(SageMaker managed 평가 잡, 🔴 별도 컴퓨트·비용)을 `04_evaluate`(로컬·빠름·저렴)의 대안으로 모든 텍스트 트랙에서 선택할 수 있습니다.
- **멀티모달 트랙(05)은 별도의 더 짧은 파이프라인**입니다 (이미지 입력, 합성/agentic 단계 없음 — 아래 4번 참조):
  ```
  00_setup ──▶ 01_data_explore ──▶ 02_train_mm_sagemaker ──▶ 03_deploy_mm_endpoint ──▶ 99_cleanup 🔴
              (cord-v2 이미지+JSON)  (vision 동결+lang LoRA)   (멀티모달 endpoint)
  ```

---

## 1. 사전 준비 체크리스트 (한 번)

- [ ] **설치 완료** — `uv venv && uv pip install -r pyproject.toml` (자세한 절차는 [GETTING_STARTED 1번](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/GETTING_STARTED.md)을 참고하세요). 코어 의존성은 sagemaker 3.x·transformers 5.x 등 실측 최신 floor로 고정되어 있습니다.
- [ ] **AWS 자격증명** — `aws sts get-caller-identity`가 계정을 반환하는지 확인합니다.
- [ ] **SageMaker 실행 역할** — `SAGEMAKER_ROLE_ARN`에 SageMaker·S3·ECR 권한이 있어야 합니다. Studio에서는 `get_execution_role()`로 자동 획득됩니다.
- [ ] **Bedrock 모델 액세스** — 콘솔에서 사용할 Claude 모델의 액세스를 활성화하고, 정확한 inference-profile ID를 확보해 `BEDROCK_CLAUDE_MODEL_ID`에 설정합니다.
- [ ] **(gated 모델인 경우) HF 토큰** — gemma-3 계열은 HF 약관 수락 후 `HF_TOKEN`이 필요하고, gemma-4(ungated)는 필요하지 않습니다. 매번 붙여넣기 싫다면 `~/.bashrc`에 `export HF_TOKEN=hf_xxx`를 한 번만 넣어 두세요(자세한 내용은 [GETTING_STARTED](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/GETTING_STARTED.md) 4-3).
- [ ] **리전 정합성** — SageMaker·Bedrock·S3가 같은 리전(`AWS_REGION`)을 쓰는지 확인합니다.
- [ ] **비용 인지** — real-time endpoint는 삭제 전까지 시간당 과금되므로, 실습이 끝나면 `99_cleanup`을 반드시 실행합니다.

> 💡 **LiteLLM은 코어에 포함되지 않습니다.** agentic 단계(⑥)는 Bedrock을 native로 호출하므로 LiteLLM 없이 완결됩니다. LiteLLM 게이트웨이가 필요하면 별도 환경에 `pip install 'litellm>=1.93.0'`으로 설치하세요(sagemaker와 importlib-metadata 의존성이 충돌하기 때문입니다).

```bash
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=us.anthropic.claude-...   # 🔴 실제 ID 재확인
# export HF_TOKEN=hf_...     # gated 모델만
export DRY_RUN=1             # 첫 완주는 1로! (저비용 파이프라인 검증)
```

---

## 2. 권장 실행 전략 — "두 번 완주"

E2E를 두 번 돕니다. **재작업·비용 폭탄을 막는 정석**입니다.

1. **1차 완주 (`DRY_RUN=1`)**: 00→06 을 소량·저비용으로 끝까지. 자격증명·권한·파이프라인·핸드오프가 다 맞는지 확인. endpoint도 작게 떠보고 `99_cleanup`.
2. **2차 완주 (`DRY_RUN=0`)**: 합성 건수↑·학습 epoch 정상·실제 평가. 이때만 본격 과금.

> `DRY_RUN` 은 `common/config.is_dry_run()` 이 읽습니다. 노트북이 이 값으로 샘플 수·epoch·합성 건수를 자동 축소합니다.

---

## 3. 단계별 실행 (한 트랙 기준)

`jupyter lab` 실행 후 `tracks/<트랙>/` 에서 번호 순서대로. 아래는 각 단계가 **무엇을 하고 / 무엇을 넘기고 / 무엇을 확인**하는지.

| # | 노트북 | 하는 일 | 다음으로 넘기는 것(`%store`) | 완료 확인 |
|---|---|---|---|---|
| ① | `00_setup` | 설치·자격증명·역할/버킷 | `role`, `bucket` | account id 출력, role/bucket 정상 |
| ② | `01_data_and_synthetic` | 시드 로드 + grounded 합성 → 학습 JSONL | `train_path` | `train.jsonl` 생성, 포맷 미리보기 정상 |
| ③ | `02_train_sft_sagemaker` | (로컬 dry-run 셀 →) SageMaker 학습 잡 | `model_data` | 잡 `Completed`, CloudWatch 링크 |
| ③-a | (선택) `02a_train_grpo_sagemaker` | SFT→GRPO 정련(RLHF) — **추출·분류 트랙만** | `model_data`(갱신) | 잡 `Completed` |
| ③-b | (선택) `02b_local_serve` | 배포 전 로컬 vLLM 프리플라이트 | — | 로컬 invoke 응답 정상 |
| ④ | `03_deploy_endpoint` | real-time endpoint 배포(A: DJL LMI 기본 / B: vLLM 옵션) + invoke 스모크 | `endpoint_name` | invoke 응답 정상 |
| ⑤ | `04_evaluate` | held-out 세트로 성공기준 수치화 (로컬·빠름·저렴) | — | 지표 출력(arg_f1/F1/ROUGE/judge) |
| ⑤-b | (선택) `04b_sagemaker_eval` | SageMaker managed 평가 잡 (`BenchMark`/`CustomScorer`/`LLMAsJudge` 조합) — **텍스트 트랙만**, 🔴 별도 컴퓨트·비용 | — | 평가 잡 `Completed`, 리포트 출력 |
| ⑥ | `05_agentic_strands` | SLM(tool) + Bedrock Claude 루프 | — | 에이전트 응답 정상 |
| ⑦ | `06_agentcore_deploy` | AgentCore Runtime 배포(프로덕션) | — | (선택) Runtime 호출 성공 |
| 🔴 | `99_cleanup` | endpoint·config·Runtime 삭제 | — | `남은 endpoint: 없음 ✅` |

### 단계별 주의
- **② 합성**: `NUM_SYNTHETIC`(기본 500/트랙)이 Bedrock 호출량=비용을 좌우. dry-run은 자동 축소. 🔴 **평가셋은 합성으로 만들지 말 것**(04는 시드 held-out 사용).
- **③ 학습**: DLC 이미지는 `common/dlc.py`가 env(`DLC_IMAGE_URI`/`DLC_TAG`)로 최신 태그를 반영하고, env가 없으면 라이브러리 버전으로 폴백합니다. 첫 실행은 이미지 pull 때문에 시작이 다소 느릴 수 있습니다.
- **④ 배포**: 기본은 **A. DJL LMI(vLLM 백엔드)** 이고, 최신 vLLM 기능이 필요하면 **B. 단독 vLLM**(EAGLE3 speculative decoding 포함) 경로를 선택할 수 있습니다. **A와 B 중 하나만** 실행하세요(둘 다 띄우면 endpoint가 두 개가 되어 과금이 중복됩니다). 컨테이너 선택 기준과 speculative decoding(§5.5)은 [`05_serving_containers.md`](05_serving_containers.md)를 참고하세요. endpoint가 기동되기까지 수 분이 걸립니다.
- **⑥/⑦ agentic**: endpoint(`sagemaker-runtime`) + Bedrock(`bedrock-runtime`) **이중 과금**. AgentCore는 GA/리전 재확인.
- **🔴 정리**: 중간에 멈추더라도 **endpoint가 떠 있으면 `99_cleanup` 먼저**. 안 그러면 계속 과금.

---

## 4. 5개 트랙을 모두 돌리려면

트랙은 **독립**입니다. 한 트랙을 완주한 뒤, 다른 트랙 폴더에서 동일 순서로 반복하면 됩니다.
```
tracks/01_extraction_to_json/   (텍스트→JSON 추출)   ← 플래그십, 여기부터
tracks/02_classification/       (intent 분류)
tracks/03_summarization/        (문서 요약)
tracks/04_domain_qa/            (도메인 QA)
tracks/05_multimodal_extraction/ (이미지→JSON 추출, 영수증·gemma-4 vision)  ← 별도 구조
```
- 텍스트 트랙(01~04)은 위 0/3번의 `00→99` 순서를 따릅니다.
- **멀티모달 트랙(05)은 노트북 세트가 다릅니다**: `00_setup → 01_data_explore → 02_train_mm_sagemaker → 03_deploy_mm_endpoint → 99_cleanup`. 시드는 `naver-clova-ix/cord-v2`(cc-by-4.0, ungated)이고, 합성 데이터 단계가 없으며(이미지+JSON 시드를 바로 사용), 학습은 `scripts/train_mm.py`(AutoModelForImageTextToText + processor, vision tower 동결 + language LoRA), 서빙은 이미지 입력을 받는 멀티모달 endpoint입니다(텍스트 전용 재-export 아님). agentic/agentcore 단계는 없습니다.
- 트랙마다 **별도 endpoint**가 뜹니다 → 각 트랙 `99_cleanup`을 각각 실행.
- 공통 로직은 `common/`이 공유하므로, 텍스트 트랙 간 차이는 데이터 어댑터(`track_data.py`)와 config뿐.
- 여러 트랙을 동시에 띄우면 GPU 인스턴스 비용이 트랙 수만큼 늘어나므로, 한 트랙씩 완주하고 정리하는 방식을 권장합니다.

---

## 5. 비용 가드 (E2E 실행자 필독)

| 리소스 | 과금 방식 | 멈추는 법 |
|---|---|---|
| SageMaker **학습 잡** | 잡 실행 시간(인스턴스) | 잡 끝나면 자동 중단. Managed Spot으로 절감 가능(주석 참고) |
| SageMaker **endpoint** | 🔴 **삭제 전까지 시간당 상시** | `99_cleanup` (`delete_endpoint`) |
| **Bedrock** Converse (합성·agentic·judge) | 토큰 호출량 | 상시 리소스 없음. 합성 건수·judge 샘플 수로 조절 |
| **AgentCore** Runtime | 배포 시 리소스 | Runtime 삭제 |

- 첫 완주는 `DRY_RUN=1`로 → 소량·소형으로 비용 최소화.
- 실습 종료 시 **모든 트랙의 `99_cleanup` 실행 + 콘솔에서 endpoint 0개 확인**.
- CloudWatch 링크(각 노트북이 출력)로 잡·endpoint·Bedrock 호출량 실시간 확인.

---

## 6. 문제 해결 (E2E 흐름에서 자주 막히는 곳)

| 증상 | 원인/해결 |
|---|---|
| `02`에서 `%store -r train_path` 없음 | `01`을 먼저 실행 안 함. 트랙 내 노트북은 **순서대로** |
| 학습 잡이 시작 직후 실패 | IAM 역할 권한(S3/ECR) 또는 DLC 태그 문제 → CloudWatch 로그 확인, `DLC_TAG` 재확인 |
| gated 모델 다운로드 401 | HF 약관 미수락/토큰 없음 → `HF_TOKEN` 설정 또는 ungated `gemma-4-12B-it` 사용 |
| Bedrock `converse` 400 | 모델 ID가 base(prefix 없음)이거나 액세스 미승인 → inference-profile ID(us./eu./…) 사용 + 콘솔 액세스 |
| endpoint invoke 응답 이상 | 서빙 컨테이너 스키마 불일치 → `03` invoke 스모크의 chat/generation 스키마 확인([`05_serving_containers.md`](05_serving_containers.md)) |
| vLLM(B) speculative decoding이 안 켜짐 | Gemma용 EAGLE3 draft head가 없거나 미지정 → `EAGLE3_SPECULATOR` 설정, fine-tuned 모델과의 정합성 실측([`05_serving_containers.md`](05_serving_containers.md) §5.5) |
| `litellm` import 오류 | 코어에 미포함(sagemaker와 의존성 충돌) → 필요 시 별도 환경에 `pip install 'litellm>=1.93.0'` |
| 비용이 계속 나감 | endpoint/AgentCore 미삭제 → `99_cleanup` + 콘솔 확인 |

---

## 7. 완료 기준 (Definition of Done)

한 트랙 E2E가 "됐다"고 말할 수 있는 조건:
- [ ] `03` invoke 스모크가 의미 있는 출력을 반환
- [ ] `04_evaluate` 지표가 나옴 (가능하면 파인튜닝 전 baseline과 비교해 개선 확인)
- [ ] (선택) `04b_sagemaker_eval`로 SageMaker managed 평가를 돌렸다면 잡이 `Completed`되고 리포트가 나옴 (🔴 별도 비용)
- [ ] `99_cleanup` 실행 → 콘솔에서 endpoint 0개
- [ ] (프로덕션 목표 시) `06_agentcore_deploy`로 Runtime 배포 확인

> ⚠️ 이 킷의 fast-changing 항목(모델 ID·DLC 태그·SDK 버전·AgentCore GA/리전)은 **실행 직전 재확인** 대상입니다(코드에 `# TODO verify` 표기). 근거·재확인 링크는 각 `docs/` 문서 하단 참조.
