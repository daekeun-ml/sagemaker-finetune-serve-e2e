# 00 · Overview — Gemma E2E Toolkit 전체 지도

> **읽는 사람**: 이 킷을 처음 여는 ML 엔지니어 / 데이터 과학자입니다. SageMaker·Bedrock을 조금 알아도 되고 몰라도 됩니다.
> **이 문서의 역할**: 킷 전체를 한눈에 매핑하는 **진입점(index)**입니다. 개념 설명은 각 상세 문서(01~05)로 연결하고, 여기서는 "무엇이 어디에 있고 어떤 순서로 도는가"만 확정합니다.
> ⚠️ **주의**: 아래 🔴 표시된 값(모델 ID·DLC 이미지 태그·SDK 버전·리전·GA 상태)은 **빠르게 바뀝니다**. 문서의 서술은 조건부이므로, **실행 직전 각 소스 페이지에서 재확인**하시기 바랍니다.
> **라이브 검증 2026-07** (근거 표는 문서 하단).

> 🚀 **바로 시작**: [`../GETTING_STARTED.md`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/GETTING_STARTED.md) (설치→스모크→dry-run→노트북)
> 🔁 **E2E 완주 런북**: [`RUN_E2E.md`](RUN_E2E.md) (단계별 핸드오프·비용·체크리스트)

---

## §0 · TL;DR

**한 줄**: task → 오픈 모델/데이터 → grounded 합성 → SageMaker 학습(HF DLC + TRL) → real-time endpoint(DJL LMI) → agentic(Strands→AgentCore) → eval 까지, **5개 독립 트랙**이 얇은 `common/` 레이어를 공유하는 step-by-step 실습 킷입니다.

1. **텍스트 트랙 파이프라인은 7단계**로 이루어집니다 — 노트북 `00→06` + `99_cleanup`이 그대로 각 단계에 대응합니다(§2). (멀티모달 트랙은 더 짧은 별도 파이프라인 — §2 하단 참조.)
2. **트랙은 5개**(추출→JSON / 분류 / 요약 / 도메인-QA / 멀티모달 추출)이며 서로 독립된 E2E입니다. 공통 로직만 `common/`으로 분리했습니다(§3).
3. **모델**은 `google/gemma-3-4b-it`를 기본으로 사용하며(🔴 gated), `MODEL_ID`만 바꾸면 `google/gemma-4-12B-it`(apache-2.0·ungated)로 승급할 수 있습니다. `gemma-4-31B`는 (요청대로) 제외했습니다(§4).
4. **실행 규율**로는 `uv`로 설치한 뒤 env를 주입하고, **`DRY_RUN=1`로 파이프라인부터 검증**한 다음(GPU dry-run은 L40S에서 확인되었습니다) 실제 실행으로 넘어가시기 바랍니다(§5).
5. **각 문서·노트북 매핑표**는 §6에, **비용/cleanup·라이선스**는 §7에 정리했습니다.

> 📎 **아직 상세 문서를 안 보셨다면**: 배포/추론이 이 킷의 핵심이므로 **[01 inference · CORE]** 부터 읽으시기를 권장합니다. 학습 경로가 궁금하시면 02를, 데이터가 궁금하시면 03을 참고하세요.

---

## §0.5 · 기존 Pain Point (이 킷이 없을 때 겪는 것)

- "Gemma를 SageMaker에서 파인튜닝→서빙"하는 예제는 조각조각 흩어져 있고, **버전·이미지 태그가 스치듯 오래된 경우가 많습니다**. 그대로 복붙하면 컨테이너 resolve에 실패합니다.
- **tier/서비스 혼동**도 흔합니다: "endpoint를 Bedrock으로 부른다", "Serverless로 LLM 띄운다", "학습은 JumpStart로" 같은 오해가 실습을 무너뜨립니다.
- Gemma 특유의 함정(chat template의 system role 거부, fp16 NaN, packing cross-contamination)을 모르고 시작하면 **조용히 망가진 학습**을 얻게 됩니다.
- 합성 데이터를 teacher 그대로 만들고 **그걸로 평가**하면 성능을 과대평가하게 됩니다.

이 킷은 위 함정을 코드 주석·노트북·본 문서에 **오개념 노트**로 박아 두어, 초심자가 밟지 않도록 돕습니다.

---

## §1 · 왜 이 구조인가 (Why)

**쉽게 말하면**: 이 킷은 "하나의 큰 튜토리얼"이 아니라 **"같은 부품을 공유하는 5개의 작은 완결 튜토리얼"**입니다. 부품(합성·학습·서빙·평가)은 `common/`에 한 번만 작성해 두고, 트랙은 데이터와 프롬프트만 갈아끼우면 됩니다. (멀티모달 트랙만 이미지 입력이라 구조가 조금 다릅니다 — §3.)

### 설계 축 대조

| 축 | 이 킷의 선택 | 대안 | 왜 이걸 골랐나 (조건부) |
|---|---|---|---|
| 학습 경로 | **HF DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA** | JumpStart 원클릭 | 커스텀 chat template·LoRA 타깃·bf16 등을 **직접 제어**해야 Gemma가 제대로 학습됩니다. 세밀한 제어가 필요 없다면 JumpStart도 유효한 선택입니다. |
| 서빙 | **SageMaker real-time endpoint (DJL LMI)** | Serverless / Async / Batch | 대화형 SLM은 상시 저지연이 필요하므로 real-time이 적합합니다. ⚠️ **Serverless는 GPU가 없어 LLM에 부적합**합니다. 배치성 작업이라면 Async/Batch를 고려하세요. |
| reasoning 모델 | **Bedrock Claude (Converse API)** | 자체 대형 모델 호스팅 | agentic 오케스트레이션은 대형 LLM이 유리하고, Bedrock은 상시 리소스 없이 토큰 단위로 과금됩니다. |
| 에이전트 | **Strands → AgentCore Runtime** | LangGraph 등 | Strands는 Bedrock/SageMaker와 정합하고, AgentCore는 관리형 런타임입니다. (LangGraph 옵션은 `pyproject.toml` extras로 열려 있습니다.) |
| 데이터 | **grounded 합성 (Bedrock Converse + critique/refine)** | 대형 오픈 합성셋 그대로 | 시드 도메인에 grounded 시켜 **task에 적합하고 라이선스도 안전**하게 만듭니다. |

### 세 가지 기술적 차이 (킷이 특별히 신경 쓴 지점)

1. **서비스 경계를 코드로 분리했습니다**: SageMaker endpoint 호출은 `boto3 sagemaker-runtime.invoke_endpoint()`(스트리밍은 `invoke_endpoint_with_response_stream`), Bedrock 호출은 `boto3 bedrock-runtime.converse()`를 씁니다. 이 둘은 **별개 서비스·별개 클라이언트**입니다 (`common/aws_utils.py`).
2. **DLC 이미지 태그를 env로 분리했습니다**: 태그는 자주 바뀌므로 코드에 박지 않고 `DLC_IMAGE_URI`/`DLC_TAG` env로 주입합니다(`common/dlc.py`). 로컬 `transformers`와 컨테이너 `transformers`도 서로 다르므로 구분합니다.
3. **Gemma 함정 방어를 기본값으로 넣었습니다**: `attn=eager`를 안전 기본값으로 두고, `bf16`을 강제하며(fp16 금지), packing은 flash-attn이 아니면 자동으로 끕니다 — 이 모두가 `tracks/*/scripts/train.py`에 내장되어 있습니다.

---

## §2 · E2E 파이프라인 (텍스트 트랙 7단계)

**쉽게 말하면**: 문제를 정하고 → 데이터를 만들고 → 학습하고 → 띄우고 → 에이전트로 감싸고 → 채점하는 흐름입니다. 각 단계가 노트북 하나에 대응합니다. (아래는 텍스트 트랙 01~04 기준입니다. 멀티모달 트랙 05는 더 짧은 별도 파이프라인 — 다이어그램 하단 참조.)

```
 task 정의
   │
   ▼
 [00_setup]      env·role·bucket·의존성 확인 (DRY_RUN 권장)
   │
   ▼
 [01_data_and_synthetic]  오픈 시드 로드 + grounded 합성(Bedrock Converse + critique/refine)
   │                       └ common/synth/bedrock_synth.py  →  messages JSONL
   ▼
 [02_train_sft_sagemaker]     HF DLC + TRL SFTTrainer(LoRA/QLoRA)  ← JumpStart 아님
   │                       └ tracks/*/scripts/train.py (로컬 dry-run ↔ SageMaker .fit() 겸용)
   │
   ├┈┈▶ (선택) [02a_train_grpo_sagemaker]  SFT→GRPO 정련(RLHF)  ← 추출·분류 트랙만
   ├┈┈▶ (선택) [02b_local_serve]           배포 전 로컬 vLLM 프리플라이트
   ▼
 [03_deploy_endpoint]     real-time endpoint (DJL LMI 서빙 컨테이너)
   │                       └ 호출: sagemaker-runtime.invoke_endpoint (별개: Bedrock)
   ▼
 [04_evaluate]            held-out 세트로 성공기준 수치화 (로컬·빠름·저렴)  ← 🔴 합성/학습셋으로 평가 금지
   │                       └ common/eval_utils.py
   ├┈┈▶ (선택) [04b_sagemaker_eval]        SageMaker managed 평가 잡 (🔴 별도 컴퓨트·비용)
   ▼
 [05_agentic_strands]     Strands Agent (reasoning=Bedrock Claude, tool=call_slm→endpoint)
   │
   ▼
 [06_agentcore_deploy]    AgentCore Runtime (ARM64, /invocations + /ping :8080)
   │                       └ agentcore/app.py
   ▼
 [99_cleanup]             endpoint·리소스 삭제 (과금 중단)
```

- **(선택) `02a_train_grpo_sagemaker`**: SFT 결과를 GRPO(RLHF)로 정련합니다. 리워드가 프로그램적으로 계산되는 **추출·분류 트랙에만** 있고, 요약·domain_qa에는 없습니다.
- **(선택) `02b_local_serve`**: SageMaker 배포 전 로컬 vLLM으로 프리플라이트하는 단계로, 모든 텍스트 트랙에서 제공됩니다.
- **(선택) `04b_sagemaker_eval`**: `04_evaluate`(로컬·빠름·저렴)의 대안으로 **SageMaker managed 평가 잡**(SDK v3 `sagemaker.train.evaluate`)을 돌립니다. evaluator 3종(`BenchMarkEvaluator` 전 트랙 · `CustomScorerEvaluator` 추출·분류 · `LLMAsJudgeEvaluator` 요약·QA)을 트랙 성격에 맞게 조합하며, 모든 텍스트 트랙(01~04)에서 제공됩니다(멀티모달 05 제외). 🔴 별도 컴퓨트·비용이 발생합니다.

### 멀티모달 트랙(05)은 별도 파이프라인 (5단계, 이미지 입력)

`tracks/05_multimodal_extraction`은 이미지 → 구조화 JSON 추출이라 합성 데이터 단계가 없고 노트북 세트가 다릅니다:

```
 [00_setup] ─▶ [01_data_explore] ─▶ [02_train_mm_sagemaker] ─▶ [03_deploy_mm_endpoint] ─▶ [99_cleanup]
              cord-v2 영수증           vision tower 동결 +          멀티모달 endpoint
              이미지+JSON 탐색          language LoRA               (이미지 입력 허용)
              (합성 단계 없음)          scripts/train_mm.py
```

> 서빙은 **이미지 입력을 받는 멀티모달 endpoint**입니다(텍스트 전용으로 재-export 하지 않음). agentic/agentcore 단계는 텍스트 트랙 전용이라 05에는 없습니다.

---

## §3 · 5개 독립 트랙 + 얇은 `common/`

**쉽게 말하면**: 텍스트 4개 트랙은 **데이터셋과 프롬프트만 다르고 파이프라인은 동일**합니다. 그래서 공통 부품은 `common/`에 한 번만 두면 됩니다. 멀티모달 트랙(05)만 이미지 입력이라 노트북 세트가 다릅니다.

| 트랙 디렉토리 | task | 시드 데이터셋 (라이선스) | 평가 메트릭 |
|---|---|---|---|
| `tracks/01_extraction_to_json` | 텍스트 → 구조화 JSON (**플래그십**) | `glaiveai/glaive-function-calling-v2` (apache-2.0) | arg_f1 + valid_json_rate + name_accuracy |
| `tracks/02_classification` | intent 분류 | `mteb/banking77` (mit) | macro-F1 + accuracy |
| `tracks/03_summarization` | 문서 요약 | `FiscalNote/billsum` (cc0-1.0) | ROUGE-L + LLM-judge |
| `tracks/04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) | LLM-judge + ROUGE-L |
| `tracks/05_multimodal_extraction` | **이미지 → 구조화 JSON** (영수증, gemma-4 vision) | `naver-clova-ix/cord-v2` (cc-by-4.0) | valid_json_rate + field 정확도 |

**텍스트 트랙(01~04)**은 동일한 노트북 세트를 가집니다: `00_setup → 01_data_and_synthetic → 02_train_sft_sagemaker → (02a_train_grpo_sagemaker, 추출·분류만) → (02b_local_serve) → 03_deploy_endpoint → 04_evaluate → (04b_sagemaker_eval, 선택) → 05_agentic_strands → 06_agentcore_deploy → 99_cleanup`. `02a`(SFT→GRPO 정련)는 리워드가 프로그램적인 추출·분류 트랙에만, `02b`(로컬 vLLM 프리플라이트)는 선택 단계이며, `04b`(SageMaker managed 평가)는 `04_evaluate`의 선택 대안입니다(🔴 별도 비용). 트랙별 특이값은 `tracks/*/track_data.py`(시드 로드+어댑터)와 `common/config.py`의 `TRACKS` 레지스트리에 정의되어 있습니다.

**멀티모달 트랙(05)**은 이미지 입력이라 구조가 다릅니다: `00_setup → 01_data_explore → 02_train_mm_sagemaker → 03_deploy_mm_endpoint → 99_cleanup` (합성 데이터 단계 없음). 학습은 `tracks/05_multimodal_extraction/scripts/train_mm.py`(AutoModelForImageTextToText + processor, vision tower 동결 + language LoRA)를 쓰고, 서빙은 이미지 입력을 받는 멀티모달 endpoint입니다.

### 공유되는 `common/` 부품

| 파일 | 역할 |
|---|---|
| `common/config.py` | `MODEL_ID`/region/role/bucket 플레이스홀더+env, `DRY_RUN`, `TRACKS` 레지스트리 |
| `common/gemma_format.py` | 표준 `messages` 어댑터 (`apply_chat_template`에 위임, 수동 마커 금지) |
| `common/aws_utils.py` | `invoke_endpoint`(sagemaker-runtime) · `converse`(bedrock-runtime) · CloudWatch 링크 · 비용 가드 |
| `common/dlc.py` | DLC 이미지 URI 해석 (계정 `763104351884`, 태그는 env 주입) |
| `common/llm_gateway.py` | (LiteLLM) Bedrock+SageMaker endpoint 단일 인터페이스 |
| `common/synth/bedrock_synth.py` | grounded 합성 (Converse + critique/refine, boto3만·무의존성) |
| `common/eval_utils.py` | 트랙별 메트릭 (추출/분류/요약/QA) + Bedrock LLM-judge |
| `tracks/*/scripts/train.py` | self-contained 학습 (로컬 dry-run ↔ SageMaker 겸용) |
| `agentcore/app.py` | AgentCore Runtime 엔트리포인트 (Strands 에이전트 호스팅) |

> ❓ **오개념 노트 — "트랙끼리 뭔가 공유하니 순서대로 해야 하나?"**
> 그렇지 않습니다. 5개 트랙은 **완전히 독립된 E2E**입니다. 관심 있는 트랙 하나만 `00→99`로 돌려도 완결됩니다. `common/`은 코드 중복을 제거하기 위한 것일 뿐, 실행 의존성이 아닙니다.

---

## §4 · 모델 선택 (Gemma)

| 모델 | 기본/승급 | 라이선스 | gated? | HF_TOKEN |
|---|---|---|---|---|
| `google/gemma-3-4b-it` | **기본** (텍스트 SLM 스윗스팟) | 커스텀 **Gemma Terms** | 🔴 gated | 필요 (약관 수락) |
| `google/gemma-4-12B-it` | 승급 (`MODEL_ID`만 교체) | **apache-2.0** | ungated | 불필요 |
| `gemma-4-31B` | — | — | — | (요청대로) **제외** |

> ❓ **오개념 노트 — "gated는 4b, ungated는 12B라니 헷갈린다."**
> 라이선스는 **모델 계열**을 따릅니다: Gemma 3/2/3n은 gated + Gemma Terms(서빙 시 use-restriction 전파 의무)이고, **Gemma 4는 apache-2.0 + ungated**입니다. 즉 승급(12B)이 오히려 라이선스 마찰이 적습니다. 🔴 재배포/서빙 전에 **live 모델 페이지의 라이선스 배너를 재확인**하세요.

승급 방법은 간단합니다: `export MODEL_ID=google/gemma-4-12B-it`만 실행하면 되고, 그 외 코드 변경은 필요 없습니다. 다만 12B는 학습/추론 인스턴스가 더 커야 하므로 `TRAIN_INSTANCE_TYPE`/`INFER_INSTANCE_TYPE`도 함께 조정하셔야 합니다.

### Gemma 학습 관용구 (train.py에 기본 내장)
- chat template은 `-it` 토크나이저에 내장되어 있으므로 `apply_chat_template`에 위임합니다(수동으로 `<start_of_turn>`를 조립하지 마세요). system role이 거부되면 첫 user 턴에 fold합니다.
- LoRA는 `target_modules="all-linear"` + `modules_to_save=["lm_head","embed_tokens"]`로 설정합니다.
- `bf16`은 필수이며(🔴 fp16는 Gemma에서 NaN을 유발), `attn=eager`가 안전 기본값입니다.
- packing은 flash-attn일 때만 켜집니다 — eager/sdpa에서는 **cross-contamination을 방지하기 위해 자동으로 꺼집니다**.

---

## §5 · 실행 방법 (How to run)

**쉽게 말하면**: (1) `uv`로 설치하고, (2) env를 주입한 뒤, (3) `DRY_RUN=1`로 파이프라인을 확인하고 나서 실제 실행으로 넘어가시면 됩니다.

### 1) 설치 — `uv` 권장 (`pyproject.toml`)
```bash
# uv 미설치 시:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml     # 또는 재현 설치:  uv sync
# 최신화:  uv lock --upgrade  /  uv lock --upgrade-package transformers
```
> pip만 쓰신다면 `pip install -r requirements.txt`를 실행하세요. 🔴 버전은 `>=` floor로 고정되어 있으며(실측 2026-07: transformers 5.14.1 / trl 1.8.0 / peft 0.19.1 / datasets 5.0.0), 실행 전에 재확인하시기 바랍니다.

### 2) 환경변수 (시크릿·계정ID·절대경로 하드코딩 금지)
```bash
export HF_TOKEN=hf_xxx                                  # gemma-3 계열(gated)만. gemma-4면 불필요
export AWS_REGION=us-east-1                              # 🔴 리전 재확인
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=us.anthropic.claude-...  # 🔴 inference-profile prefix 포함, 상세페이지에서 정확 ID
export DRY_RUN=1                                         # 먼저 파이프라인 검증, 실제 실행 시 0
```

### 3) DRY_RUN 우선
`DRY_RUN=1`로 두면 학습은 소량·1 epoch·짧은 seq로, 합성은 소량으로 돌면서 **파이프라인만** 검증합니다(`common/config.py:is_dry_run`). 🔴 **GPU dry-run은 L40S에서 검증되었으며**, 다른 GPU/메모리에서는 배치·seq 길이를 재조정해야 할 수 있습니다. 파이프라인이 확인되면 `DRY_RUN=0`으로 실제 실행하세요.

> ❓ **오개념 노트 — "로컬 `transformers`와 SageMaker가 같은 버전이겠지?"**
> 그렇지 않습니다. 로컬 env의 `transformers`는 데이터 준비/dry-run용이고, **SageMaker 컨테이너 버전은 DLC 이미지 태그**가 결정합니다. 컨테이너 안에서 상위 버전이 필요하면 `tracks/*/scripts/requirements.txt`가 이를 업그레이드합니다.

---

## §6 · 문서 · 노트북 매핑표

### 상세 문서 (docs/) — 개념은 여기로
| 문서 | 다루는 것 | 대응 노트북 단계 | 주요 참조 코드 |
|---|---|---|---|
| **01 inference `[CORE]`** | SageMaker 추론 4옵션, endpoint 호출(sagemaker-runtime), DJL LMI 서빙 | `03_deploy_endpoint` | `common/aws_utils.py`, `common/dlc.py` |
| **02 finetuning** | HF DLC + TRL SFTTrainer, LoRA/QLoRA, Gemma 관용구 | `02_train_sft_sagemaker` | `tracks/*/scripts/train.py`, `common/gemma_format.py` |
| **03 synthetic** | grounded 합성(Converse + critique/refine), 라이선스·PII | `01_data_and_synthetic` | `common/synth/bedrock_synth.py`, `tracks/*/track_data.py` |
| **04 agentic** | Strands(Bedrock reasoning + SLM tool) → AgentCore Runtime | `05_agentic_strands`, `06_agentcore_deploy` | `agentcore/app.py`, `common/llm_gateway.py` |
| **05 serving containers** | DJL LMI vs vLLM 단독, 백엔드 선택(OPTION_ROLLING_BATCH) | `03_deploy_endpoint` | `common/dlc.py` |
| **evaluate** (해당 시) | held-out 평가, 트랙별 메트릭, LLM-judge (로컬 `04_evaluate` / 선택 managed `04b_sagemaker_eval`) | `04_evaluate`, (선택) `04b_sagemaker_eval` | `common/eval_utils.py` |

> 상세 문서(01~05)는 로컬에 있으면 `docs/` 하위에서 참조하실 수 있습니다. 없더라도 이 개요만으로 실행 흐름은 파악하실 수 있습니다.

### 노트북 단계 ↔ 산출물
| 노트북 | 산출물 | 비고 |
|---|---|---|
| `00_setup` | env/role/bucket 확인 | `DRY_RUN` 권장 |
| `01_data_and_synthetic` | `messages` JSONL(합성) | S3 업로드 |
| `02_train_sft_sagemaker` | 학습 잡 → 모델 아티팩트(S3) | LoRA 머지 산출물 포함 |
| `03_deploy_endpoint` | real-time endpoint | 🔴 과금 시작 |
| `04_evaluate` | 메트릭 리포트 | held-out만 (로컬·빠름·저렴) |
| (선택) `04b_sagemaker_eval` | managed 평가 리포트 | SageMaker 평가 잡, 텍스트 트랙만 · 🔴 별도 비용 |
| `05_agentic_strands` | 로컬 에이전트 루프 | endpoint + Bedrock |
| `06_agentcore_deploy` | AgentCore Runtime | 🔴 Runtime 과금 |
| `99_cleanup` | 리소스 삭제 | 과금 중단 |

---

## §7 · 비용/cleanup · 라이선스

### 🔴 비용 & cleanup (필수)
- **real-time endpoint는 삭제 전까지 시간당 과금**됩니다(GPU 인스턴스). 실습 후에는 반드시 `99_cleanup.ipynb`를 실행하거나 `predictor.delete_endpoint()`를 호출하세요.
- **AgentCore Runtime**은 배포 시 Runtime 리소스가 과금되므로, 사용하지 않을 때는 정리해 주세요.
- **Bedrock Converse**는 토큰 단위로 과금되며(상시 리소스는 없지만 대량 합성/평가 시 비용이 발생), Training Job은 잡이 종료되면 과금이 멈춥니다.
- 각 노트북은 학습/배포 직후 **CloudWatch 다이렉트 링크**를 출력합니다(`common/aws_utils.cw_links`).

> ❓ **오개념 노트 — "endpoint를 안 부르면 공짜겠지?"**
> 그렇지 않습니다. real-time endpoint는 **호출 여부와 무관하게 provisioned 인스턴스가 시간당 과금**됩니다. 쓰지 않는다면 삭제하는 것이 정답입니다.

### ⚠️ 라이선스 요약
- Gemma 3/2/3n은 **Gemma Terms + gated**입니다(HF 토큰·약관 수락 필요, 서빙 시 use-restriction 전파 의무).
- Gemma 4는 **apache-2.0 + ungated**로 마찰이 가장 적습니다.
- 시드 데이터셋은 전부 permissive한 것만 선별했으나, share-alike(dolly 등) 파생물은 주의하시기 바랍니다.
- 🔴 재배포/서빙 전에 각 모델·데이터셋의 **live 라이선스 배너를 재확인**하세요.

---

## §8 · 근거 (라이브 검증 2026-07)

🔴 아래 값 중 **모델 ID·DLC 이미지 태그·SDK 버전·리전·GA 상태**는 빠르게 바뀌므로, **실행 직전에 재확인**하시기 바랍니다(`# TODO verify`).

| 주제 | URL |
|---|---|
| SageMaker 추론 옵션(Real-time/Serverless/Async/Batch) | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html |
| endpoint 호출 (`invoke_endpoint`) | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html |
| endpoint 스트리밍 호출 | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointWithResponseStream.html |
| Bedrock Converse API (별개 서비스) | https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html |
| Bedrock inference profiles (us./eu./apac./global.) | https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html |
| DLC 이미지 목록 (계정 763104351884, 태그) | https://github.com/aws/deep-learning-containers/blob/master/available_images.md |
| DLC 페이지(reference) | https://aws.github.io/deep-learning-containers/reference/available_images/ |
| DJL LMI 서빙 컨테이너(OPTION_ROLLING_BATCH=vLLM/TRT-LLM) | https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html |
| DJL LMI (deep-learning-containers 레포) | https://github.com/aws/deep-learning-containers/tree/master/large_model_inference |
| HF on SageMaker (DLC + estimator) | https://huggingface.co/docs/sagemaker/index |
| TRL SFTTrainer | https://huggingface.co/docs/trl/sft_trainer |
| PEFT LoRA | https://huggingface.co/docs/peft/index |
| Gemma 모델 카드(라이선스·chat template) | https://huggingface.co/google/gemma-3-4b-it |
| Strands Agents | https://github.com/strands-agents/sdk-python |
| AgentCore (bedrock-agentcore SDK) | https://github.com/aws/bedrock-agentcore-sdk-python |
| AgentCore Runtime 문서 | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html |

---

**다음 문서 →** `docs/01_inference.md` `[CORE]` · **트랙 시작 →** `tracks/01_extraction_to_json/00_setup.ipynb`

<sub>이 문서의 fact-critical 주장(추론 4옵션·서비스 경계·DLC 계정/패턴·DJL LMI 백엔드·Gemma 라이선스 gated 여부·학습 경로=HF DLC(≠JumpStart)·AgentCore HTTP 계약)은 aws-fact-checker 검증 대상으로 전달됨. 🔴 빠르게 바뀌는 값은 실행 전 재확인.</sub>
