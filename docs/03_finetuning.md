# 03 · 파인튜닝 접근법 — DLC + 커스텀 train.py(TRL SFTTrainer · PEFT LoRA/QLoRA)

!!! info "Scope"
    Amazon SageMaker AI에서 Gemma를 처음 파인튜닝해 보는 엔지니어를 위한 문서입니다.
    HuggingFace `transformers`/`trl`은 대략 알지만 SageMaker AI 학습 Job·DLC·LoRA 관용구는
    처음이어도 괜찮습니다.

    - **선행 조건**: 각 코스의 `01_data_and_synthetic.ipynb`까지 실행해
      `data/train.jsonl`(conversational `messages`)을 만들어 둔 상태.
      Training Job이 무엇이고 `/opt/ml/*` 경로 규약이 왜 있는지가 낯설면
      [SageMaker AI 기초](01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: 학습 경로 선택 · Gemma 관용구 · LoRA/QLoRA · 머지/re-export ·
      `MaxRuntimeExceeded` 함정 · SFT→GRPO 데이터 규율
    - **여기서 다루지 않는 것**: endpoint 배포는 [SageMaker AI 추론](04_sagemaker_inference.md),
      합성 데이터는 [Grounded 합성 데이터](02_synthetic_data.md)

이 문서와 관련된 리포지토리 파일:

- `common/config.py`: Gemma 프리셋(`GEMMA4_PRESETS`/`DEFAULT_MODEL_ID`), 코스 정의(`TRACKS` — 코드 식별자는 초기 이름을 그대로 유지합니다), HF 토큰 조회
- `common/dlc.py`: DLC 이미지 URI 해석(`DLC_IMAGE_URI` → `DLC_REPOSITORY`+`DLC_TAG` → SDK 폴백)
- `common/gemma_format.py`: 코스별 raw row를 표준 `messages`로 변환, 수동 호출용 `fold_system_into_user` 헬퍼
- `common/grpo_data.py`: GRPO prompt 소스 준비(`holdout`/`synth`/`failures`)
- `tracks/*/scripts/train.py`: SFT 학습 스크립트(LoRA/QLoRA, 머지, 텍스트 re-export)
- `tracks/*/scripts/train_grpo.py`: GRPO 정련 스크립트(프로그램적 reward, 추출·분류 코스)
- `tracks/05_multimodal_extraction/scripts/train_mm.py`: 이미지→JSON 멀티모달 SFT(vision tower 유지, re-export 없음)
- `tracks/*/scripts/requirements.txt`: 학습 컨테이너 안에서 올리는 transformers/trl/peft 핀

노트북 순서: `01_data_and_synthetic` → `02_train_sft_sagemaker` → (선택) `02a_train_grpo_sagemaker`

!!! warning "빠르게 바뀌는 값"
    **모델 ID·DLC 이미지 태그·SDK 버전·리전·GA 상태**는 이 문서에서 가장 빨리 낡는 부분입니다. `transformers`/`trl`/`peft` 핀, ECR 태그, `ml.g5`/`ml.g6` 용량 가용성, Gemma 라이선스 배너는 **실행 직전에 다시 확인**하세요.
    계정 ID·role ARN·토큰·버킷·절대경로는 문서와 코드 어디에도 하드코딩하지 않고 전부 env/노트북으로 주입합니다.

---

## TL;DR

**이 kit은 SageMaker AI 파인튜닝 두 경로(JumpStart vs 자체 스크립트) 중 DLC 컨테이너 + TRL `SFTTrainer` + PEFT LoRA/QLoRA를 담은 self-contained `scripts/train.py`를 택했습니다. 커스텀 학습 로직과 최신 Gemma를 즉시 쓰기 위해서입니다.**

1. **JumpStart가 아니라 커스텀 스크립트인 이유** — JumpStart는 정해진 모델을 정해진 레시피로 빠르게 돌리는 데 최적입니다. 다만 최신 Gemma 릴리스 반영과 커스텀 SFT 로직(chat template 처리, LoRA 타깃 제한, 텍스트 re-export)의 제어가 어렵습니다. 자세히는 [왜 커스텀 train.py 경로인가](#왜-커스텀-trainpy-경로인가)를 보세요.
2. **`train.py` 한 파일이 로컬 `--dry_run`과 SageMaker AI 학습 Job을 겸합니다.** 로컬 GPU에서 파이프라인을 검증한 그 파일이 클라우드에서 그대로 돕니다. 자세히는 [train.py — 로컬 dry-run과 SageMaker AI 학습 Job](#trainpy--로컬-dry-run과-sagemaker-ai-학습-job)을 보세요.
3. **Gemma 관용구 6종은 협상 불가입니다**: chat template 위임, 멀티모달 base는 `language_model` 한정 LoRA, bf16(fp16 금지), `eager` attention, flash-attention 아니면 packing off, boolean 하이퍼는 `str2bool`. 자세히는 [Gemma 파인튜닝 관용구](#gemma-파인튜닝-관용구)를 보세요.
4. **`stopping_condition`을 생략하면 SDK가 1시간을 넣습니다.** 학습이 100% 끝나고 머지 중에 Job이 죽어 배포 불가가 됩니다. 자세히는 [MaxRuntimeExceeded — 학습 뒤 머지에서 잘리는 함정](#maxruntimeexceeded--학습-뒤-머지에서-잘리는-함정)을 보세요.
5. **SFT 데이터를 GRPO에 재사용하면 학습이 아예 안 됩니다.** rollout이 전부 만점이 되어 advantage가 0으로 수렴합니다. 자세히는 [SFT에서 GRPO로 — 데이터를 갈아야 하는 이유](#sft에서-grpo로--데이터를-갈아야-하는-이유)를 보세요.

---

## 기존 Pain Point

처음 SageMaker AI에서 Gemma를 파인튜닝할 때 실제로 부딪히는 벽은 다음과 같습니다.

- **"JumpStart 버튼 하나면 되는 거 아니야?"** 싶지만, 막상 돌려보면 최신 Gemma가 목록에 없거나 chat template·특수토큰·저장 포맷 같은 세부를 건드릴 수 없어 결과가 이상하게 나옵니다.
- 로컬에서 잘 되던 학습 스크립트가 SageMaker AI에서 **`--use_qlora`에서 크래시**합니다. `store_true` 플래그에 `--use_qlora True`가 들어오면서 죽는 것인데, 원인을 모른 채 몇 시간을 날리기 쉽습니다.
- **fp16으로 돌렸더니 loss가 NaN**이 됩니다. Gemma에서 흔히 겪는 함정입니다.
- 학습은 됐는데 **출력이 엉망**입니다. chat template을 손으로 조립했거나, packing으로 샘플이 서로 오염된 경우입니다.
- 학습은 끝났는데 **서빙이 안 됩니다**. 아티팩트 루트에 완전한 HF 모델이 없거나(어댑터만 있거나), 멀티모달 config가 남아 vLLM이 image processor를 찾다가 죽습니다.
- 학습 Job이 `Failed`도 아닌 **`Stopped`로 끝나고 `FailureReason`이 비어 있습니다**. 로그에는 에러가 한 줄도 없습니다.

이 문서는 위 함정들을 `train.py`와 노트북이 어떻게 **미리 막아 두었는지** 설명합니다.

---

## 왜 커스텀 train.py 경로인가

!!! abstract "쉽게 말하면"
    SageMaker AI에서 파인튜닝하는 문서화된 길은 두 갈래입니다. 하나는 **JumpStart**로 미리 포장된 모델·레시피를 SDK 몇 줄로 돌리는 방식이고, 다른 하나는 **직접 학습 스크립트 + DLC 컨테이너**를 써서 `ModelTrainer`에 `entry_script`를 넘기는 방식입니다.
    전자는 "메뉴 주문", 후자는 "장을 봐서 직접 요리"에 비유할 수 있습니다. 이 kit은 레시피(Gemma 관용구)를 정확히 통제해야 하므로 후자를 골랐습니다.

### JumpStart vs 자체 train.py

| 축 | [JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) 파인튜닝 | **DLC + 자체 `train.py`** (이 kit) |
|---|---|---|
| 인터페이스 | `JumpStartEstimator` / 콘솔 | `sagemaker.train.model_trainer.ModelTrainer` + `SourceCode` |
| 모델 커버리지 | 큐레이션된 목록(신규 릴리스 지연 가능) | HF Hub의 최신 Gemma 즉시 (`MODEL_SIZE`/`MODEL_ID`만 교체) |
| 학습 로직 제어 | 제한적(정해진 레시피) | 완전 제어(LoRA 타깃 정규식, 텍스트 re-export, packing 안전장치, 머지) |
| 커스텀 의존성 | 어려움 | `scripts/requirements.txt`로 자유롭게(`transformers>=5.14.1` 등) |
| 로컬 == 클라우드 | 아니오 | 예 — 동일 `train.py`가 `--dry_run`/학습 Job 겸용 |
| 진입 난이도 | 낮음 | 중간(대신 투명·이식성) |
| 언제 고르나 | 표준 레시피로 충분·빠른 baseline | 커스텀 로직 또는 최신 모델이 필요할 때 |

**SDK 버전도 함께 확인하세요. v3에는 `HuggingFace` estimator가 아예 없습니다.**
[SageMaker Python SDK V3 개요](https://sagemaker.readthedocs.io/en/stable/#migration-from-v2)는 v3가 "Estimator·Model·Predictor 같은 legacy 인터페이스를 `ModelTrainer`/`ModelBuilder`로 대체한다"고 명시합니다.
`HuggingFace` estimator는 그 `Estimator` 계열의 프레임워크 서브클래스였으므로 **함께 제거**됐습니다.
그래서 학습은 `ModelTrainer`(+`SourceCode`/`Compute`/`InputData`/`StoppingCondition`)로 정의합니다.

??? info "더 읽을 거리 — 이 kit이 확인한 SDK 버전"
    이 kit은 SDK 3.16.0에서 실측했습니다(`HuggingFace` import 부재는 설치본에서 직접 확인했습니다).
    `pyproject.toml`이 고정하는 것은 `sagemaker>=3.16.0` floor이므로, 설치본은 그보다 높은 버전일 수 있습니다.

지금까지 나온 이름(`JumpStartEstimator`·`ModelTrainer`·`ModelBuilder`·제거된 `Estimator`)은 전부 **SageMaker Python SDK라는 한 계층 안의 클래스**입니다.
그 계층이 어디에 앉아 있는지를 보면 위 표의 선택이 실제로 얼마만큼의 범위를 가지는지도 함께 드러납니다.

[![Training Job을 만드는 세 가지 호출 계층 다이어그램. 왼쪽에는 호출 주체로 Dev desktops, App Servers, Amazon EC2, SageMaker Notebooks, Amazon EMR이 세로로 놓이고 각각에서 가운데의 세 계층으로 화살표가 들어간다. 위쪽 AWS SDKs 박스는 CreateTrainingJob()과 CreateModel()을 노출하며 Java·Node·PHP·.NET·Ruby·Python·Go·C++를 지원하고, 가운데 SageMaker Python SDK 박스는 ModelTrainer.train()과 ModelBuilder.deploy()를, 아래쪽 SageMaker Spark Library 박스는 org.apache.spark.ml.Estimator interface를 노출한다. 세 박스에서 나온 화살표는 모두 오른쪽의 같은 대상인 SageMaker AI Training Job으로 모인다](images/sm_sdks.png)](images/sm_sdks.png)

*세 화살표의 도착지가 하나라는 것이 요점입니다. 어느 계층으로 부르든 AWS가 만드는 리소스는 같은 Training Job입니다.*

**이 절의 선택은 계층을 갈아타는 결정이 아니라, 같은 계층 안에서 어느 래퍼를 쓸지의 결정입니다.** 위 표의 대안인 JumpStart도 `JumpStartEstimator`라는 같은 계층의 클래스입니다.
어느 쪽을 골라도 AWS 쪽에 도착하는 것은 `CreateTrainingJob` 하나입니다. 그래서 Job 상태·시간 제한·[경로 규약](01_sagemaker_basics.md#경로-규약--컨테이너-안의-정해진-경로) 같은 규칙은 경로 선택과 무관하게 똑같이 적용됩니다.

가운데 줄에 적힌 두 API가 그대로 이 kit의 두 노트북입니다.

| 그림의 API | 이 kit의 노트북 |
|---|---|
| `ModelTrainer.train()` | `02_train_sft_sagemaker` |
| `ModelBuilder`(+`deploy()`) | `03_deploy_endpoint` |

맨 위 줄(AWS SDKs)도 이 kit이 실제로 씁니다. **Job을 만드는 것은 Python SDK지만, 만든 뒤 들여다보고 지우는 코드는 boto3입니다.**

- 조회: `common/aws_utils.py`의 `training_job_status()`가 `sagemaker` 클라이언트로 `describe_training_job`을 호출합니다.
- 삭제: `99_cleanup.ipynb`가 `delete_endpoint`/`delete_endpoint_config`/`delete_model`을 씁니다.
- 호출: endpoint 호출은 `sagemaker-runtime`의 `invoke_endpoint`입니다.

Python SDK가 감싸 주지 않는 조회·삭제 API가 필요하면 한 층 내려가면 됩니다.
그 층은 그림의 언어 목록(Java·Node·Go 등) 어디서나 같아서, 학습 Job 제출을 Java 애플리케이션에서 해도 됩니다. 그때 없는 것은 `SourceCode`·`Compute` 같은 **편의 래퍼뿐**입니다.
아래 줄의 SageMaker Spark Library는 EMR/Spark 파이프라인용이라 이 kit에는 등장하지 않습니다.

### 기술적 차이 3가지

1. **최신 모델 반영** — Gemma는 분기마다 릴리스가 갱신됩니다. 이 경로에서는 `MODEL_SIZE`(E2B/E4B/12B/26B-A4B/31B) 또는 `MODEL_ID` env 하나만 바꾸면 승급이 끝납니다(`common/config.py`의 `GEMMA4_PRESETS`, `DEFAULT_MODEL_ID`). JumpStart는 큐레이션 목록에 오를 때까지 기다려야 할 수 있습니다.
2. **학습 로직 투명성** — Gemma는 손대야 할 관용구가 많습니다([Gemma 파인튜닝 관용구](#gemma-파인튜닝-관용구)). `train.py`는 이 결정들을 코드로 명시하고 있어 리뷰·재현·이식이 쉽습니다. 특히 멀티모달 base를 텍스트로 서빙하려면 저장 단계에 손을 대야 하는데, 관리형 레시피에서는 불가능한 개입입니다.
3. **이식성(로컬↔SageMaker AI 단일 소스)** — SageMaker AI는 `source_dir`만 컨테이너에 올립니다. 그래서 `train.py`는 `common/`에 **의존하지 않는 self-contained** 파일로 작성했습니다. 로컬 GPU에서 검증한 바로 그 파일이 클라우드에서 그대로 돕니다.

??? question "오개념 — “JumpStart가 더 production-ready 아닌가요?”"
    그렇지 않습니다. production 여부는 경로가 아니라 운영(체크포인트·모니터링·재현성)이 결정합니다.
    커스텀 스크립트 경로도 managed spot·checkpoint·CloudWatch로 충분히 production-ready합니다. 선택 기준은 성숙도가 아니라 "레시피 제어가 필요한가"입니다.

---

## Gemma 파인튜닝 관용구

!!! abstract "쉽게 말하면"
    Gemma는 "그냥 돌리면" 미묘하게 틀립니다. 아래 6가지는 Gemma 모델 카드·TRL 문서와 이 kit의 실측에 근거한 관용구이며, `train.py`에 이미 반영되어 있습니다.
    데이터 포맷부터 저장 포맷까지 한 줄로 이어져 있으므로, 하나만 빠뜨려도 학습이나 서빙 중 한쪽이 깨집니다.

```
  데이터(JSONL)                train.py 처리                      결과
  {"messages":[...]}  ─► apply_chat_template (SFTTrainer 자동)  ─► 올바른 -it 포맷
       (system?)      ─► 데이터 준비 때 첫 user턴에 fold(수동)  ─► 템플릿 에러 없음
                      ─► LoRA: language_model 한정(멀티모달)     ─► vision proj 미터치
                      ─► bf16 + eager + packing(조건부 off)      ─► NaN·오염 없음
                      ─► 머지 → 텍스트 arch로 re-export          ─► vLLM이 그대로 로드
```

### chat template과 system fold

- **마커를 손으로 조립하지 마세요.** Gemma **-it(instruction-tuned)** 토크나이저에는 chat template이 **내장**되어 있습니다. 데이터가 conversational 포맷(`{"messages":[{"role","content"},...]}`)이면 `SFTTrainer`가 **자동으로 `apply_chat_template`을 적용**합니다. 출력 role은 `assistant`가 아니라 `model`로 렌더링되며, 이 매핑도 템플릿이 처리합니다.
- **system role은 실행 전 재확인 대상입니다.** Gemma 계열 템플릿에는 전용 system 슬롯이 없는 경우가 많아, `{"role":"system",...}`을 넣으면 **템플릿 적용 시 예외가 납니다**. 정확한 동작은 모델별 `tokenizer_config`에 달려 있습니다.
- **자동 복구는 없습니다.** `common/gemma_format.py`의 `render_prompt`는 `apply_chat_template`을 그대로 호출할 뿐 try/except 재시도를 하지 않습니다. 폴백 함수(`fold_system_into_user`)는 제공되지만 **호출은 사용자 몫**입니다.
- 따라서 system 지시가 필요하면 **반드시 첫 `user` 턴 맨 앞에 직접 접어 넣거나(fold)**, 데이터 준비 단계(`01_data_and_synthetic`)에서 미리 병합해 두세요. `{"role":"system"}` 행을 그대로 `data/train.jsonl`에 넣으면 예외가 **SageMaker AI 학습 Job 안에서** 터집니다(용량 대기 + DLC pull + GPU 과금을 다 치른 뒤).

??? info "더 읽을 거리 — 템플릿 원본 문서"
    템플릿이 실제로 뱉는 `<start_of_turn>`/`<end_of_turn>` 마커 구조는 [Gemma formatting and system instructions](https://ai.google.dev/gemma/docs/core/prompt-structure)가 원본입니다.
    conversational 포맷을 `SFTTrainer`가 자동 처리한다는 규칙은 [TRL의 dataset formats 문서](https://huggingface.co/docs/trl/en/dataset_formats)에 있습니다.

??? question "오개념 — “system 프롬프트를 messages에 그냥 넣으면 되지 않나요?”"
    될 때도 있고 안 될 때도 있고, **실패하면 kit이 대신 고쳐 주지 않습니다.** 템플릿이 system role을 지원하지 않으면 `apply_chat_template`에서 에러가 납니다.
    `common/gemma_format.py`의 `build_messages`는 system을 그대로 담아 주고, `render_prompt`는 `apply_chat_template`을 바로 호출합니다(자동 재시도 없음). `fold_system_into_user`는 **직접 불러야 하는** 폴백 함수입니다.
    학습 데이터는 처음부터 fold된 형태로 만들어 두세요.

### LoRA target — 멀티모달은 language_model만

```python
# 텍스트 전용 base
LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",   # train.py 기본값(--lora_r/--lora_alpha/--lora_dropout)
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],           # 특수토큰 학습
)

# 멀티모달 base (gemma-4 전 사이즈 · gemma-3 4b+) — 위에서 두 인자만 교체
LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    modules_to_save=None,
)
```

**`r=16`·`alpha=16`(스케일 `alpha/r = 1.0`)이 출발점입니다.** `r`/`lora_alpha`/`lora_dropout`은 `train.py`의 CLI 인자 기본값이고, 노트북도 같은 값(`'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05`)을 제출합니다.
어댑터 용량을 늘리려면 `r`을 32/64로 올리고 보통 `alpha`도 같은 비율로 함께 올립니다. `bias="none"`과 `task_type="CAUSAL_LM"`은 고정입니다.

- 텍스트 전용 base라면 [PEFT `LoraConfig` API 문서](https://huggingface.co/docs/peft/en/package_reference/lora)의 `target_modules="all-linear"`로 모든 linear에 어댑터를 붙이고, `modules_to_save=["lm_head","embed_tokens"]`로 임베딩·출력 헤드를 full-train 대상에 넣습니다. LoRA는 원래 이 둘을 건드리지 않으므로, 빠뜨리면 chat 특수토큰 표현이 어긋납니다.
- **gemma-4는 전 사이즈가 멀티모달입니다**(vision, E4B/12B는 audio 포함). 텍스트 SFT라도 로더가 `AutoModelForImageTextToText`이므로 타깃 결정이 달라집니다.
- 이름 리스트나 `all-linear`를 주면 **크래시합니다.** language의 proj는 평범한 `nn.Linear`지만 vision/audio tower의 동명 proj는 커스텀 `Gemma4ClippableLinear`라 peft가 지원하지 않습니다(`ValueError: Target module ... is not supported`).
- 그래서 멀티모달에서는 **정규식으로 `language_model` 경로만 한정**합니다. 실측에서는 language의 `nn.Linear` 258개만 매칭됐고(ClippableLinear 0개), `get_peft_model`이 성공해 `lora_A` 516개가 부착됐습니다.
- 멀티모달에서는 embed/lm_head를 `modules_to_save`로 두면 vision 임베딩까지 얽힐 수 있어 생략합니다(순수 텍스트 LoRA).

### bf16 필수, fp16 금지

`SFTConfig(bf16=True)`로 학습하고 모델은 `dtype=torch.bfloat16`으로 로드합니다. **fp16 금지**입니다. fp16은 Gemma에서 오버플로/NaN을 일으킵니다.
bf16을 지원하지 않는 GPU라면 QLoRA의 `bnb_4bit_compute_dtype=torch.bfloat16` 경로를 쓰되 하드웨어 호환성을 먼저 확인하세요.
`transformers` 5.x부터는 `torch_dtype`이 `dtype`으로 이름이 바뀌었습니다(구 이름은 deprecation 경고).

### attention — eager가 안전 기본

`attn_implementation="eager"`가 Gemma의 **안전 기본값**입니다(soft-cap / sliding-window 정합성 때문).
`train.py`가 받는 값은 `eager`·`sdpa`·`flash_attention_2` 세 가지이며, 특별한 이유가 없다면 eager로 시작하시기를 권장합니다.

### packing — flash-attention 아니면 off

**`train.py`는 flash-attention 계열일 때만 packing을 켭니다.** eager/sdpa에서는 `--packing True`를 주더라도 경고를 출력하고 자동으로 끕니다. 즉 안전 기본값 eager에서는 사실상 off이며, 이는 버그가 아니라 의도된 비활성화입니다.

packing은 여러 짧은 샘플을 한 시퀀스로 이어 붙여 throughput을 올려 줍니다. 하지만 attention이 샘플 경계를 마스킹하지 못하면 **샘플끼리 서로 참조하는 cross-contamination(교차 오염)** 이 발생합니다.
packing throughput이 필요하면 flash-attention을 명시적으로 선택하세요.

!!! danger "조용히 망가지는 학습"
    fp16 NaN과 packing 교차 오염은 **에러 없이 결과만 망가지는** 두 가지입니다. loss는 그럭저럭 떨어지는데 출력이 엉망이면 이 둘을 먼저 의심하세요.
    `bf16=True`와 packing 자동 차단은 `train.py`의 기본값이므로, 하이퍼파라미터로 억지로 뒤집지 마세요.

### boolean 하이퍼파라미터 — str2bool

- SageMaker AI는 **모든 하이퍼파라미터를 `--key value`로 직렬화**해 entry script에 넘깁니다. 즉 `use_qlora=True`가 `--use_qlora True`로 전달됩니다. 흔히 쓰는 `action="store_true"`는 값을 받지 않으므로 이때 **크래시**합니다. 그 변환을 하는 자리는 SDK v3의 [`hyperparameters_to_cli_args`](https://github.com/aws/sagemaker-python-sdk/blob/master/sagemaker-train/src/sagemaker/train/container_drivers/common/utils.py) 소스입니다.
- 해결책은 `type=_str2bool, nargs="?", const=True`입니다. 이렇게 하면 로컬의 bare-flag(`--dry_run`)와 SageMaker AI의 `--use_qlora True`를 **양쪽 모두** 받을 수 있습니다.

```python
def _str2bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")
```

??? info "더 읽을 거리"
    관용구의 근거를 코드 수준에서 확인하려면 [TRL 공식 저장소](https://github.com/huggingface/trl)와 [PEFT 공식 저장소](https://github.com/huggingface/peft)를 보세요. `SFTConfig`의 packing 처리와 `LoraConfig`의 `target_modules` 매칭 규칙이 버전마다 달라지므로, 핀을 올릴 때는 문서보다 소스가 빠릅니다.

---

## train.py — 로컬 dry-run과 SageMaker AI 학습 Job

**파일 하나가 두 무대에서 똑같이 공연합니다.** 리허설(로컬 GPU 소량)과 본공연(SageMaker AI 학습 Job)이 같은 대본을 씁니다.

```
로컬 개발 GPU                          SageMaker AI 학습 Job
─────────────                          ────────────────────
python train.py --dry_run              trainer.train(input_data_config=[InputData(...)])
  --train_file ./sample.jsonl              │  source_dir='scripts' 업로드
  --output_dir ./out                       │  (train.py + requirements.txt)
       │                                    ▼
       └── epochs=1, seq<=512, 32행     hyperparameters → --key value
           파이프라인만 검증                  SM_CHANNEL_TRAIN / SM_MODEL_DIR
```

오른쪽 열의 `trainer.train(...)` 한 줄이 실제로 무엇을 세우는지는, 그 호출이 만들어 내는 인프라를 보면 분명해집니다.

[![ModelTrainer 코드와 그 코드가 세우는 인프라의 대응 다이어그램. 왼쪽은 35줄짜리 ModelTrainer 스니펫이고, 점선 화살표 두 개가 오른쪽으로 나갑니다. 위쪽 점선은 `image_uris.retrieve(...)` 블록에서 Amazon ECR의 PyTorch Training Container Image로, 아래쪽 점선은 `estimator.train(input_data_config=[...])` 줄에서 Amazon SageMaker Managed Cluster로 이어집니다. 클러스터 안에는 Instance 1(Training container + EBS Volume)과 인스턴스 두 개가 더 있고, 오른쪽 실선은 `os.env('SM_MODEL_DIR')`이 s3://bucket/path/to/model로 나가는 방향, `os.env('SM_CHANNEL_TRAINING')`과 `SM_CHANNEL_TESTING`이 S3에서 EBS Volume으로 들어오는 방향을 나타냅니다.](images/sm_training.png)](images/sm_training.png)

*코드 한 줄이 어떤 인프라를 세우는지가 화살표로 드러납니다. 점선 둘은 `image_uris.retrieve()`가 ECR 이미지를, `estimator.train()`이 클러스터를 지정하는 방향이고, 오른쪽 실선은 컨테이너와 S3가 `SM_*` 환경변수로 주고받는 방향입니다.*

이 그림을 이 kit의 값으로 옮기면 이렇습니다.

| 그림의 요소 | 이 kit의 값 |
|---|---|
| 위쪽 점선 — ECR 이미지 | `.env`의 `DLC_IMAGE_URI`(완전 URI)를 `training_image`로 그대로 넘겨 태그까지 고정. SDK가 버전 조합으로 URI를 조립하는 `image_uris.retrieve()` 경로는 쓰지 않음 |
| 아래쪽 점선 — 클러스터 | `Compute(instance_type=..., instance_count=1)` — 인스턴스 박스 셋 중 하나만 뜨고, 여러 노드를 묶는 `distributed=Torchrun()`도 안 씀 |
| 오른쪽 실선 — `s3://bucket/path/to/model` | `output_data_config`를 생략해 SDK 기본 출력 경로에 맡기고, Job이 끝난 뒤 `job.model_artifacts.s3_model_artifacts`로 URI를 읽어 배포에 넘김 |

그리고 그림이 흘리기 쉬운 사실이 하나 있습니다. 오른쪽의 env 이름은 SageMaker AI가 정해 주는 상수가 아니라 **내가 붙인 채널 이름에서 파생**됩니다.
그림은 채널이 둘(`training`·`testing`)이라 `SM_CHANNEL_TRAINING`/`SM_CHANNEL_TESTING`입니다. 이 kit은 `InputData(channel_name='train', ...)` 하나만 넘기므로 컨테이너에 심기는 이름은 `SM_CHANNEL_TRAIN` 하나입니다.
평가 채널이 필요하면 `InputData`를 하나 더 넣고 `train.py`에서 그 이름의 env를 읽으면 됩니다.

인스턴스 박스 안에 **EBS Volume**이 함께 그려진 것이 [경로 규약](01_sagemaker_basics.md#경로-규약--컨테이너-안의-정해진-경로)의 물리적 근거입니다.
`/opt/ml/*`은 그 볼륨 위에 있고, 볼륨은 클러스터와 함께 사라집니다(`trainer.train()`이 부르는 `CreateTrainingJob`이 클러스터를 만들고, Job이 끝나면 회수합니다).
그림에서 그 볼륨 밖으로 나가는 화살표가 `SM_MODEL_DIR` 하나뿐이라는 점이 아래 두 규칙의 이유 전부입니다.

- **입력 경로 해석**: `--train_file`이 주어지면 그 파일을 쓰고, 없으면 `SM_CHANNEL_TRAIN`(기본 `/opt/ml/input/data/train`)의 첫 `.jsonl`을 사용합니다.
- **출력**: `--output_dir`의 기본값이 `SM_MODEL_DIR`(SageMaker AI가 `/opt/ml/model`로 세팅)이므로, 학습 산출물이 자동으로 S3 아티팩트가 됩니다.
- **`--dry_run`**: `epochs=1`, `max_seq_length<=512`, 데이터 32행으로 강제하고 중간 체크포인트 저장도 끕니다. **파이프라인(데이터 로드→토크나이즈→몇 step→저장)만** 검증하고 몇 분 안에 끝납니다. 실제 학습은 `--dry_run` 없이 실행하세요.
- **`--max_train_samples`**: 데이터 파일은 그대로 두고 앞 N건만 학습합니다. 노트북 기본값은 `MAX_TRAIN_SAMPLES=200`, `EPOCHS=2`로, 핸즈온에서 파이프라인이 끝까지 도는지 확인하는 용도입니다. 정식 학습은 `None`(전량)과 `EPOCHS=3~5`로 올려 따로 돌리세요.
- **의존성**: 무거운 import(`torch`/`trl`/`peft`)는 `main()` 안에서 지연 로드하므로, 인자 파싱 오류가 있으면 빠르게 죽습니다. 컨테이너 내부 패키지는 `scripts/requirements.txt`가 설치합니다.
- **학습 이미지**: 이 kit은 `.env`의 `DLC_IMAGE_URI`(리전 포함 완전 URI)를 그대로 씁니다. 기본값은 순수 PyTorch 학습 DLC(`pytorch-training`)이고, `requirements.txt`가 컨테이너 안에서 transformers/trl/peft를 최신으로 올립니다.

베이스에 transformers가 baked-in된 HF DLC(`huggingface-pytorch-training`)도 같은 방식으로 지정할 수 있습니다(AWS의 [SageMaker + HuggingFace 공식 가이드](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)).
자세한 URI 규칙은 [DLC 이미지 URI 패턴](04_sagemaker_inference.md#dlc-이미지-uri-패턴)을 참고하세요.

??? info "더 읽을 거리 — requirements.txt의 핀"
    `scripts/requirements.txt`가 설치하는 것은 `transformers>=5.14.1` / `trl>=1.8.0` / `peft>=0.19.1` / `datasets>=5.0.0` / `accelerate>=1.0.0` / `bitsandbytes>=0.44.0`입니다.
    floor는 상한이 없으므로 컨테이너가 더 최신 버전을 받을 수 있고, 그 조합은 **실행 전 재확인** 대상입니다(그 파일 주석에 적어 둔 최신 조합은 transformers 5.14.1 / trl 1.9.0 / peft 0.19.1입니다).

### 텍스트 전용 re-export와 KV-shared 복원

멀티모달 base로 텍스트 SFT를 한 뒤 **그냥 저장하면 서빙이 깨집니다.** `train.py`는 저장 단계에서 두 가지를 더 합니다.

1. **텍스트 arch로 re-export** — 머지된 멀티모달 모델에서 `language_model` 서브모듈만 골라 `text_config`(model_type `gemma4_text`) + `Gemma4ForCausalLM`(arch에 `Unified`가 있으면 `Gemma4UnifiedForCausalLM`)으로 다시 저장합니다. 멀티모달 config(`*ForConditionalGeneration`)가 남으면 vLLM이 image/audio processor를 찾다가 `Can't load image processor`로 죽습니다.
2. **KV-shared dead weight 복원** — gemma-4 E계열(`num_kv_shared_layers>0`)은 뒤쪽 레이어가 앞 레이어의 KV를 재사용하므로, transformers가 그 레이어에 `k_norm`/`k_proj`/`v_proj` 모듈을 **아예 만들지 않습니다**. `save_pretrained`를 거치면 원본에 있던 텐서가 소실됩니다. 반면 vLLM `Gemma4Attention`은 `k_norm`을 전 레이어에 등록하므로 `weights were not initialized ... k_norm`으로 엔진 초기화가 실패합니다. `_revive_kv_shared_from_base`가 base 체크포인트에서 그 텐서만 골라 읽어 되살립니다.

이 텐서는 forward에서 사용되지 않는 dead weight이고 LoRA 타깃에도 없으므로, base 값을 되살리는 것은 정확도에 무해합니다.
12B/26B-A4B는 `num_kv_shared_layers=0`이라 이 복원이 필요 없습니다. 서빙 쪽 관점은 [서빙 컨테이너 선택](05_serving_containers.md)에 정리돼 있습니다.

??? info "더 읽을 거리 — 두 단계의 실측 근거"
    **re-export** — 재키잉은 `model.language_model.*` → `model.*` + `lm_head`이고 실측 키 100% 매칭(E4B/12B/26B)입니다. 빈 뼈대를 `init_empty_weights`로 만들고 `load_state_dict(assign=True)`로 텐서를 이식해 사본을 만들지 않습니다(호스트 RAM 절약).

    **KV-shared 복원** — 소실 규모는 E4B 실측으로 42층 중 shared 18층 → 레이어 24~41 × 3개 = **정확히 54개**입니다. vLLM 초기화 실패는 [vLLM issue #44788](https://github.com/vllm-project/vllm/issues/44788)에 보고된 증상과 같습니다. 복원 전 665키(vLLM 실패) → 복원 후 719키(원본과 동일, vLLM 로드 성공)이고, 생성 결과도 transformers와 일치했습니다.

??? question "오개념 — “train.py에서 왜 common/config.py를 import 하지 않나요?”"
    SageMaker AI는 `source_dir`(=`scripts/`)만 컨테이너에 올리기 때문입니다. `common/`을 참조하면 클라우드에서 ImportError가 납니다.
    그래서 `train.py`는 **의도적으로 self-contained**하며, 설정은 노트북이 hyperparameters/environment로 주입합니다.

---

## LoRA vs QLoRA와 인스턴스 사이징

### LoRA vs QLoRA

| 축 | LoRA | QLoRA (`--use_qlora True`) |
|---|---|---|
| base 가중치 | bf16 그대로 | 4bit **nf4** 양자화(double-quant) |
| GPU 메모리 | 더 큼 | 더 작음(작은 단일 GPU에 적합) |
| 품질 | 기준 | 대개 근접(태스크 의존) |
| 속도 | 대개 빠름 | 양자화 오버헤드로 약간 느릴 수 있음 |
| 추가 의존성 | 없음 | `bitsandbytes>=0.44.0`(버전 재확인) |
| compute dtype | bf16 | `bnb_4bit_compute_dtype=bf16` |
| 언제 | GPU 메모리 여유 | 메모리가 빡빡할 때 |

`train.py`의 `--use_qlora` 기본값은 `False`지만, `common/config.py`의 gemma-4 프리셋은 전 사이즈 `use_qlora: True`를 권장값으로 들고 있고 노트북도 `'use_qlora': True`로 제출합니다.

### 인스턴스 사이징 — GPU와 호스트 RAM

| 프리셋(`MODEL_SIZE`) | 모델 ID | 규모 | 프리셋 학습 인스턴스 | 필요 transformers |
|---|---|---|---|---|
| `E2B` | `google/gemma-4-E2B-it` | effective 2.3B (on-disk 5.12B) | `ml.g5.2xlarge` | >= 5.5.0 |
| `E4B` (기본) | `google/gemma-4-E4B-it` | effective 4.5B (PLE 포함 ~8B) | `ml.g5.2xlarge` | >= 5.5.0 |
| `12B` | `google/gemma-4-12B-it` | 11.95B dense (unified arch) | `ml.g5.12xlarge` | >= 5.10.1 |
| `26B-A4B` | `google/gemma-4-26B-A4B-it` | MoE total 25.2B / active 3.8B | `ml.g5.12xlarge` | >= 5.5.0 |
| `31B` | `google/gemma-4-31B-it` | 31.27B dense | `ml.g6e.12xlarge` | >= 5.5.0 |

- 이 kit의 `.env`는 `TRAIN_INSTANCE_TYPE=ml.g6.2xlarge`로 프리셋을 덮어씁니다(`ml.g5.2xlarge`의 용량 대기가 길어서).
- `InsufficientInstanceCapacity`로 막히면 `AWS_REGION`이나 인스턴스 타입만 바꿔 재시도하세요.
- **GPU만 보지 말고 호스트 RAM도 보세요.** QLoRA 학습 자체는 GPU에 들어가지만, 학습 후 머지·re-export가 base를 bf16 full로 **CPU에 로드**하므로 RAM이 병목입니다(초기 버전은 여기서 OOM으로 죽었습니다).
- `train.py`는 머지 전에 학습 모델을 해제하고 base를 `low_cpu_mem_usage`로 로드해 사본을 최소화합니다. E4B peak RAM은 약 **17.5GB**로 실측됐습니다. `ml.g6.2xlarge`는 L4 24GB GPU + 32GB RAM이라 여유가 있고, 12B/26B는 머지 시 RAM이 더 커 `ml.g6.12xlarge` 급을 권장합니다.
- 정확한 GPU 메모리·인스턴스 스펙·리전 가용성은 **실행 전 SageMaker AI 인스턴스 문서에서 재확인**하세요.
- 비용을 줄이려면 SDK v3에서는 `Compute(enable_managed_spot_training=True)`를 쓰되, **`StoppingCondition(max_wait_time_in_seconds=...)`와 체크포인트 설정을 반드시 함께** 넘기세요. 빠뜨리면 `CreateTrainingJob`이 `ValidationException`으로 거부합니다.
- `max_wait_time_in_seconds`는 "Spot 용량 대기 시간 + 실행 시간"의 상한이고 **`max_runtime_in_seconds`보다 크거나 같아야** 합니다(SDK 3.16.0 `MaxWaitTimeInSeconds` 규약). 이 kit 코드에는 이 값이 설정돼 있지 않으므로(SDK 기본 `None`) spot을 켤 때 직접 추가해야 합니다. 절감 폭은 리전·수급에 따라 달라지므로 절대값으로 약속하지 마세요.

### merge_adapter — 서빙 단순화

- `--merge_adapter True`(기본값)이면 학습 후 LoRA 어댑터를 base에 **머지**해 **`output_dir` 루트**(=`/opt/ml/model`, `model.tar.gz` 루트)에 단일 모델로 저장합니다. 원본 어댑터는 `output_dir/adapter`에 따로 보관하고, 서빙 시 어댑터를 따로 로드할 필요가 없습니다.
- **서빙 루트에 완전한 HF 모델(config.json + 가중치)이 와야** vLLM/SGLang/DJL LMI가 `/opt/ml/model`에서 엔진을 감지합니다. 루트에 `adapter_config.json`만 있으면 엔진 감지 실패로 서버가 죽습니다. 그래서 머지 결과를 하위 디렉터리가 아니라 **루트에** 저장합니다.
- `--dry_run`에서는 시간 낭비를 막기 위해 머지를 건너뜁니다(어댑터만 루트에 저장).
- 어댑터만 저장하고 런타임에 얹는 방식도 가능합니다. 머지하면 저장·배포 단계에서 "작은 어댑터"의 이점은 사라집니다. 다만 LoRA의 본래 이점은 **학습 효율**(적은 파라미터 업데이트)에 있고 머지는 **서빙 편의**를 위한 별개 선택입니다. 여러 어댑터를 스왑해야 한다면 머지하지 말고 어댑터를 그대로 보관하세요.

??? question "오개념 — “QLoRA가 항상 낫지 않나요?”"
    그렇지 않습니다. QLoRA는 **메모리 절약** 기법이지 품질 향상 기법이 아닙니다.
    메모리가 충분하다면 LoRA가 더 단순하고 빠를 수 있습니다. 태스크·GPU 상황에 맞춰 조건부로 고르세요.

---

## MaxRuntimeExceeded — 학습 뒤 머지에서 잘리는 함정

**`ModelTrainer`에 `stopping_condition`을 넘기지 않으면 SDK가 1시간을 자동으로 넣습니다.**
이 한도는 학습 코드 시간만이 아니라 **용량 대기 + 이미지 pull + 학습 + 머지/업로드 전체**를 포함합니다. 그래서 "학습은 100% 끝났는데 머지 중에 죽어 배포 불가"가 됩니다.
근거는 [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk) `sagemaker/train/defaults.py`의 `DEFAULT_MAX_RUNTIME_IN_SECONDS = 3600`입니다(SDK 3.16.0 실측).

### 실측 — 1시간 한도에 걸린 학습 Job

학습 Job `gemma-summarization-train-20260731084146`, `ml.g6.2xlarge`에서 관측한 타임라인입니다.

| 단계 | 소요 | 누적 |
|---|---|---|
| Pending (GPU 용량 대기) | 6분 | 6분 |
| Downloading (DLC pull) | 3분 | 9분 |
| Training — 189/189 step 전부 완료 | 55분 | 64분 |
| ❌ 머지 도중 강제 종료 | — | 1시간 한도 도달 |

`train_runtime=3306s`로 학습은 정상 종료됐고 `Adapter saved` → `Merging LoRA adapter...` 로그까지 남았습니다.
그런데 산출물 `model.tar.gz`(542MB)에는 `adapter/`와 `checkpoint-*/`만 있고 **루트에 머지 모델이 없어**(정상 Job은 11.7GB) 서빙이 불가능했습니다.
[merge_adapter — 서빙 단순화](#merge_adapter--서빙-단순화)의 "루트에 완전한 HF 모델이 와야 한다"는 조건이 깨진 것입니다.

### 왜 진단이 어려운가

- `FailureReason`이 **비어 있습니다**. Job 상태는 `Failed`가 아니라 `Stopped`이고 `SecondaryStatus`만 `MaxRuntimeExceeded`이므로, `describe-training-job`까지 파고들어야 원인을 알 수 있습니다.
- CloudWatch 로그에는 **에러가 한 줄도 없습니다**(학습이 성공했으니까). 로그만 보면 정상 종료처럼 보입니다.
- SageMaker AI는 종료 시 `SIGTERM` 후 **120초**를 주고 그 사이의 산출물을 업로드합니다. 즉 **불완전한 아티팩트가 "생성"되므로** 파일이 있는데 못 쓰는 상태가 됩니다.

### 대응 — stopping_condition 명시

```python
from sagemaker.core.training.configs import StoppingCondition

MAX_RUNTIME_HOURS = 4   # SFT. GRPO는 rollout 때문에 6.
trainer = ModelTrainer(
    ...,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_HOURS * 3600),
)
```

- **넉넉히 잡아도 손해가 없습니다**: Job이 정상 종료되면 그 시점에 과금이 멈춥니다. 이 값은 요금이 아니라 **폭주 방지 상한**입니다.
- 반대로 실습 비용을 확실히 막으려 낮출 때는 **머지·업로드용으로 최소 15분**을 남기세요(E4B 실측: 머지 약 2분 + 업로드 약 3분. 모델이 커지면 늘어납니다).
- 노트북은 제출 전에 **예상 시간을 계산해 한도와 비교하고, 초과하면 `assert`로 막습니다**. GPU·라이브러리 버전이 달라지면 기준 s/step이 움직여 `assert`가 실제와 어긋나므로, Job 로그의 실제 step 시간으로 갱신하세요.

??? info "더 읽을 거리 — API 한도와 assert 기준값"
    [StoppingCondition API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에 따르면 `MaxRuntimeInSeconds`는 API 최대 28일·API 기본 1일이고, 종료 시 `SIGTERM` 후 120초의 유예를 줍니다. 1시간은 SDK 쪽 기본값입니다.

    노트북 `assert`의 기준값은 `ml.g6.2xlarge` QLoRA에서 seq 2048 약 17s/step, seq 512 약 7s/step으로 실측됐고, step 수는 `ceil(건수/8) × epochs`입니다.

### 함께 고친 것 — 체크포인트 누적

`/opt/ml/model` **전체가** `model.tar.gz`로 올라가므로, epoch마다 쌓인 체크포인트가 업로드 시간을 늘립니다(= 한도를 잡아먹습니다).
서빙은 머지된 루트만 읽으므로 `SFTConfig(save_total_limit=1)`로 1개만 남깁니다(체크포인트 3개 = 0.7GB, 전부 서빙에 불필요).

??? question "오개념 — “핸즈온인데 MAX_RUNTIME_HOURS=4면 4시간 과금되나요?”"
    아닙니다. 한도는 **강제 종료 시점**일 뿐이고 실제 과금은 Job이 실행된 시간만큼입니다.
    노트북 기본값(`MAX_TRAIN_SAMPLES=200`, `EPOCHS=2`)이면 20~25분 안에 끝납니다(`ml.g6.2xlarge` 실측).

---

## 모델 접근과 라이선스 전파

### gemma-4 기본과 gemma-3 옵션

| | **gemma-4 계열** (이 kit 기본) | gemma-3 계열 (옵션) |
|---|---|---|
| 예 ([HF `google` 조직](https://huggingface.co/google)) | `google/gemma-4-E4B-it` / `-12B-it` / `-26B-A4B-it` | `google/gemma-3-4b-it` 등 |
| 라이선스 | **apache-2.0** | 커스텀 **Gemma** 라이선스(use-restriction 포함) |
| 접근 | UNGATED — 토큰 불필요 | GATED — HF 토큰 + 약관 수락 필요 |
| HF_TOKEN | 불필요(비워도 됨) | 필요(`MODEL_IS_GATED=1` + env/`hf auth login`) |
| 전환 방법 | `MODEL_SIZE`(E2B/E4B/12B/26B-A4B/31B) | `MODEL_ID`로 직접 지정 |

- **env에 시크릿을 넣지 않는 쪽을 권장합니다**: `hf auth login`만 해도 kit이 토큰을 인식합니다. 조회 순서는 (1) env `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` → (2) `hf auth login`이 저장한 파일 토큰(`$HF_HOME/token`)입니다(`common/config.py`의 `get_hf_token()`). ungated 모델에는 토큰을 넣지 않아도 되고, 넣어도 무해합니다.
- 서빙 쪽은 다릅니다. 이 kit의 endpoint는 학습 산출 모델(S3 `model_data`)을 서빙하므로 HF에서 가중치를 당기지 않습니다. 토큰을 서빙 env에 실으면 `describe_endpoint` 같은 리소스 메타데이터에 평문으로 남으므로, `get_serving_hf_token()`은 **`MODEL_IS_GATED=1`일 때만** 토큰을 반환합니다.
- **라이선스 전파**: Gemma 라이선스(gemma-3 등)는 파인튜닝은 물론 **머지 산출물·서빙 결과물까지** use-restriction이 전파됩니다. apache-2.0(gemma-4)에는 그런 제약이 없습니다.
- gated·ungated 여부와 라이선스는 바뀔 수 있으므로, 재배포·서빙 전에 쓰려는 모델의 카드에서 라이선스 배너를 다시 확인하세요. 옵션 경로라면 [`gemma-3-4b-it` 모델 카드](https://huggingface.co/google/gemma-3-4b-it), 기본 경로라면 [`gemma-4-E4B-it` 모델 카드](https://huggingface.co/google/gemma-4-E4B-it)입니다.

시드 데이터셋은 전부 permissive한 것으로 골랐습니다. 세부는 `common/config.py`의 `TRACKS`를 참조하세요.

| 시드 데이터셋 | 라이선스 |
|---|---|
| glaive-function-calling-v2 | apache-2.0 |
| `mteb/banking77` (parquet 미러) | mit |
| billsum | cc0-1.0 |
| databricks-dolly-15k | cc-by-sa-3.0 |
| cord-v2 | cc-by-4.0 |

`PolyAI/banking77` 원본은 스크립트 기반이라 `datasets>=5.0.0`에서 로드되지 않아 parquet 미러로 바꿨습니다.

??? question "오개념 — “gemma-4가 ungated니까 큰 걸 쓰면 되지 않나요?”"
    라이선스 관점만 보면 편한 것은 맞습니다. 하지만 크기·비용·품질 요구가 서로 다릅니다.
    기본은 단일 GPU 친화적인 `E4B`(effective 4.5B)이고, 용량이 더 필요할 때 `12B`나 `26B-A4B`로 조건부 승급하세요. 12B는 `transformers>=5.10.0`이 필요하고 머지 단계의 호스트 RAM 요구도 커집니다.

---

## SFT에서 GRPO로 — 데이터를 갈아야 하는 이유

!!! abstract "쉽게 말하면"
    SFT는 (입력, 정답) 쌍으로 정답을 모방하고, GRPO는 **prompt만** 받아 스스로 생성한 뒤 reward로 채점합니다.
    같은 데이터를 쓰면 누출이고, 더 나쁘게는 **학습이 아예 안 됩니다.** `02a_train_grpo_sagemaker`를 실행하기 전에 [advantage ≈ 0 문제](#왜-학습이-안-되는가--advantage--0)와 [RL prompt 소스 3가지](#rl-prompt-소스-3가지)를 읽으세요.

### 왜 학습이 안 되는가 — advantage ≈ 0

GRPO는 prompt 하나당 `num_generations`개(이 kit 기본 8)를 생성해 **그룹 안에서 상대 비교**로 학습합니다.

```
prompt → rollout 8개 → 각각 reward 채점 → 그룹 평균 대비 편차(advantage)로 gradient
```

SFT가 이미 잘 맞히는 prompt를 주면 **8개가 모두 만점**이 됩니다. 그룹 내 편차가 없으니 advantage가 0에 수렴하고 **gradient가 거의 흐르지 않습니다.** GPU 시간만 쓰고 배우는 것이 없습니다.

그래서 **슬라이스만 분리해도 부족합니다.** 같은 분포에서 잘라낸 다른 100건은 누출은 막지만, 모델이 여전히 잘 맞히므로 advantage 문제는 그대로 남습니다.

### SFT vs GRPO 데이터

| | SFT | GRPO / RL |
|---|---|---|
| 목적 | 형식·기본 능력 습득 | SFT가 실패하는 구간 개선 |
| 필요한 데이터 | (입력, **정답**) 쌍 | **prompt** (+ 프로그램적 채점용 reference) |
| 정답의 역할 | 학습 입력으로 직접 사용 | **reward 계산에만** — 모델에 보여주지 않음 |
| 좋은 prompt | 대표적인 것 | 모델이 어려워하는 것 |
| 연산량 | 1 forward/backward | prompt당 rollout N개 → 수 배 |

`train_grpo.py`의 변환을 보면 정답이 학습 입력이 아님이 드러납니다.

```python
def _to_grpo(example):
    prompt    = [m for m in msgs if m["role"] != "assistant"]   # ← 모델에 들어가는 것
    reference = ...assistant content...                          # ← reward 계산용만
```

### RL prompt 소스 3가지

이 kit은 `common/grpo_data.py`로 세 가지를 제공하고, 노트북에서 `GRPO_PROMPT_SOURCE`로 고릅니다. 기본값은 `synth`입니다.

| 소스 | 무엇 | 비용·선행조건 | advantage 확보 |
|---|---|---|---|
| `synth` (기본) | Bedrock으로 prompt만 생성 + 난이도 제약 | Bedrock 과금(소액) | 확보 |
| `failures` | `04_evaluate`에서 틀린 건만 | 03·04 선행 필요 | 가장 강함 |
| `holdout` | SFT가 쓰지 않은 구간 | 무료·즉시 | 약함(같은 분포) |

- **`synth`가 기본인 이유**: `holdout`은 무료지만 같은 분포라 advantage가 잘 생기지 않습니다. `synth`는 생성 프롬프트에 **난이도 제약**을 걸어 어려운 예시를 만듭니다. 제약은 **생성 프롬프트에만** 넣습니다 — critique에도 넣으면 시드와 다르다며 전부 기각합니다(실측 8/8 기각).
- **`synth`의 함정**: SFT 합성과 **같은 시드**를 주면 분포가 또 겹칩니다. 노트북은 `NUM_SEED_SAMPLES` 이후 구간의 시드만 넘겨 이를 피합니다. RL은 정답이 학습 입력이 아니므로 prompt 생성이 SFT 합성보다 쉽고 쌉니다. 다만 이 kit의 reward는 프로그램적 채점이라 reference가 필요해 (input, output) 형태로 만듭니다.
- **`failures`**: 실무에서 가장 효율적인 경로입니다. reward 신호가 강한 구간에만 집중합니다. 실패가 0건이면 GRPO로 얻을 것이 적다는 뜻이므로(좋은 신호) `N_EVAL`을 키워 더 어려운 케이스를 찾으세요.
- **`holdout`**: 추가 비용 없이 파이프라인을 끝까지 볼 수 있게 하는 값입니다. 누출은 막지만, 학습 후 reward가 거의 변하지 않으면 [advantage ≈ 0 문제](#왜-학습이-안-되는가--advantage--0)로 보고 다른 소스로 옮기세요.

??? info "더 읽을 거리 — 난이도 제약의 실측 효과"
    추출 코스 실측에서는 제약 없이 합성하면 8건 전부 인자 0개였습니다(시드 분포가 인자 없는 함수 94%).
    제약을 걸면 **인자 없음 0건 / 평균 인자 2.1개**가 되고, 값을 간접 표현("the day after tomorrow")하는 입력이 나옵니다.

프로덕션에서는 하나가 더 있습니다: **실제 트래픽 로그**. 분포가 진짜라서 가장 가치 있지만 공개 데이터로 재현할 수 없어 이 kit에는 넣지 않았습니다.
held-out 분리 규율은 [held-out 규율](02_synthetic_data.md#held-out-규율--합성으로-평가-금지)을 참고하세요.

### 왜 추출·분류 코스에만 GRPO가 있나

GRPO에는 **프로그램적으로 채점 가능한 reward**가 필요합니다. `train_grpo.py`의 `--reward_kind`가 받는 값도 `extraction`과 `classification` 둘뿐입니다.

| 코스 | 프로그램적 채점 | GRPO 노트북 |
|---|---|---|
| 추출(JSON) | ✅ 가능 — JSON 유효성 + 함수명/인자 F1 | ✅ 제공 (`02a_train_grpo_sagemaker`) |
| 분류 | ✅ 가능 — 라벨 정확 일치 | ✅ 제공 (`02a_train_grpo_sagemaker`) |
| 요약 · 도메인 QA | ❌ 불가 — "좋은 요약"을 규칙으로 채점 불가 | ❌ 없음 |

요약·QA는 LLM-judge를 reward로 쓸 수는 있지만, rollout마다 judge를 호출해야 해 비용·시간이 급증하고 judge 편향이 학습에 섞이므로 이 kit에서는 제외했습니다.

??? question "오개념 — “SFT 없이 base에서 바로 GRPO 하면 안 되나요?”"
    됩니다. `train_grpo.py`는 `model` 채널이 없으면 HF base로 폴백합니다.
    다만 형식조차 안 잡힌 상태에서는 rollout이 전부 낮은 점수라 역시 편차가 작고 수렴이 불안정합니다. **SFT → GRPO**가 정석인 이유입니다.

---

## 자주 나오는 오개념

아래 항목들은 앞 절에서 다루지 않은, 학습 방식 자체와 컨테이너 계층을 헷갈릴 때 생기는 착각입니다.

??? question "오개념 — “파인튜닝은 전체 가중치를 학습하는 것 아닌가요?”"
    이 kit은 그렇지 않습니다. **PEFT LoRA/QLoRA**로 어댑터만 학습하며 full fine-tune이 아닙니다.
    그래서 단일 GPU에서도 SLM을 돌릴 수 있습니다. 다만 텍스트 전용 base에서는 `modules_to_save`로 `lm_head`/`embed_tokens`를 full-train 대상에 포함시킵니다.

환경 차이를 과소평가하는 착각도 같은 유형입니다.

??? question "오개념 — “로컬에서 됐으니 SageMaker AI에서도 그대로 되겠지”"
    같은 `train.py`를 쓰므로 대체로 맞습니다. 다만 세 가지가 다릅니다.
    (1) boolean 하이퍼는 `--key value`로 들어옵니다([str2bool](#boolean-하이퍼파라미터--str2bool)), (2) 데이터는 `SM_CHANNEL_TRAIN`으로 들어옵니다, (3) 컨테이너의 `transformers` 버전은 로컬과 다를 수 있습니다(`requirements.txt`가 조정).
    이 세 가지는 dry-run으로 미리 Job을 수 있습니다.

컨테이너 이미지를 고를 때 나오는 혼동은 다음과 같습니다.

??? question "오개념 — “DLC 이미지 태그는 최신으로 아무거나 넣으면 되지 않나요?”"
    그렇지 않습니다. DLC 태그는 **AWS가 게시한 조합만** 유효합니다.
    `common/dlc.py`는 `DLC_IMAGE_URI`(완전 URI) → `DLC_REPOSITORY`+`DLC_TAG` → SDK `image_uris.retrieve` 순으로 해석합니다. ECR 계정은 `763104351884`(대부분의 리전 공용), 패턴은 `763104351884.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`이며 현행 태그는 [DLC available images](https://aws.github.io/deep-learning-containers/reference/available_images/) 페이지에서 확인하세요. 태그는 자주 갱신되므로 **실행 직전에 다시 확인**해야 합니다.
    학습 이미지는 **리전별 private ECR만** 허용됩니다. `public.ecr.aws/...` URI를 주면 실패하고, 리전을 옮길 때는 `AWS_REGION`과 URI의 리전을 함께 바꿔야 합니다.

이미지 종류를 혼동하는 경우도 흔합니다.

??? question "오개념 — “DLC(컨테이너)와 DLAMI(AMI)는 같은 것 아닌가요?”"
    다릅니다. DLC는 **워크로드 컨테이너 이미지**(학습/추론)이고 DLAMI는 **노드 호스트 이미지(AMI)** 입니다.
    이 kit은 SageMaker AI 학습 컨테이너로 DLC를 씁니다. 이미지에 무엇이 들어 있는지는 [AWS Deep Learning Containers 저장소](https://github.com/aws/deep-learning-containers)의 Dockerfile로 직접 확인할 수 있습니다.

마지막은 단계 경계에 대한 착각입니다.

??? question "오개념 — “학습이 곧 배포 아닌가요?”"
    아닙니다. `02_train_sft_sagemaker`(학습)와 `03_deploy_endpoint`(추론 endpoint)는 별개의 단계입니다.
    SageMaker AI 추론에는 Real-time / Serverless / Asynchronous / Batch Transform 네 옵션이 있고 **Serverless는 GPU가 없어 LLM에 부적합**합니다. 이 kit은 real-time endpoint를 씁니다. 자세히는 [왜 Real-time인가](04_sagemaker_inference.md#왜-real-time인가--추론-4옵션-비교)를 보세요.

---

## 비용과 cleanup

!!! danger "비용과 cleanup"
    학습 Job은 **실행 시간만큼** 과금되고, GPU 인스턴스는 저렴하지 않습니다. 반드시 로컬 `--dry_run`으로 파이프라인을 먼저 검증한 뒤 실제 학습을 제출하세요.
    학습 Job은 끝나면 인스턴스가 자동 해제되지만 **endpoint는 켜 두면 계속 과금**됩니다. 코스를 마쳤으면 `99_cleanup` 노트북으로 endpoint와 아티팩트를 정리하세요.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker AI 학습 Job (`ml.g6.2xlarge` 등) | 인스턴스 시간당 과금, Job 종료 시 자동 중지 | 자동 종료. `stopping_condition`으로 상한 고정 |
| SageMaker AI endpoint (`03_deploy_endpoint`) | 인스턴스 시간당 **상시** 과금 | `99_cleanup.ipynb`로 반드시 삭제 |
| S3 아티팩트 (`model.tar.gz`, 체크포인트) | 저장 용량당 누적 과금 | `99_cleanup.ipynb`. `save_total_limit=1`로 크기 억제 |
| CloudWatch 학습 로그 | 저장 용량당 누적 과금 | 로그 그룹 보존 기간 설정 |
| Bedrock (GRPO `synth` prompt 생성) | 입력·출력 토큰당 과금 | 상시 리소스 없음. `N_GRPO`로 총량 제어 |

managed spot(`Compute(enable_managed_spot_training=True)`)을 쓰면 학습 비용을 줄일 수 있습니다.
단 **`StoppingCondition(max_wait_time_in_seconds=...)`(>= `max_runtime_in_seconds`)와 체크포인트 설정이 필수**입니다. 빠뜨리면 Job 생성이 `ValidationException`으로 거부됩니다([인스턴스 사이징](#인스턴스-사이징--gpu와-호스트-ram) 참고).
절감 수치는 리전·수급에 따라 다르므로 절대값으로 약속하지 마세요.
