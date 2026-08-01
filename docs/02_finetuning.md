# 02. Fine-tuning 접근법 — HuggingFace DLC + TRL SFTTrainer + PEFT LoRA/QLoRA

> **대상 독자**: SageMaker에서 Gemma를 처음 파인튜닝해 보는 엔지니어입니다(HuggingFace `transformers`/`trl`은 대략 알지만 SageMaker 학습 잡·DLC·LoRA 관용구는 처음인 분).
> **⚠️ 주의**: 🔴 표시(모델 ID·DLC 이미지 태그·SDK 버전·리전·GA 상태)는 **빠르게 바뀝니다**. 아래 값에는 근거를 달아 두었지만 **실행 직전 원문에서 재확인**하시기 바랍니다. 계정 ID·role ARN·토큰·버킷·절대경로는 이 문서에 **하드코딩하지 않으며**, 전부 env/노트북으로 주입합니다.
> **라이브 검증 2026-07** (하단 소스 표 참조). 세부 태그/버전은 재확인 대상입니다.
> **관련 파일**: `common/config.py`, `common/dlc.py`, `tracks/*/scripts/train.py`, `tracks/*/scripts/requirements.txt`, 노트북 `01_data_and_synthetic` → `02_train_sft_sagemaker`.

---

## §0 TL;DR

**한 줄**: 이 킷은 SageMaker 파인튜닝 두 경로(JumpStart vs 자체 스크립트) 중 **HuggingFace DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA를 담은 self-contained `scripts/train.py`**를 택했습니다 — 커스텀 학습 로직과 최신 Gemma를 즉시 쓰기 위해서입니다.

1. **왜 JumpStart가 아니라 HF DLC인가** — JumpStart는 "정해진 모델을 정해진 레시피로" 빠르게 돌리는 데 최적이지만, 최신 Gemma 릴리스 반영과 커스텀 SFT 로직(chat template fold, `modules_to_save`, packing 안전장치 등)의 제어가 어렵습니다. 이 킷은 커스텀 제어가 필요하기 때문에 HF DLC를 선택했습니다.
2. **`train.py`는 로컬 GPU `--dry_run` == SageMaker `.fit()`** — 동일한 파일이 두 환경에서 그대로 돕니다. 로컬에서 파이프라인을 검증한 뒤 그대로 클라우드로 올리면 됩니다.
3. **Gemma 관용구 6종은 협상 불가입니다**: `apply_chat_template`(system role fold), `target_modules="all-linear"` + `modules_to_save=["lm_head","embed_tokens"]`, **bf16**(fp16 금지), `attn_implementation="eager"`(안전 기본), **packing은 flash-attention 아니면 off**, boolean 하이퍼는 `str2bool`로 처리합니다.
4. **LoRA vs QLoRA는 GPU 메모리로 결정합니다** — 작은 단일 GPU라면 QLoRA(4bit nf4)를, 여유가 있으면 LoRA를 쓰세요. 서빙 단순화가 목표라면 학습 후 `merge_adapter`를 사용합니다.
5. **gated vs ungated**: 🔴 `gemma-3` 계열은 gated(HF_TOKEN + 약관 수락 필요)이고, `gemma-4`(apache-2.0)는 ungated입니다. 라이선스는 **머지·서빙 결과물까지 전파**된다는 점에 유의하세요.

---

## §0.5 기존 Pain Point

처음 SageMaker에서 Gemma를 파인튜닝하려 할 때 실제로 부딪히게 되는 벽들은 다음과 같습니다:

- **"JumpStart 버튼 하나면 되는 거 아니야?"** → 막상 돌려보면 최신 Gemma가 목록에 없거나, chat template·특수토큰 처리 같은 세부를 건드릴 수 없어 결과가 이상하게 나옵니다.
- **로컬에서 잘 되던 학습 스크립트가 SageMaker에선 `--use_qlora`에서 크래시**합니다 → `store_true` 플래그에 `--use_qlora True`가 들어오면서 죽는 것인데, 원인을 모른 채 몇 시간을 날리기 쉽습니다.
- **fp16으로 돌렸더니 loss가 NaN**이 됩니다 → Gemma에서 흔히 겪는 함정입니다.
- **학습은 됐는데 출력이 엉망**입니다 → chat template을 손으로 조립했거나, system role을 그대로 넣어 토크나이저가 거부했거나, packing으로 샘플이 서로 오염된 경우입니다.
- **모델을 못 받습니다(403)** → gated 모델인데 토큰/약관 수락이 없는 경우입니다. 반대로 ungated인데 토큰을 억지로 넣어 헷갈리는 경우도 있습니다.

이 문서에서는 위 함정들을 `train.py`가 어떻게 **미리 막아 두었는지** 설명합니다.

---

## §1 왜 HF DLC 경로인가? (JumpStart와 대조)

**쉽게 말하면**: SageMaker에서 파인튜닝하는 문서화된 길은 두 갈래입니다. 하나는 **JumpStart**로 미리 포장된 모델·레시피를 SDK 몇 줄로 돌리는 방식이고, 다른 하나는 **직접 학습 스크립트 + 컨테이너**를 써서 HuggingFace estimator에 `entry_point`를 넘기는 방식입니다. 전자는 "메뉴 주문", 후자는 "직접 요리"에 비유할 수 있습니다.

### 대조표 — SageMaker 파인튜닝 두 경로

| 축 | JumpStart 파인튜닝 | **HF DLC + 자체 `train.py`** (이 킷) |
|---|---|---|
| 인터페이스 | `JumpStartEstimator` / 콘솔 | `sagemaker.huggingface.HuggingFace` estimator |
| 모델 커버리지 | 큐레이션된 목록(신규 릴리스 지연 가능) | 🔴 HF Hub의 **최신 Gemma 즉시** (`MODEL_ID`만 교체) |
| 학습 로직 제어 | 제한적(정해진 레시피) | **완전 제어**(chat template fold, `modules_to_save`, packing 안전장치, merge) |
| 커스텀 의존성 | 어려움 | `scripts/requirements.txt`로 자유롭게(🔴 `transformers>=5.14.1` 등) |
| 로컬 == 클라우드 | 아니오 | **예** — 동일 `train.py`가 `--dry_run`/`.fit()` 겸용 |
| 진입 난이도 | 낮음 | 중간(대신 투명·이식성) |
| 언제 고르나 | 표준 레시피로 충분·빠른 baseline | **커스텀 로직 or 최신 모델이 필요할 때** |

> **비유**: JumpStart가 밀키트라면, HF DLC는 장을 봐서 직접 요리하는 방식입니다. 이 킷은 레시피(Gemma 관용구)를 정확히 통제해야 하므로 후자를 골랐습니다.

### 기술적 차이 3가지

1. **최신 모델 반영** — 🔴 Gemma는 분기마다 릴리스가 갱신됩니다. HF DLC 경로에서는 `MODEL_ID` env 하나만 바꾸면 `gemma-3-4b-it` → `gemma-4-12B-it` 승급이 끝납니다(`common/config.py`의 `DEFAULT_MODEL_ID`). 반면 JumpStart는 큐레이션 목록에 오를 때까지 기다려야 할 수 있습니다.
2. **학습 로직 투명성** — Gemma는 손대야 할 관용구가 많습니다(§2). `train.py`는 이 모든 결정을 코드로 명시적으로 담고 있어, 리뷰·재현·이식이 쉽습니다.
3. **이식성(로컬↔SageMaker 단일 소스)** — SageMaker는 `source_dir`만 컨테이너에 올립니다. 그래서 `train.py`는 `common/`에 **의존하지 않는 self-contained** 파일로 작성했습니다. 로컬 GPU에서 `--dry_run`으로 파이프라인을 검증한 **바로 그 파일**이 클라우드에서 그대로 돕니다.

> ❓ **"JumpStart가 더 production-ready 아닌가?"** — production 여부는 경로가 아니라 운영(체크포인트·모니터링·재현성)이 결정합니다. HF DLC 경로도 spot·checkpoint·CloudWatch로 충분히 production-ready합니다. 선택 기준은 성숙도가 아니라 "레시피 제어가 필요한가"입니다.

---

## §2 Gemma 파인튜닝 관용구 (여기가 핵심)

**쉽게 말하면**: Gemma는 "그냥 돌리면" 미묘하게 틀립니다. 아래 6가지는 Gemma 모델 카드/TRL 문서에 근거한 관용구이며, `train.py`에 이미 반영되어 있습니다.

```
  데이터(JSONL)                train.py 처리                    결과
  {"messages":[...]}  ─► apply_chat_template (SFTTrainer 자동)  ─► 올바른 -it 포맷
       (system?)      ─► system 거부 → 첫 user턴에 fold
                      ─► LoRA all-linear + save[lm_head,embed]  ─► 특수토큰 학습됨
                      ─► bf16 + eager + packing(조건부 off)      ─► NaN·오염 없음
```

### 2.1 chat template — `apply_chat_template`, system role는 fold

- Gemma **-it(instruction-tuned)** 토크나이저에는 chat template이 **내장**되어 있습니다. 데이터가 conversational 포맷(`{"messages":[{"role","content"},...]}`)이면 TRL `SFTTrainer`가 **자동으로 `apply_chat_template`을 적용**합니다. `<start_of_turn>` 같은 마커를 **손으로 조립하지 마세요**.
- 🔴 **Gemma chat template은 `system` role을 거부**합니다(별도 system 슬롯이 없습니다). system 지시가 필요하면 **첫 `user` 턴 맨 앞에 접어 넣으세요(fold)**. 데이터 준비 단계(`01_data_and_synthetic`)에서 미리 이 형태로 만들어 두시는 것이 안전합니다.

> ❓ **"system 프롬프트를 messages에 그냥 넣으면 되지 않나?"** — Gemma에 `{"role":"system",...}`을 넣으면 template 적용 시 에러가 납니다. 반드시 첫 user 메시지로 fold하거나 데이터셋에서 미리 병합하세요.

### 2.2 LoRA target — `all-linear` + `modules_to_save`

```python
LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],   # 특수토큰 학습
)
```

- `target_modules="all-linear"`는 모든 linear 레이어에 어댑터를 붙입니다(Gemma 권장 설정).
- `modules_to_save=["lm_head","embed_tokens"]` — LoRA는 원래 임베딩/출력 헤드를 건드리지 않지만, chat 특수토큰을 제대로 학습시키려면 이 둘을 **full-train 대상으로 저장**해야 합니다. 빠뜨리면 특수토큰 표현이 어긋납니다.

### 2.3 bf16 필수, fp16 금지

- `SFTConfig(bf16=True)`로 설정합니다. 🔴 **Gemma에서 fp16은 오버플로/NaN**을 일으킵니다. `train.py`는 모델을 `torch_dtype=torch.bfloat16`으로 로드하고 `bf16=True`로 학습합니다. bf16을 지원하지 않는 GPU라면 QLoRA의 `bnb_4bit_compute_dtype=torch.bfloat16` 경로를 쓰되, 하드웨어 호환성을 반드시 확인하세요.

### 2.4 attention — `eager`가 안전 기본

- `attn_implementation="eager"`가 Gemma의 **안전 기본값**입니다(soft-cap / sliding-window 정합성 때문). `sdpa`·`flash_attention_2`도 선택할 수 있지만, 특별한 이유가 없다면 eager로 시작하시기를 권장합니다.

### 2.5 packing — flash-attention 아니면 OFF

- packing은 여러 짧은 샘플을 한 시퀀스로 이어 붙여 throughput을 올려 줍니다. 하지만 attention이 샘플 경계를 마스킹하지 못하면 **샘플끼리 서로 참조하는 cross-contamination(교차 오염)**이 발생합니다.
- `train.py`는 **`flash_attention_2`/`flash_attention_3`일 때만 packing을 켭니다.** eager/sdpa에서는 `--packing True`를 주더라도 **자동으로 끄고**(경고를 출력) 넘어갑니다. 즉 안전 기본값인 eager에서는 사실상 off입니다.

> ❓ **"packing을 켰는데 왜 안 켜졌지?"** — 버그가 아닙니다. eager/sdpa에서 오염을 방지하기 위해 코드가 의도적으로 비활성화한 것입니다. packing throughput이 필요하다면 flash-attention을 명시적으로 선택하세요.

### 2.6 boolean 하이퍼파라미터 — `str2bool`

- 🔴 SageMaker HuggingFace estimator는 **모든 하이퍼파라미터를 `--key value`로 직렬화**합니다. 즉 `use_qlora=True`가 `--use_qlora True`로 전달됩니다. 흔히 쓰는 `action="store_true"`는 값을 받지 않으므로 이때 **크래시**합니다.
- 해결책은 `type=_str2bool, nargs="?", const=True`입니다. 이렇게 하면 로컬의 bare-flag(`--dry_run`)와 SageMaker의 `--use_qlora True`를 **양쪽 모두** 받을 수 있습니다.

```python
def _str2bool(v) -> bool:
    return str(v).strip().lower() in ("1","true","yes","y","t")
```

---

## §3 `train.py` — 로컬 dry-run == SageMaker `.fit()`

**쉽게 말하면**: 파일 하나가 두 무대에서 똑같이 공연하는 셈입니다. 리허설(로컬 GPU 소량)과 본공연(SageMaker)이 같은 대본을 사용합니다.

```
로컬 개발 GPU                          SageMaker 학습 잡
─────────────                          ─────────────────
python train.py --dry_run              estimator.fit({'train': s3})
  --train_file ./sample.jsonl              │  source_dir='scripts' 업로드
  --output_dir ./out                       │  (train.py + requirements.txt)
       │                                    ▼
       └── epochs=1, seq<=512, 32행     hyperparameters → --key value
           파이프라인만 검증                  SM_CHANNEL_TRAIN / SM_MODEL_DIR
```

- **입력 경로 해석**: `--train_file`이 주어지면 그 파일을 쓰고, 없으면 `SM_CHANNEL_TRAIN`(기본 `/opt/ml/input/data/train`)의 첫 `.jsonl`을 사용합니다.
- **출력**: `--output_dir`의 기본값이 `SM_MODEL_DIR`(SageMaker가 `/opt/ml/model`로 세팅)이므로, 학습 산출물이 자동으로 S3 아티팩트가 됩니다.
- **`--dry_run`**: `epochs=1`, `max_seq_length<=512`, 데이터 최대 32행으로 강제합니다. **파이프라인(데이터 로드→토크나이즈→몇 step→저장)만** 검증하고 몇 분 안에 끝납니다. 실제 학습은 `--dry_run` 없이 실행하세요.
- **의존성**: 무거운 import(`torch`/`trl`/`peft`)는 `main()` 안에서 지연 로드하므로, 인자 파싱 오류가 있으면 빠르게 죽습니다. 컨테이너 내부의 최신 패키지는 `scripts/requirements.txt`가 설치합니다(🔴 `transformers>=5.14.1 / trl>=1.8.0 / peft>=0.19.1`, 실행 전 재확인).

> ❓ **"`train.py`에서 왜 `common/config.py`를 import 안 하지?"** — SageMaker는 `source_dir`(=`scripts/`)만 컨테이너에 올리기 때문입니다. `common/`을 참조하면 클라우드에서 ImportError가 납니다. 그래서 `train.py`는 **의도적으로 self-contained**하게 작성했습니다. 설정은 노트북이 hyperparameters/env로 주입합니다.

---

## §4 LoRA vs QLoRA · 인스턴스 사이징 · merge

### 대조표 — LoRA vs QLoRA

| 축 | LoRA | QLoRA (`--use_qlora True`) |
|---|---|---|
| base 가중치 | bf16 그대로 | 4bit **nf4** 양자화(double-quant) |
| GPU 메모리 | 더 큼 | **더 작음**(작은 단일 GPU에 적합) |
| 품질 | 기준 | 대개 근접(태스크 의존) |
| 속도 | 대개 빠름 | 양자화 오버헤드로 약간 느릴 수 있음 |
| 추가 의존성 | 없음 | `bitsandbytes`(🔴 버전 재확인) |
| compute dtype | bf16 | `bnb_4bit_compute_dtype=bf16` |
| 언제 | GPU 메모리 여유 | **메모리가 빡빡할 때** |

> ❓ **"QLoRA가 항상 낫다?"** — 그렇지 않습니다. QLoRA는 **메모리 절약** 기법이지 품질 향상 기법이 아닙니다. 메모리가 충분하다면 오히려 LoRA가 더 단순하고 빠를 수 있습니다. 태스크·GPU 상황에 맞춰 조건부로 선택하세요.

### 인스턴스 사이징 (조건부)

- 🔴 이 킷의 기본값은 `ml.g5.2xlarge`(단일 GPU)로, SLM(gemma-3-4b)의 LoRA/QLoRA에 합리적인 선택입니다(`common/config.py`의 `TRAIN_INSTANCE_TYPE`). 다만 이는 **절대적인 값이 아닙니다.**
- 더 큰 모델(`gemma-4-12B`)이나 긴 시퀀스, 비양자화 LoRA를 쓴다면 메모리가 더 큰 인스턴스로 올리거나 QLoRA로 내려야 합니다.
- 정확한 GPU 메모리·인스턴스 스펙·리전 가용성은 **실행 전 SageMaker 인스턴스 문서에서 재확인**하세요(🔴 fast-changing).
- 비용 절감을 위해서는 `use_spot_instances=True`(+ `max_wait`, checkpoint 필요)를 노트북에서 조건부로 활성화하시면 됩니다.

### merge_adapter — 서빙 단순화

- `--merge_adapter True`(기본값)이면 학습 후 LoRA 어댑터를 base에 **머지**해 **`output_dir` 루트**(=`/opt/ml/model`, tar.gz 루트)에 단일 모델로 저장하고, 원본 어댑터는 `output_dir/adapter`에 따로 보관합니다. 이렇게 하면 서빙 시 어댑터를 따로 로드할 필요가 없습니다.
  - 🔴 **서빙 루트에 완전한 HF 모델(config.json+가중치)이 와야** DJL LMI/vLLM이 `HF_MODEL_ID=/opt/ml/model`에서 엔진을 감지합니다. 루트에 `adapter_config.json`만 있으면 `Failed to detect engine of the model`로 서버가 죽습니다. 그래서 머지 결과를 **하위(`merged/`)가 아니라 루트에** 저장합니다.
- `--dry_run`에서는 시간 낭비를 막기 위해 머지를 건너뜁니다(어댑터만 루트에 저장).
- 어댑터만 저장하고 런타임에 얹는 방식도 가능하지만, 이 킷은 **서빙 단순화**를 위해 머지를 기본으로 삼았습니다.

> ❓ **"머지하면 LoRA의 장점(작은 어댑터)이 사라지지 않나?"** — 저장·배포 단계에서는 그렇습니다. 하지만 LoRA의 장점은 **학습 효율**(적은 파라미터 업데이트)에 있고, 머지는 **서빙 편의**를 위한 별개의 선택입니다. 여러 어댑터를 스왑해야 한다면 머지하지 말고 어댑터를 그대로 보관하세요.

---

## §4.5 `MaxRuntimeExceeded` — 🔴 학습이 다 끝났는데 결과물이 버려지는 함정

### 한 줄

`ModelTrainer`에 **`stopping_condition`을 넘기지 않으면 SDK가 1시간을 자동으로 넣습니다**. 이 한도는 학습 코드 시간만이 아니라 **용량 대기 + 이미지 pull + 학습 + 머지/업로드 전체**를 포함하므로, "학습은 100% 끝났는데 머지 중에 죽어 배포 불가"가 됩니다.

### 실측 (`gemma-summarization-train-20260731084146`, ml.g6.2xlarge, 2026-07-31)

| 단계 | 소요 | 누적 |
|---|---|---|
| Pending (GPU 용량 대기) | 6분 | 6분 |
| Downloading (DLC pull) | 3분 | 9분 |
| Training — **189/189 step 전부 완료** | 55분 | 64분 |
| ⛔ 머지 도중 강제 종료 | — | 1시간 한도 도달 |

`train_runtime=3306s`로 학습은 정상 종료됐고 `Adapter saved` → `Merging LoRA adapter...` 로그까지 남았습니다. 그런데 산출물 `model.tar.gz`(542MB)에는 `adapter/`와 `checkpoint-*/`만 있고 **루트에 머지 모델이 없어**(정상 잡은 11.7GB) 서빙이 불가능했습니다. §4의 "서빙 루트에 완전한 HF 모델이 와야 한다"는 조건이 깨진 것입니다.

### 왜 진단이 어려운가

- `FailureReason`이 **비어 있습니다**. 잡 상태는 `Failed`가 아니라 `Stopped`이고, `SecondaryStatus`만 `MaxRuntimeExceeded`입니다 → "왜 멈췄지?"를 `describe-training-job`으로 파고들어야 알 수 있습니다.
- CloudWatch 로그에는 **에러가 한 줄도 없습니다**(학습이 성공했으니까). 로그만 보면 정상 종료처럼 보입니다.
- SageMaker는 종료 시 `SIGTERM` 후 **120초**를 주고 그 사이의 산출물을 업로드합니다([StoppingCondition 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)) → **불완전한 아티팩트가 "생성"되므로** 파일이 있는데 못 쓰는 상태가 됩니다.

### 대응 (이 킷의 기본값)

```python
from sagemaker.core.training.configs import StoppingCondition

MAX_RUNTIME_HOURS = 4   # SFT. GRPO는 rollout 때문에 6.
trainer = ModelTrainer(
    ...,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_HOURS * 3600),
)
```

- **넉넉히 잡아도 손해가 없습니다** — 잡이 정상 종료되면 그 시점에 과금이 멈춥니다. 이 값은 요금이 아니라 **폭주 방지 상한**입니다. (API 최대 28일, 기본 1일. SDK 기본값 1시간은 SDK 쪽 선택입니다 — `sagemaker/train/defaults.py`)
- 반대로 실습 비용을 확실히 막으려 낮출 때는 **머지·업로드용으로 최소 15분**을 남기세요(실측 머지 ~2분 + 업로드 ~3분, 모델이 커지면 늘어남).
- 노트북은 제출 전에 **예상 시간을 계산해 한도와 비교하고, 초과하면 `assert`로 막습니다**(실측 기준 seq 2048 ≈ 17s/step, seq 512 ≈ 6s/step, ml.g6.2xlarge QLoRA).

### 함께 고친 것 — 체크포인트 누적

`/opt/ml/model` **전체가** `model.tar.gz`로 올라가므로, epoch마다 쌓인 체크포인트가 업로드 시간을 늘립니다(= 한도를 잡아먹음). 서빙은 머지된 루트만 읽으므로 `save_total_limit=1`로 1개만 남깁니다(실측: 체크포인트 3개 = 0.7GB, 전부 서빙에 불필요).

> ❓ **"핸즈온인데 `MAX_RUNTIME_HOURS=4`면 4시간 과금되는 거 아냐?"** — 아닙니다. 한도는 **강제 종료 시점**일 뿐이고, 실제 과금은 잡이 실행된 시간만큼입니다. 기본값(`MAX_TRAIN_SAMPLES=200`, `EPOCHS=2`)이면 실측 20~25분 안에 끝납니다.

---

## §5 gated vs ungated · 라이선스 전파

### 대조표 — Gemma 접근/라이선스

| | 🔴 `gemma-3` 계열 (예: `gemma-3-4b-it`) | 🔴 `gemma-4` 계열 (예: `gemma-4-12B-it`) |
|---|---|---|
| 라이선스 | 커스텀 **Gemma** 라이선스(use-restriction 포함) | **apache-2.0** |
| 접근 | **GATED** — HF 토큰 + 약관 수락 필요 | **UNGATED** — 토큰 불필요 |
| HF_TOKEN | **필요**(env/`environment`로 주입) | 불필요(비워도 됨) |
| 승급 방법 | 기본값 | `MODEL_ID`만 교체 |

- `train.py`/`config.py`는 `HF_TOKEN` 또는 `HUGGING_FACE_HUB_TOKEN`을 env에서 읽습니다(huggingface_hub 버전별로 이름이 다르므로 둘 다 세팅하시기를 권장합니다). **토큰은 코드에 하드코딩하지 말고**, 노트북/`environment`로 주입하세요.
- 🔴 **라이선스 전파**: Gemma 라이선스(gemma-3 등)는 파인튜닝은 물론 **머지 산출물·서빙 결과물까지** use-restriction이 전파됩니다. 반면 apache-2.0(gemma-4)에는 그런 제약이 없습니다. **재배포·서빙 전에 라이브 모델 카드의 라이선스 배너를 재확인**하세요.
- 시드 데이터셋은 전부 permissive한 것으로 골랐습니다(글레이브 apache-2.0, banking77 cc-by-4.0, billsum cc0-1.0, dolly cc-by-sa-3.0). 세부는 `common/config.py`의 `TRACKS`를 참조하세요.

> ❓ **"gemma-4가 ungated니까 무조건 gemma-4 쓰면 되겠네?"** — 라이선스 관점만 보면 편한 것은 맞습니다. 하지만 크기(12B)·비용·품질 요구가 서로 다릅니다. 기본은 문서화가 가장 잘 된 `gemma-3-4b-it`를 쓰고, 라이선스 자유나 더 큰 용량이 필요할 때 `gemma-4-12B-it`로 조건부 승급하세요. (🔴 `gemma-4-31B`는 이 킷에서 제외했습니다.)

> ❓ **"ungated 모델에도 HF_TOKEN을 넣어야 안전한 거 아냐?"** — 넣지 않아도 됩니다. gemma-4 계열은 토큰 없이 받아집니다. 다만 넣어도 무해하므로, 트랙을 오갈 때 편의상 세팅해 두셔도 괜찮습니다.

---

## §5.5 SFT → GRPO (RL) — 🔴 **데이터를 반드시 갈아야 합니다**

> `02a_train_grpo_sagemaker` 를 실행하기 전에 읽으세요. "SFT 데이터를 그대로 쓰면 되지 않나?"의 답입니다.

### 한 줄
**SFT와 RL은 필요한 데이터가 다릅니다.** SFT는 (입력, 정답) 쌍으로 정답을 모방하고, RL은 **prompt만** 받아
스스로 생성한 뒤 reward로 채점합니다. 같은 데이터를 쓰면 누출이고, 더 나쁘게는 **학습이 아예 안 됩니다.**

### 왜 학습이 안 되는가 — advantage ≈ 0
GRPO는 prompt 하나당 `num_generations`개(이 킷 기본 8)를 생성해 **그룹 안에서 상대 비교**로 학습합니다.

```
prompt → rollout 8개 → 각각 reward 채점 → 그룹 평균 대비 편차(advantage)로 gradient
```

SFT가 이미 잘 맞히는 prompt를 주면 **8개가 모두 만점**이 됩니다. 그룹 내 편차가 없으니 advantage가 0에
수렴하고, **gradient가 거의 흐르지 않습니다.** GPU 시간만 쓰고 배우는 것이 없습니다.

🔴 그래서 **슬라이스만 분리해도 부족합니다.** 같은 분포에서 잘라낸 다른 100건은 누출은 막지만
모델이 여전히 잘 맞히므로 advantage 문제는 그대로 남습니다.

### 대조표 — 무엇이 다른가

| | SFT | GRPO / RL |
|---|---|---|
| 목적 | 형식·기본 능력 습득 | **SFT가 실패하는 구간** 개선 |
| 필요한 데이터 | (입력, **정답**) 쌍 | **prompt** (+ 프로그램적 채점용 reference) |
| 정답의 역할 | 학습 입력으로 직접 사용 | **reward 계산에만** — 모델에 보여주지 않음 |
| 좋은 prompt | 대표적인 것 | **모델이 어려워하는 것** |
| 연산량 | 1 forward/backward | prompt당 rollout N개 → **수 배** |

`train_grpo.py` 의 변환을 보면 정답이 학습 입력이 아님이 드러납니다:
```python
def _to_grpo(example):
    prompt    = [m for m in msgs if m["role"] != "assistant"]   # ← 모델에 들어가는 것
    reference = ...assistant content...                          # ← reward 계산용만
```

### 그럼 RL 데이터는 어디서 오나 — 3가지 경로

이 킷은 `common/grpo_data.py` 로 세 가지를 제공하고, 노트북에서 `GRPO_PROMPT_SOURCE` 로 고릅니다.

| 소스 | 무엇 | 비용·선행조건 | advantage 확보 |
|---|---|---|---|
| `holdout` | SFT가 쓰지 않은 구간 | 무료·즉시 | ⚠️ 같은 분포라 약함 |
| `synth` | Bedrock으로 **prompt만** 생성 | Bedrock 과금 | 분포를 넓혀 유리 |
| `failures` | 04_evaluate에서 **틀린 건만** | 04 선행 필요 | 🔴 가장 강함 |

- **`holdout`** (튜토리얼 기본) — 추가 비용 없이 파이프라인을 끝까지 볼 수 있게 하는 값입니다.
  누출은 막지만, 학습 후 reward가 거의 변하지 않으면 위 advantage 문제로 보고 다른 소스로 옮기세요.
- **`synth`** — RL은 정답이 학습 입력이 아니므로 **prompt 생성이 SFT 합성보다 쉽고 쌉니다.**
  (다만 이 킷의 reward는 프로그램적 채점이라 reference가 필요해 (input, output) 형태로 만듭니다.)
  ⚠️ SFT 합성과 **같은 시드**를 주면 분포가 또 겹치므로, SFT가 쓰지 않은 시드 구간을 넘기세요.
- **`failures`** — 실무에서 가장 효율적인 경로입니다. reward 신호가 강한 구간에만 집중합니다.
  실패가 0건이면 GRPO로 얻을 것이 적다는 뜻이니(좋은 신호), `N_EVAL` 을 키워 더 어려운 케이스를 찾으세요.

프로덕션에서는 여기에 하나 더 있습니다 — **실제 트래픽 로그**. 분포가 진짜라서 가장 가치 있지만,
공개 데이터로 재현할 수 없어 이 킷에는 넣지 않았습니다.

### 왜 추출·분류 트랙에만 GRPO가 있나
GRPO에는 **프로그램적으로 채점 가능한 reward**가 필요합니다.
- 추출 → JSON 유효성 + 함수명/인자 일치로 채점 가능 ✅
- 분류 → 라벨 정확 일치로 채점 가능 ✅
- 요약·QA → "좋은 요약"을 규칙으로 채점할 수 없습니다. LLM-judge를 reward로 쓸 수 있지만
  rollout마다 judge를 호출해야 해 비용·시간이 급증하고 judge 편향이 학습에 섞입니다 → 이 킷은 제외.

> ❓ **"SFT 없이 base에서 바로 GRPO 하면 안 되나?"** — 됩니다(`train_grpo.py` 가 `model` 채널이 없으면
> HF base로 폴백). 다만 형식조차 안 잡힌 상태에서는 rollout이 전부 낮은 점수라 역시 편차가 작고
> 수렴이 불안정합니다. **SFT → GRPO** 가 정석인 이유입니다.

---

## §6 ❓ 오개념 노트 모음

> ❓ **"파인튜닝 = 전체 가중치 학습"** — 이 킷은 **PEFT LoRA/QLoRA**로 어댑터만 학습하며, full fine-tune이 아닙니다. 그래서 단일 GPU에서도 SLM을 돌릴 수 있습니다.

> ❓ **"로컬에서 됐으니 SageMaker에서도 그대로 되겠지"** — 같은 `train.py`를 쓰므로 대체로 맞습니다. 다만 (1) boolean은 `--key value`로 들어오고(§2.6), (2) 데이터는 `SM_CHANNEL_TRAIN`으로 들어오며, (3) 컨테이너의 `transformers` 버전은 로컬과 다를 수 있습니다(→ `requirements.txt`가 조정). 이 3가지는 dry-run으로 미리 잡을 수 있습니다.

> ❓ **"DLC 이미지 태그는 최신으로 아무거나 넣으면 되지"** — 그렇지 않습니다. 🔴 DLC 태그는 **AWS가 게시한 조합만** 유효합니다(임의의 최신 버전이 아닙니다). `common/dlc.py`는 `DLC_IMAGE_URI`/`DLC_TAG` env로 정확한 태그를 주입하게 하고, 없으면 SDK 버전 조합으로 폴백합니다. 현행 태그는 available_images 페이지에서 확인하세요. ECR 계정은 🔴 `763104351884`(대부분의 리전 공용)이며, 패턴은 `763104351884.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`입니다.

> ❓ **"DLC(컨테이너) = DLAMI(AMI)?"** — 서로 다릅니다. DLC는 **워크로드 컨테이너 이미지**(학습/추론)이고, DLAMI는 **노드 호스트 이미지(AMI)**입니다. 이 킷은 SageMaker 학습 컨테이너로 **DLC**를 씁니다.

> ❓ **"학습이 곧 배포"** — 아닙니다. `02_train_sft_sagemaker`(학습)와 `03_deploy_endpoint`(추론 endpoint)는 별개의 단계입니다. 🔴 SageMaker 추론은 Real-time / Serverless(⚠️GPU 없음→LLM 부적합) / Asynchronous / Batch Transform의 4옵션이 있으며, 이 킷은 **real-time endpoint**를 씁니다(배포는 03 문서를 참조하세요).

---

## §7 비용 / cleanup 주의

- **학습 잡은 실행 시간만큼 과금**됩니다. 🔴 GPU 인스턴스(`ml.g5.*` 등)는 저렴하지 않으니, 반드시 **로컬 `--dry_run`으로 파이프라인을 먼저 검증**한 뒤 실제 `.fit()`을 돌리세요.
- spot(`use_spot_instances=True`)을 쓰면 학습 비용을 크게 줄일 수 있습니다(체크포인트 설정 필요). 다만 AWS의 절감 수치는 리전·수급에 따라 다르므로 절대값으로 약속하지 마세요.
- S3 아티팩트(모델·머지 산출물)와 학습 로그는 **누적해서 과금**됩니다. 트랙 완료 후에는 `99_cleanup` 노트북으로 endpoint·아티팩트를 정리하세요.
- **학습 잡 자체는 끝나면 인스턴스가 자동으로 해제**되지만, endpoint(03 단계)는 **켜 두면 계속 과금**됩니다 — 이 문서의 범위 밖이지만 미리 알아 두시기 바랍니다.

---

## 라이브 검증 소스 (2026-07)

🔴 아래 세부 사항(정확한 태그·버전·모델 ID·리전·GA)은 **실행 직전에 재확인**하셔야 합니다.

| 주제 | URL |
|---|---|
| Gemma 모델 카드/라이선스(gated·ungated 배너 재확인) | https://huggingface.co/google |
| Gemma chat template / fine-tuning 가이드 | https://ai.google.dev/gemma/docs |
| TRL `SFTTrainer` / `SFTConfig`(packing·chat template) | https://huggingface.co/docs/trl |
| PEFT `LoraConfig`(`target_modules`, `modules_to_save`) | https://huggingface.co/docs/peft |
| SageMaker HuggingFace estimator(`entry_point`/`source_dir`/hyperparameters) | https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/ |
| SageMaker + HuggingFace(공식 가이드) | https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html |
| SageMaker JumpStart(대조용) | https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html |
| `StoppingCondition`(§4.5 — MaxRuntime 범위·SIGTERM 120초·최대 28일) | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html |
| DLC available images(ECR 계정 763104351884 · 현행 태그) | https://aws.github.io/deep-learning-containers/reference/available_images/ |
| AWS Deep Learning Containers(공식 GitHub) | https://github.com/aws/deep-learning-containers |
| TRL(공식 GitHub) | https://github.com/huggingface/trl |
| PEFT(공식 GitHub) | https://github.com/huggingface/peft |
| SageMaker Python SDK(공식 GitHub) | https://github.com/aws/sagemaker-python-sdk |
| SageMaker HuggingFace 예제(공식 GitHub) | https://github.com/aws/amazon-sagemaker-examples |

---

### 네비게이션
- ◀ 이전: [`01_data_and_synthetic`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/tracks/01_extraction_to_json/01_data_and_synthetic.ipynb) — 시드 + grounded 합성 데이터(conversational `messages` 포맷 생성)
- ▶ 다음: [`02_train_sft_sagemaker`](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/tracks/01_extraction_to_json/02_train_sft_sagemaker.ipynb) — 이 문서의 `train.py`를 dry-run → `.fit()`으로 실행
- 참조 코드: `tracks/*/scripts/train.py`, `common/config.py`, `common/dlc.py`
