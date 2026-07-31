# 03 · Grounded 합성 데이터 생성

> **대상 독자** — 파인튜닝용 라벨 데이터가 부족해 합성으로 보강하려는 개발자를 위한 문서입니다. Bedrock/SageMaker를 처음 다뤄도 괜찮습니다.
> **⚠️ 주의** — Bedrock은 **토큰 과금**입니다(대량 생성 시 비용 발생). 🔴 모델 ID·SDK 버전·리전은 **빠르게 바뀌므로** 실행 전 재확인 표시(🔴)를 반드시 확인하세요. 시크릿/계정ID/절대경로 하드코딩 금지.
> **라이브 검증 2026-07** — 이 프로젝트의 `synth-toolkit-recon` 워크플로우(2026-07-19, GitHub/PyPI 실측)를 기반으로 합니다. 세부 값은 실행 시점에 다시 확인하세요.

관련 킷 파일은 다음과 같습니다: `common/synth/bedrock_synth.py` · `common/synth/README.md` · `common/config.py` · `common/aws_utils.py` · `common/gemma_format.py` · `common/llm_gateway.py` · 각 트랙의 `01_data_and_synthetic.ipynb`.

---

## §0 TL;DR

**한 줄** — 합성은 "자유 생성"이 아니라 **seed에 grounded하고 critique/refine로 걸러낸** 데이터만 채택하는 방식이며, 기본 경로는 외부 SDG 라이브러리가 0개인 `bedrock_synth.py`(boto3 Bedrock Converse)입니다.

1. **왜 grounded인가** — 근거 없는 자유 생성은 hallucination과 distribution drift를 학습셋에 주입합니다. 그래서 생성을 seed 도메인/라벨공간에 묶고, LLM critique(groundedness·relevance)로 임계값에 미달하는 예시를 폐기합니다.
2. **기본 경로(무의존성)** — `bedrock_synth.generate_grounded()`는 생성 → critique → PII/중복 필터 → `messages` JSONL 순으로 동작합니다. boto3만 사용하므로 라이브러리 노후화 리스크가 없습니다.
3. **몇 개 생성할지는 USER가 결정** — 기본값은 `NUM_SYNTHETIC`(500/트랙)입니다. 절대값이 아니므로 데이터·비용·품질 트레이드오프를 보며 조정하세요.
4. **held-out 규율(절대원칙)** — 🔴 **합성 데이터로는 평가하지 마세요**. 증강하기 이전의 seed에서 held-out을 먼저 분리한 뒤, 나머지만 증강합니다.
5. **라이브러리 대안** — 활발히 유지보수되는 **Kiln**(native Bedrock)과 **Bespoke Curator**(LiteLLM 경유)만 권장합니다. **distilabel은 정체 상태라 배제**합니다.

---

## §0.5 기존 Pain Point

- 라벨 데이터가 수백 건뿐이라 파인튜닝이 과소적합되거나 불안정해집니다.
- "LLM한테 그냥 예시 만들어달라고 하면 되지 않나?" 싶지만, 자유 생성은 seed 분포를 벗어난 예시를 만들어 **teacher 모델의 편향·환각을 그대로 학습**시킵니다(distribution drift).
- 합성으로 성능을 평가하면 **과대평가**로 이어집니다. teacher를 모방한 데이터로 student를 채점하는 순환 오류이기 때문입니다.
- SDG 라이브러리를 붙였더니 **1년째 릴리스가 없어** 의존성이 썩는 경우도 있습니다(예: distilabel).

---

## §1 왜 grounded + critique인가? (자유 생성이 아니라)

### 쉽게 말하면
"백지에 상상해서 써봐"(자유 생성)라고 시키는 대신, **"이 실제 샘플들과 같은 도메인·스타일·라벨공간 안에서, 겹치지 않는 새 예시를 만들어"**(grounded)라고 지시합니다. 그리고 만들어진 결과를 **다시 채점자 LLM에게 검수시켜** 합격한 것만 남기는 방식입니다.

### 대조표 — 자유 생성 vs grounded+critique

| 축 | 자유 생성 (free) | grounded + critique/refine (이 킷) |
|---|---|---|
| 근거 | 없음 (모델 사전지식) | seed 샘플의 도메인·엔티티·라벨공간 |
| distribution drift | 큼 (seed 분포 이탈) | 억제 (seed 회전으로 다양성 확보) |
| hallucination | 학습셋에 그대로 주입 | groundedness 임계값 미달 폐기 |
| 품질 게이트 | 없음 | LLM judge가 groundedness·relevance 채점 |
| 안전 | 수동 | PII 정규식 + 중복(sha256) 필터 자동 |

### 기술적 차이 3가지
1. **seed 회전 grounding** — batch마다 서로 다른 seed chunk를 근거로 넣어(`seeds_per_batch`) 다양성과 근거를 동시에 확보합니다. 이렇게 하면 특정 seed에 과적합된 복제를 방지할 수 있습니다.
2. **2축 critique 게이트** — 생성물마다 `groundedness`(seed 도메인 일치)와 `relevance`(task 적합)를 0~1로 재채점하며(temperature=0.0), `min_groundedness`/`min_relevance`(기본 0.6)에 미달하면 폐기합니다.
3. **채택분만 accumulate** — 목표치인 `n_total`에 도달할 때까지 루프를 돕니다. PII/중복 필터를 통과하고 critique를 넘긴 예시만 `SynthExample`로 쌓입니다.

> ❓ **"critique도 결국 같은 LLM이 하는데 의미가 있나?"**
> 생성은 temperature=0.8(다양성)로, critique는 temperature=0.0(결정론적 판정)으로 설정해 **역할과 설정을 분리**합니다. 완벽한 진리 검증은 아니지만, seed 도메인 이탈이나 라벨공간 위반 같은 **명백한 drift를 걸러내는 저비용 게이트**로서 유효합니다. 최종 품질 판정은 §held-out 평가(`04_evaluate`)가 담당합니다.

---

## §2 기본 경로 — `common/synth/bedrock_synth.py` (무의존성)

### 쉽게 말하면
"boto3로 Bedrock Claude를 불러서 만들고, 같은 Claude로 검수하고, 정규식으로 PII/중복을 걸러 JSONL로 저장한다" — 이것이 전부입니다. 외부 합성 라이브러리는 **일부러 사용하지 않습니다**. 노후화 리스크가 없고, IAM/VPC/Guardrails 같은 AWS 네이티브 거버넌스를 그대로 활용할 수 있기 때문입니다.

### 파이프라인 (ASCII)
```
seed 샘플(증강 이전)
      │  seed 회전 (seeds_per_batch)
      ▼
[생성] Bedrock Converse  ── GEN_SYSTEM, temp=0.8, JSON array
      │  _extract_json (코드펜스/부분블록 방어 파싱)
      ▼
[필터1] PII (email/phone/SSN/card 정규식) → 탈락
[필터2] 중복 (input+output sha256) → 탈락
      ▼
[critique] Bedrock Converse ── CRITIQUE_SYSTEM, temp=0.0
      │  groundedness / relevance (0~1)
      ▼
  g ≥ min_groundedness AND r ≥ min_relevance ?
      │ yes
      ▼
to_messages() → 표준 messages (트랙별 어댑터, gemma_format.build_messages)
      ▼
save_jsonl → {"messages":[...]} JSONL  →  SFTTrainer conversational 입력
```

### 핵심 호출 (노트북 `01_data_and_synthetic`에서)
```python
from common.synth import bedrock_synth as bs
synth = bs.generate_grounded(
    task_instruction=td.TASK_INSTRUCTION,      # 트랙별
    seed_texts=td.seed_texts_for_synth(seeds), # 증강 이전 seed
    n_total=config.NUM_SYNTHETIC,              # USER 결정값 (아래 §3)
    model_id=config.BEDROCK_CLAUDE_MODEL_ID,   # 🔴 env 주입, 하드코딩 금지
    region=config.BEDROCK_REGION,
    to_messages=td.to_messages,                # {"input","output"} → messages
)
```
- 저수준 호출은 `common/aws_utils.bedrock_converse()`가 담당합니다(boto3 `bedrock-runtime` 클라이언트의 `converse()`). 🔴 이는 SageMaker endpoint 호출(`sagemaker-runtime`의 `invoke_endpoint`)과 **별개 서비스**이니, §오개념 노트를 참고하세요.
- 출력은 `{"messages":[...]}` JSONL입니다. `gemma_format.build_messages`로 만든 표준 messages이므로 TRL `SFTTrainer`의 conversational 포맷에 바로 들어갑니다(학습 경로는 `tracks/*/scripts/train.py`).

> ❓ **"PII 정규식이 순수 숫자열도 지우지 않나?"**
> 그렇지 않습니다. 실제 코드는 phone/card 패턴에 **구분자(공백/-/괄호)를 요구**하기 때문에, function-call JSON에 흔한 타임스탬프·금액·id 같은 순수 긴 숫자열은 PII로 오탐하지 않습니다. 이는 오탐 때문에 유효한 합성이 폐기되는 것을 막기 위한 설계입니다.

---

## §3 몇 개 생성할까 — `NUM_SYNTHETIC`은 USER 결정

- 기본값은 **500/트랙**입니다(`config.NUM_SYNTHETIC`, env `NUM_SYNTHETIC`로 오버라이드). 이는 Gate 4에서 정한 **합리적 기본값일 뿐, 절대 규칙은 아닙니다**.
- **언제 늘리나요** — seed 다양성이 높고 도메인이 넓을 때, 그리고 held-out 지표가 데이터량에 비례해 오를 때 늘리세요.
- **언제 줄이나요** — Bedrock 비용이 부담될 때, 또는 seed가 좁아 다양성이 금방 포화될 때(중복 필터가 많이 걸립니다) 줄이세요.
- dry-run(`DRY_RUN=1`)은 파이프라인 검증용으로 소량(노트북에서 6건)만 생성하므로, 실제 값과는 별개입니다.

> ❓ **"합성을 많이 만들수록 항상 좋은가요?"**
> 그렇지 않습니다. seed 다양성이 한계에 다다르면 추가 생성은 **중복·near-duplicate**만 늘리고(중복 필터가 폐기합니다) 비용만 증가시킵니다. 양보다는 **held-out 지표(`04_evaluate`)로 검증한 유효 증가분**을 기준으로 조정하세요.

---

## §4 held-out 규율 — 합성으로 평가 금지 (절대원칙)

### 쉽게 말하면
채점 시험지(held-out)를 **증강하기 전에 미리 떼어놓고**, 나머지만 합성으로 부풀립니다. 합성 데이터는 teacher(Claude)를 모방한 것이므로, 합성으로 채점하면 "teacher를 얼마나 잘 베꼈나"만 재게 되어 **과대평가**로 이어집니다.

```
seed 전체
   ├─ [먼저 분리] held-out  ──────────────►  04_evaluate 전용 (합성 유입 0)
   └─ 나머지 seed  ──►  grounded 합성 증강  ──►  train.jsonl (SFTTrainer)
```

- 평가는 `04_evaluate.ipynb`에서 **held-out으로만** 수행합니다. 트랙별 메트릭은 다음과 같습니다. 추출은 `arg_f1`+`valid_json_rate`, 분류는 macro-F1+accuracy, 요약은 ROUGE-L+LLM-judge, QA는 LLM-judge+ROUGE-L을 사용합니다(`common/eval_utils.py`).
- 구현상 주의할 점이 있습니다(현행 노트북 기준). `04_evaluate`는 간단히 seed pool의 **뒤쪽 N건**을 held-out으로 쓰고 학습에는 앞쪽을 씁니다. 🔴 production에서는 **고정 시드(seed) 기반의 결정론적 분리**를 사용해 학습/held-out 인덱스가 확실히 겹치지 않게 하세요(노트북 markdown에도 명시되어 있습니다).

> ❓ **"합성 데이터의 groundedness 점수가 높으면 평가에 써도 되나요?"**
> 안 됩니다. groundedness는 "seed 도메인에 맞는가"를 나타낼 뿐 "정답인가"를 뜻하지 않습니다. 게다가 합성은 정의상 teacher의 출력이므로, 이를 채점 기준으로 쓰면 **평가 순환 오류**가 그대로 남습니다. held-out은 반드시 실제 seed(증강 이전)에서만 뽑아야 합니다.

---

## §5 라이브러리 대안 — 활발히 유지보수되는 것만

기본 경로(`bedrock_synth.py`)만으로도 충분하지만, 오케스트레이션이나 대량 실행이 필요하다면 아래 대안을 붙일 수 있습니다. 🔴 버전과 라이선스는 실행 전에 다시 확인하세요(PyPI/repo 실측 2026-07).

| 도구 | Bedrock 연동 | 상태 (2026-07 실측) | 라이선스 | 쓸 때 |
|---|---|---|---|---|
| **`bedrock_synth.py` (이 킷)** | boto3 native | — | 킷 코드 | 기본값. 의존성 0, AWS 거버넌스 |
| **Kiln** (`kiln-ai`) | **native** (`ModelProviderName.amazon_bedrock`) | 가장 활발 (v1.0.4 @ 2026-07-16) | ⚠️ core lib MIT / repo 루트 커스텀 | GUI+오케스트레이션 원할 때 |
| **Bespoke Curator** | **LiteLLM 경유** (`bedrock/...`) | 활발 (0.1.29 @ 2026-07-13) | Apache-2.0 | 코드-우선, 대량·구조화·캐싱 |
| ~~distilabel~~ | — | 🔴 **정체** (마지막 v1.5.3 @ 2025-01-28, 2026 릴리스 0건) | — | **배제 — 사용 금지** |

- **Kiln** — native Bedrock 연동을 코드 수준에서 확인한 유일한 도구입니다. repo는 `github.com/Kiln-AI/Kiln`이며, `pip install kiln-ai`로 설치합니다. ⚠️ 리포 루트 라이선스와 core lib 라이선스가 다르므로 재배포 전에 반드시 확인하세요.
- **Bespoke Curator** — native 커넥터는 아니고 LiteLLM을 경유합니다(`bedrock/<model>` + AWS 자격증명). repo는 `github.com/bespokelabsai/curator`입니다. 이 킷의 `common/llm_gateway.py`(LiteLLM)와 라우팅 규약이 일치하므로 연결이 자연스럽습니다.
- 대안을 쓰더라도 **grounded + critique 원칙은 동일하게 적용**하고, 출력은 이 킷의 `messages` JSONL로 변환해 `train.py`에 넣으세요.

> ❓ **"유명한 distilabel을 왜 안 쓰나요?"**
> 유명세와 유지보수는 별개입니다. 정찰 시점 기준 **2026년 릴리스가 0건**(마지막 2025-01)이라 프로덕션 의존성으로는 부적합합니다. 이 킷은 "노후화 리스크 0"을 기본 원칙으로 삼습니다. 🔴 상태는 변할 수 있으니 채택 전에 repo 활동을 다시 확인하세요.

---

## §6 비용 / cleanup 주의

- **Bedrock = 토큰 과금** — 입력과 출력 토큰을 기준으로 과금됩니다. 이 킷은 예시당 **생성 1회 + critique 1회**를 호출하므로, `NUM_SYNTHETIC`이 클수록, 그리고 seed chunk가 길수록 토큰이 늘어납니다. 대량 생성에 앞서 **소량으로 단가를 측정한 뒤 총량을 추정**하기를 권장합니다.
- AWS 마케팅 수치("최대 N% 절감" 등)는 **AWS 주장**으로 표기하고 출처를 붙이세요(이 문서에서는 인용하지 않습니다).
- **모델 ID** 🔴 — `BEDROCK_CLAUDE_MODEL_ID`에는 inference-profile prefix(`us.`/`eu.`/`apac.`/`global.`)가 필수입니다. 최신(4.6+) Claude는 dateless pinned-snapshot 형식일 수 있습니다. 호출 시점에 Bedrock 콘솔에서 정확한 문자열을 확인한 뒤 **env로 주입**하세요(하드코딩 금지).
- **cleanup** — 합성 생성 자체는 상시 리소스를 남기지 않고 호출 시에만 과금됩니다. 다만 같은 노트북 흐름의 endpoint나 학습 잡은 별도이므로 `99_cleanup.ipynb`로 정리하세요.

---

## §7 ❓ 오개념 노트 모음

> ❓ **"합성 데이터 생성이 곧 SageMaker endpoint 호출 아닌가요?"**
> 아닙니다. 합성 생성은 **Bedrock**(`bedrock-runtime`의 `converse`)으로 teacher LLM(Claude)을 부르는 것이고, 학습된 SLM 서빙은 **SageMaker**(`sagemaker-runtime`의 `invoke_endpoint`, 스트리밍은 `invoke_endpoint_with_response_stream`)로 처리합니다. 둘은 **서로 다른 서비스**입니다.

> ❓ **"Bedrock Converse가 곧 SageMaker Batch Transform인가요?"**
> 아닙니다. 합성 대량 생성은 Bedrock API를 반복 호출하는 작업입니다. SageMaker 추론 4옵션(Real-time / Serverless / Asynchronous / Batch Transform)은 **학습된 모델을 서빙**하는 이야기이며, 특히 🔴 **Serverless는 GPU가 없어 LLM/SLM 서빙에 부적합**합니다(합성 생성과는 무관합니다).

> ❓ **"grounded면 seed를 그대로 복사하는 것 아닌가요?"**
> 아닙니다. 프롬프트가 "verbatim 복사 금지, 같은 도메인/스타일/라벨공간의 **새** 예시"를 요구하고, 중복 필터(sha256)가 동일하거나 유사한 복제를 걸러냅니다. seed는 어디까지나 근거일 뿐, 정답을 복사할 대상이 아닙니다.

> ❓ **"critique 임계값 0.6은 고정값인가요?"**
> 아닙니다. `min_groundedness`/`min_relevance` 인자로 트랙별로 조정할 수 있습니다. 정밀한 라벨 태스크는 임계값을 올리고, 다양성이 중요한 QA는 낮출 수 있습니다.

---

## 라이브 검증 2026-07

| 주제 | URL |
|---|---|
| Bedrock Converse API | https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html |
| Bedrock inference profiles (region prefix) | https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html |
| Bedrock 요금(토큰 과금) | https://aws.amazon.com/bedrock/pricing/ |
| SageMaker 추론 옵션 (Real-time/Serverless/Async/Batch) | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html |
| SageMaker Runtime `invoke_endpoint` (boto3) | https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html |
| TRL SFTTrainer (conversational/messages) | https://huggingface.co/docs/trl/en/sft_trainer |
| boto3 `bedrock-runtime` converse | https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html |
| Kiln (native Bedrock) — repo | https://github.com/Kiln-AI/Kiln |
| Bespoke Curator — repo | https://github.com/bespokelabsai/curator |
| LiteLLM (Bedrock 라우팅) | https://docs.litellm.ai/docs/providers/bedrock |
| (배제) distilabel — repo (릴리스 케이던스 확인용) | https://github.com/argilla-io/distilabel |

🔴 위 링크의 값(모델 ID·리전·요금·라이브러리 버전/라이선스)은 **빠르게 바뀌므로, 실행하거나 게시하기 전에 반드시 다시 확인하세요**.

관련 노트북/파일은 다음과 같습니다: `tracks/*/01_data_and_synthetic.ipynb` · `common/synth/bedrock_synth.py` · `common/synth/README.md` · `common/config.py`(`NUM_SYNTHETIC`, `BEDROCK_CLAUDE_MODEL_ID`) · `common/aws_utils.py`(`bedrock_converse`) · `common/gemma_format.py`(`build_messages`) · `common/eval_utils.py` · `tracks/*/04_evaluate.ipynb`(held-out).

---

### 핸드오프 — aws-fact-checker에 넘긴 fact-critical 주장

| 주장 | 상태 | 비고 |
|---|---|---|
| SageMaker Serverless에 GPU 없음 → LLM 부적합 | ✅ 라이브 검증(2026-07) 근거 사용, 🔴 재확인 표시 유지 | 정책 변경 가능 |
| Bedrock 합성 호출(`bedrock-runtime` converse) ≠ SageMaker endpoint(`sagemaker-runtime` invoke) 별개 서비스 | ✅ 검증 근거 | — |
| distilabel 정체(마지막 릴리스 2025-01, 2026 릴리스 0건) | ⚠️ **fact-checker 확인 필요** | repo 활동은 시점 의존 — 게시 전 재확인 |
| Kiln v1.0.4(2026-07-16) / Bespoke Curator 0.1.29(2026-07-13) 버전·라이선스 | ⚠️ **fact-checker 확인 필요** | PyPI/repo 실측값, 빠르게 변함 |
| inference-profile prefix(us./eu./apac./global.) 필수 | ✅ 검증 근거 | 모델 ID 형식은 재확인 |
| `NUM_SYNTHETIC` 기본 500/트랙 | ✅ 킷 `config.py` 실측 | USER 결정값 |
