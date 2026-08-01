# 02 · Grounded 합성 데이터 — seed에 근거하고 critique로 걸러내기

!!! info "읽는 사람과 범위"
    파인튜닝용 라벨 데이터가 부족해 합성으로 보강하려는 개발자를 위한 문서입니다. Bedrock/SageMaker를 처음 다뤄도 괜찮습니다.
    선행 조건은 없습니다 — 이 문서가 대응하는 노트북이 각 트랙의 첫 단계(`01_data_and_synthetic.ipynb`)입니다.
    다루는 것은 grounded 생성·critique 게이트·held-out 규율·라이브러리 대안이고, 학습 자체는 [파인튜닝](03_finetuning.md)이 다룹니다.

이 문서와 관련된 킷 파일:

- `common/synth/bedrock_synth.py` — grounded 생성 + critique/refine 본체(`generate_grounded`), PII/중복 필터
- `common/synth/README.md` — 기본 경로와 오픈 라이브러리 대안의 선택 근거
- `common/config.py` — 모델 ID·리전·`NUM_SYNTHETIC` 등 env 기반 설정
- `common/aws_utils.py` — Bedrock Converse 저수준 호출(`bedrock_converse`)과 SageMaker/Bedrock 서비스 경계
- `common/gemma_format.py` — 트랙별 raw row → 표준 `messages` 변환(`build_messages`)
- `common/llm_gateway.py` — LiteLLM 경유로 Bedrock과 SageMaker endpoint를 단일 인터페이스로 호출(대안 경로)
- `common/eval_utils.py` — 트랙별 held-out 평가 메트릭

노트북 순서: 각 트랙의 `01_data_and_synthetic.ipynb`(생성) → (학습·배포) → `04_evaluate.ipynb`(held-out 전용)

!!! warning "빠르게 바뀌는 값"
    Bedrock 모델 ID·inference-profile prefix·sampling 파라미터 지원 여부·토큰 단가·boto3/SDK 버전·리전, 그리고 대안 라이브러리의 버전·라이선스·유지보수 상태는 **실행 직전에 다시 확인**하세요.
    특히 Claude 세대가 바뀌면 `temperature` 같은 파라미터가 조용히 deprecated 됩니다(아래 실측 참고).
    시크릿·계정 ID·절대경로는 문서와 코드 어디에도 하드코딩하지 마세요.

---

## TL;DR

**합성은 "자유 생성"이 아니라 seed에 grounded하고 critique/refine로 걸러낸 데이터만 채택하는 방식이며, 기본 경로는 외부 SDG 라이브러리가 0개인 `bedrock_synth.py`(boto3 Bedrock Converse)입니다.**

1. **근거 없는 자유 생성은 hallucination과 distribution drift를 학습셋에 주입합니다.** 그래서 생성을 seed 도메인·라벨공간에 묶고, LLM critique(groundedness·relevance)로 임계값 미달 예시를 폐기합니다 — [왜 자유 생성이 아니라 grounded인가](#왜-자유-생성이-아니라-grounded인가).
2. **기본 경로는 무의존성입니다.** `generate_grounded()`는 생성 → PII/중복 필터 → critique → `messages` JSONL 순으로 동작하며 boto3만 씁니다 — [기본 경로 — 무의존성 bedrock_synth](#기본-경로--무의존성-bedrock_synth).
3. **몇 개 생성할지는 USER가 결정합니다.** 기본값은 `config.NUM_SYNTHETIC`(200)이고 이 킷의 `.env`는 100으로 낮춰 둔 상태입니다 — [생성 건수 결정 — NUM_SYNTHETIC 기본값](#생성-건수-결정--num_synthetic-기본값).
4. **합성 데이터로는 평가하지 마세요.** 증강 이전 seed에서 held-out을 먼저 분리한 뒤 나머지만 증강합니다 — [held-out 규율 — 합성으로 평가 금지](#held-out-규율--합성으로-평가-금지).
5. **라이브러리는 활발히 유지보수되는 것만 권장합니다.** Kiln(native Bedrock)과 Bespoke Curator(LiteLLM 경유)이고, distilabel은 배제합니다 — [라이브러리 대안](#라이브러리-대안).

---

## 기존 Pain Point

- 라벨 데이터가 수백 건뿐이라 파인튜닝이 과소적합되거나 불안정합니다.
- "LLM한테 그냥 예시 만들어달라고 하면 되지 않나?" 싶지만, 자유 생성은 seed 분포를 벗어난 예시를 만들어 **teacher 모델의 편향·환각을 그대로 학습**시킵니다(distribution drift).
- 합성으로 성능을 평가하면 **과대평가**로 이어집니다. teacher를 모방한 데이터로 student를 채점하는 순환 오류이기 때문입니다.
- SDG 라이브러리를 붙였더니 **1년째 릴리스가 없어** 의존성이 썩는 경우도 있습니다(예: distilabel).
- 합성 스크립트를 돌렸는데 결과가 0건이고, 로그는 `skipped ('text')` 한 줄뿐이라 원인을 알 수 없습니다.

이 문서는 앞의 세 가지를 grounded 생성 + critique 게이트 + held-out 분리 규율로 막고, 의존성 노후화는 외부 SDG 라이브러리 0개로, 진단 불가 로그는 예외 종류까지 남기는 로깅으로 해결하는 방법을 설명합니다.

---

## 왜 자유 생성이 아니라 grounded인가

!!! abstract "쉽게 말하면"
    "백지에 상상해서 써봐"(자유 생성)라고 시키는 대신, **"이 실제 샘플들과 같은 도메인·스타일·라벨공간 안에서, 겹치지 않는 새 예시를 만들어"**(grounded)라고 지시합니다.
    그리고 만들어진 결과를 **다시 채점자 LLM에게 검수시켜** 합격한 것만 남깁니다.

### 대조표 — 자유 생성 vs grounded critique

| 축 | 자유 생성 (free) | grounded + critique/refine (이 킷) |
|---|---|---|
| 근거 | 없음 (모델 사전지식) | seed 샘플의 도메인·엔티티·라벨공간 |
| distribution drift | 큼 (seed 분포 이탈) | 억제 (seed 회전으로 다양성 확보) |
| hallucination | 학습셋에 그대로 주입 | groundedness 임계값 미달 폐기 |
| 품질 게이트 | 없음 | LLM judge가 groundedness·relevance 채점 |
| 안전 | 수동 | PII 정규식 + 중복(sha256) 필터 자동 |

### 기술적 차이 3가지

1. **seed 회전 grounding** — batch마다 서로 다른 seed chunk를 근거로 넣어(`seeds_per_batch`, 기본 4) 다양성과 근거를 동시에 확보합니다. 특정 seed에 과적합된 복제를 막는 장치입니다.
2. **2축 critique 게이트** — 생성물마다 `groundedness`(seed 도메인 일치)와 `relevance`(task 적합)를 0~1로 재채점하고, `min_groundedness`/`min_relevance`(기본 0.6)에 미달하면 폐기합니다. 판정용 시스템 프롬프트(`CRITIQUE_SYSTEM`)는 생성용(`GEN_SYSTEM`)과 완전히 분리되어 있습니다.
3. **채택분만 accumulate** — 목표치 `n_total`에 도달할 때까지 라운드를 반복하고, PII/중복 필터를 통과하고 critique를 넘긴 예시만 `SynthExample`로 쌓입니다. 진전이 없는 라운드가 3회 연속이면(`MAX_STALE = 3`) 무한루프를 피해 중단하고 수율 저조 경고를 남깁니다.

??? question "오개념 — “critique도 결국 같은 LLM이 하는데 의미가 있나?”"
    한계는 있지만 유효합니다. 같은 모델이라도 **역할·프롬프트·컨텍스트를 분리**합니다. 생성은 `GEN_SYSTEM`으로 seed chunk 전체를 받아 다양성을 만들고, critique는 `CRITIQUE_SYSTEM`("strict data-quality judge")으로 seed 상위 5건만 받아 후보 1건을 점수화합니다.
    완벽한 진리 검증은 아니지만 seed 도메인 이탈이나 라벨공간 위반 같은 **명백한 drift를 걸러내는 저비용 게이트**입니다.
    최종 품질 판정은 [held-out 평가](#held-out-규율--합성으로-평가-금지)가 담당합니다.

---

## 기본 경로 — 무의존성 bedrock_synth

!!! abstract "쉽게 말하면"
    "boto3로 [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)를 통해 Claude를 불러서 만들고, 같은 Claude로 검수하고, 정규식으로 PII/중복을 걸러 JSONL로 저장한다" — 이것이 전부입니다.
    외부 합성 라이브러리는 **일부러 사용하지 않습니다**. 노후화 리스크가 없고, IAM/VPC/Guardrails 같은 AWS 네이티브 거버넌스를 그대로 쓸 수 있기 때문입니다.

### 파이프라인 흐름

```
seed 샘플(증강 이전)
      │  seed 회전 (seeds_per_batch=4)
      ▼
[생성] Bedrock Converse  ── GEN_SYSTEM, JSON array, batch_size=5
      │  _extract_json (코드펜스/부분블록 방어 파싱)
      │  _as_text (output이 dict/list로 와도 문자열 정규화)
      ▼
[필터1] PII (email/phone/SSN/card 정규식) → 탈락
[필터2] 중복 (input+output sha256) → 탈락
      ▼
[critique] Bedrock Converse ── CRITIQUE_SYSTEM
      │  groundedness / relevance (0~1)
      ▼
  g ≥ min_groundedness AND r ≥ min_relevance ?
      │ yes
      ▼
to_messages() → 표준 messages (트랙별 어댑터, gemma_format.build_messages)
      ▼
save_jsonl → {"messages":[...]} JSONL  →  SFTTrainer conversational 입력
```

PII/중복 필터가 critique **앞**에 있는 순서가 중요합니다. 탈락이 확정된 후보에 critique 호출(= 토큰 비용)을 쓰지 않기 위한 배치입니다. 생성 배치와 critique는 모두 `ThreadPoolExecutor`로 병렬 처리됩니다 — Bedrock 호출이 I/O 바운드이기 때문입니다.

### 핵심 호출

```python
from common.synth import bedrock_synth as bs
synth = bs.generate_grounded(
    task_instruction=td.TASK_INSTRUCTION,      # 트랙별 (critique 기준으로도 쓰임)
    seed_texts=td.seed_texts_for_synth(seeds), # 증강 이전 seed
    n_total=NUM_SYNTH,                         # USER 결정값
    model_id=config.BEDROCK_CLAUDE_MODEL_ID,   # env 주입, 하드코딩 금지
    region=config.BEDROCK_REGION,
    to_messages=td.to_messages,                # {"input","output"} → messages
    max_workers=config.SYNTH_MAX_WORKERS,      # 동시 Bedrock 호출 수 (기본 10)
    accepted_ref=synth_ref,                    # 실시간 미리보기용 참조
    progress_cb=_preview,
)
```

- 저수준 호출은 `common/aws_utils.bedrock_converse()`가 담당합니다(boto3 `bedrock-runtime` 클라이언트의 [`converse()`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html)). 이는 SageMaker endpoint 호출(`sagemaker-runtime`의 `invoke_endpoint`)과 **별개 서비스**입니다 — [자주 나오는 오개념](#자주-나오는-오개념)에서 자세히 다룹니다.
- 출력은 `{"messages":[...]}` JSONL입니다. `gemma_format.build_messages`로 만든 표준 messages이므로 [TRL `SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer)의 conversational 포맷에 바로 들어갑니다(학습 경로는 `tracks/*/scripts/train.py`).
- 병렬 호출이 많으면 Bedrock throttling(429)이 납니다. boto3 클라이언트는 `mode="adaptive"` 재시도로 구성되어 있고 시도 횟수는 `BEDROCK_MAX_ATTEMPTS`(기본 8)로 조정합니다. 그래도 429가 계속되면 `SYNTH_MAX_WORKERS`를 낮추세요.

### max_tokens 실측값과 sampling 파라미터

두 가지 기본값은 Bedrock 호출을 실제로 돌려 보고 정해졌습니다.

- **`max_tokens`를 넉넉히 잡아야 합니다.** Claude Sonnet 5는 응답 전에 `reasoningContent`(추론 블록)에 토큰을 먼저 씁니다(실측 2026-07-31). 2048이면 추론만 하다 `stopReason=max_tokens`로 잘려 text 블록이 비거나 JSON이 중간에 끊깁니다("응답에 text 블록이 없습니다" / 파싱 실패). 그래서 생성은 `max(4096, 900 × batch_size)`, critique는 2048을 씁니다. `batch_size=5`면 생성 상한은 4,500 토큰입니다.
- **sampling 파라미터는 아예 보내지 않습니다.** Claude 4.x는 `temperature`/`top_p` 동시 지정이 불가하고, Claude 5+는 `temperature` 자체가 deprecated입니다. 지정하면 매 호출이 거부 후 폴백 재시도를 타 **호출 수가 2배**가 됩니다. 기본은 `maxTokens`만 보내고, 필요하면 `temperature` 또는 `top_p` 중 하나만 명시합니다(지정 시에도 `top_p` 우선, deprecated로 거부되면 조용히 제거 후 재시도).

### 생성 지시와 채점 기준의 분리

생성 난이도를 올리려고 제약("인자 2개 이상" 등)을 `task_instruction`에 섞으면, critique도 그 기준으로 채점해 seed와 다르다며 groundedness를 낮춰 **전부 기각**합니다(실측: 8건 생성 → 8건 기각).

그래서 `gen_instruction` 인자가 따로 있습니다. **생성만 어렵게 하고 채점은 원래 도메인 기준으로 두려면 제약을 `gen_instruction`에 넣고 `task_instruction`은 seed 도메인 그대로 유지**하세요.

??? question "오개념 — “PII 정규식이 순수 숫자열도 지우지 않나?”"
    그렇지 않습니다. phone/card 패턴은 **구분자(공백/`-`/괄호)를 요구**하도록 좁혀 두었습니다.
    따라서 function-call JSON에 흔한 타임스탬프·금액·id 같은 순수 긴 숫자열은 PII로 오탐하지 않습니다. 오탐 때문에 유효한 합성이 폐기되는 것을 막기 위한 설계입니다.

---

## 생성 건수 결정 — NUM_SYNTHETIC 기본값

- `config.NUM_SYNTHETIC`의 기본값은 **200/트랙**이고, env `NUM_SYNTHETIC`으로 오버라이드합니다. 이 킷의 `.env`는 **100**으로 낮춰 둔 상태입니다 — 요약 트랙(`03_summarization`)은 seed 문서가 커서 호출당 지연이 큽니다(실측: seed 1건 중앙 1,651자 vs 추출 트랙 475자 → 배치 프롬프트 약 10,900자). 잘림은 아니라 순수 지연입니다(출력 2,554 토큰 < `max_tokens` 4,500).
- seed 샘플 수는 `config.NUM_SEED_SAMPLES`(기본 300)로 따로 잡습니다. 합성 건수와 seed 건수는 별개 값입니다.
- **언제 늘리나요** — seed 다양성이 높고 도메인이 넓을 때, 그리고 held-out 지표가 데이터량에 비례해 오를 때 늘리세요.
- **언제 줄이나요** — Bedrock 비용이 부담될 때, 또는 seed가 좁아 다양성이 금방 포화될 때(중복 필터가 많이 걸립니다) 줄이세요.
- dry-run(`DRY_RUN=1`)은 파이프라인 검증용입니다. 노트북이 seed 8건 / 합성 6건 / `max_batches=3`으로 낮춰 잡으므로 실제 값과는 별개입니다.

??? question "오개념 — “합성을 많이 만들수록 항상 좋은가요?”"
    그렇지 않습니다. seed 다양성이 한계에 다다르면 추가 생성은 **중복·near-duplicate**만 늘리고(중복 필터가 폐기합니다) 비용만 증가시킵니다.
    양보다는 [held-out 지표](#held-out-규율--합성으로-평가-금지)로 검증한 유효 증가분을 기준으로 조정하세요.

---

## held-out 규율 — 합성으로 평가 금지

!!! abstract "쉽게 말하면"
    채점 시험지(held-out)를 **증강하기 전에 미리 떼어놓고**, 나머지만 합성으로 부풀립니다.
    합성 데이터는 teacher(Claude)를 모방한 것이므로, 합성으로 채점하면 "teacher를 얼마나 잘 베꼈나"만 재게 됩니다.

```
seed 전체
   ├─ [먼저 분리] held-out  ──────────────►  04_evaluate 전용 (합성 유입 0)
   └─ 나머지 seed  ──►  grounded 합성 증강  ──►  train.jsonl (SFTTrainer)
```

평가는 `04_evaluate.ipynb`에서 **held-out으로만** 수행합니다. 트랙별 메트릭은 추출이 `arg_f1`(primary) + `valid_json_rate` + `name_accuracy`, 분류가 macro-F1 + accuracy, 요약이 ROUGE-L + LLM-judge(groundedness/coverage), 도메인 QA가 LLM-judge(correctness/helpfulness/groundedness 1~5) + ROUGE-L proxy입니다(`common/eval_utils.py`).

!!! danger "합성/학습셋으로 평가하면 점수가 조용히 부풀려집니다"
    넉넉히 로드해서 뒤쪽 N건을 쓰는 방식(`pool[-N_EVAL:]`)은 위험합니다. `N_EVAL=50`이면 150건만 로드되어 held-out이 학습 구간(0~299) **안쪽**에 통째로 들어갑니다.
    그래서 `04_evaluate`는 `01_data_and_synthetic`이 학습에 쓴 앞 `NUM_SEED_SAMPLES`건(기본 300)을 **명시적으로 건너뛰고** 그 뒤 `N_EVAL`건(기본 50)을 씁니다.

`load_seed_examples`는 같은 인덱스를 항상 같은 순서로 돌려주므로(분류 트랙은 고정 시드 42로 셔플) 이 분리는 재현 가능합니다. 시드가 학습 구간보다 작으면 assert가 먼저 실패하므로, `NUM_SEED_SAMPLES`를 줄이거나 더 큰 시드 데이터셋을 쓰세요.

??? question "오개념 — “합성 데이터의 groundedness 점수가 높으면 평가에 써도 되나요?”"
    안 됩니다. groundedness는 "seed 도메인에 맞는가"를 나타낼 뿐 "정답인가"를 뜻하지 않습니다.
    게다가 합성은 정의상 teacher의 출력이므로, 이를 채점 기준으로 쓰면 **평가 순환 오류**가 그대로 남습니다. held-out은 반드시 실제 seed(증강 이전)에서만 뽑아야 합니다.

---

## 라이브러리 대안

기본 경로(`bedrock_synth.py`)만으로도 충분하지만, 오케스트레이션이나 대량 실행이 필요하다면 아래 대안을 붙일 수 있습니다. **버전·라이선스·유지보수 상태는 실행 전 재확인** 대상입니다.

| 도구 | Bedrock 연동 | 상태 (2026-07 실측) | 라이선스 | 쓸 때 |
|---|---|---|---|---|
| **`bedrock_synth.py` (이 킷)** | boto3 native | 킷 코드 | 킷 코드 | 기본값. 의존성 0, AWS 거버넌스 |
| **[Kiln](https://github.com/Kiln-AI/Kiln)** (`kiln-ai`) | ✅ 지원 — native (`ModelProviderName.amazon_bedrock`) | 가장 활발 (v1.0.4 @ 2026-07-16) | 확인 필요 — core lib MIT / repo 루트 커스텀 | GUI+오케스트레이션 원할 때 |
| **[Bespoke Curator](https://github.com/bespokelabsai/curator)** | LiteLLM 경유 (`bedrock/...`) | 활발 (0.1.29 @ 2026-07-13) | Apache-2.0 | 코드-우선, 대량·구조화·캐싱 |
| ~~[distilabel](https://github.com/argilla-io/distilabel)~~ | 해당 없음 | ❌ 정체 (마지막 v1.5.3 @ 2025-01-28, 2026 릴리스 0건) | 해당 없음 | 배제 — 사용 금지 |

- **Kiln** — native Bedrock 연동을 코드 수준에서 확인한 유일한 도구입니다. repo는 `github.com/Kiln-AI/Kiln`이며 `pip install kiln-ai`로 설치합니다. **리포 루트 라이선스와 core lib 라이선스가 다르므로** 재배포 전에 반드시 확인하세요.
- **Bespoke Curator** — native 커넥터는 아니고 LiteLLM을 경유합니다(`bedrock/<model>` + AWS 자격증명). repo는 `github.com/bespokelabsai/curator`입니다. 이 킷의 `common/llm_gateway.py`(LiteLLM)와 [Bedrock 라우팅 규약](https://docs.litellm.ai/docs/providers/bedrock)이 일치하므로 연결이 자연스럽습니다.
- 참고로 배제한 것들도 남겨 둡니다. **meta-llama/synthetic-data-kit**은 문서→QA/CoT 생성에 특화되었으나 케이던스가 둔화되고(2025-10 이후) Bedrock이 미문서화입니다. **Augmentoolkit**은 커밋은 활발하나 config/CLI 중심이라 라이브러리성이 낮고 Bedrock 미문서화입니다. **DeepFabric**(구 promptwright)은 LiteLLM을 쓰지 않아 Bedrock 미지원, **NVIDIA NeMo Curator**는 활발하지만 Bedrock 소비 커넥터가 없습니다(자체 NIM/vLLM 엔드포인트 호스팅용). **DataDreamer / fabricator**는 정체 상태입니다.
- 대안을 쓰더라도 **grounded + critique 원칙은 동일하게 적용**하고, 출력은 이 킷의 `messages` JSONL로 변환해 `train.py`에 넣으세요.

??? question "오개념 — “유명한 distilabel을 왜 안 쓰나요?”"
    유명세와 유지보수는 별개입니다. 실측 시점 기준 **2026년 릴리스가 0건**(마지막 2025-01)이라 프로덕션 의존성으로는 부적합합니다.
    이 킷은 "노후화 리스크 0"을 기본 원칙으로 삼습니다. repo 활동 상태는 변할 수 있으니 채택 전에 다시 확인하세요.

---

## 자주 나오는 오개념

??? question "오개념 — “합성 데이터 생성이 곧 SageMaker endpoint 호출 아닌가요?”"
    아닙니다. 합성 생성은 **Bedrock**(`bedrock-runtime`의 `converse`)으로 teacher LLM(Claude)을 부르는 것이고, 학습된 SLM 서빙은 **SageMaker**([`sagemaker-runtime`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html)의 `invoke_endpoint`, 스트리밍은 `invoke_endpoint_with_response_stream`)로 처리합니다.
    둘은 서로 다른 서비스이고, IAM 권한도 요금 체계도 따로입니다.

??? question "오개념 — “Bedrock Converse가 곧 SageMaker Batch Transform인가요?”"
    아닙니다. 합성 대량 생성은 Bedrock API를 반복 호출하는 작업입니다.
    [SageMaker 추론 4옵션](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)(Real-time / Serverless / Asynchronous / Batch Transform)은 **학습된 모델을 서빙**하는 이야기입니다. 특히 **Serverless는 GPU가 없어 LLM/SLM 서빙에 부적합**합니다(합성 생성과는 무관합니다) — 다만 Serverless의 GPU 지원 여부는 바뀔 수 있는 값이므로 링크에서 현행 스펙을 다시 확인하세요. 자세한 비교는 [SageMaker 추론](04_sagemaker_inference.md)을 참고하세요.

??? question "오개념 — “grounded면 seed를 그대로 복사하는 것 아닌가요?”"
    아닙니다. 생성 프롬프트가 "verbatim 복사 금지, 같은 도메인/스타일/라벨공간의 **새** 예시"를 명시적으로 요구하고, 중복 필터(sha256, 공백 정규화 + 소문자화)가 동일하거나 사실상 같은 복제를 걸러냅니다.
    seed는 어디까지나 근거일 뿐, 정답을 복사할 대상이 아닙니다.

??? question "오개념 — “critique 임계값 0.6은 고정값인가요?”"
    아닙니다. `min_groundedness`/`min_relevance` 인자로 트랙별 조정이 가능합니다.
    정밀한 라벨 태스크는 임계값을 올리고, 다양성이 중요한 QA는 낮출 수 있습니다. 다만 올릴수록 수율이 떨어져 같은 `n_total`을 채우는 데 호출이 늘어납니다.

---

## 비용과 cleanup

!!! danger "비용과 cleanup"
    Bedrock은 [입력·출력 토큰 과금](https://aws.amazon.com/bedrock/pricing/)이고, 이 킷은 예시당 **생성 1회 + critique 1회**를 호출합니다. `NUM_SYNTHETIC`이 크고 seed chunk가 길수록 토큰이 선형으로 늘어납니다.
    대량 생성 전에 **소량으로 단가를 측정한 뒤 총량을 추정**하세요. 기각된 후보의 생성 토큰도 이미 과금된 상태입니다.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| Bedrock Converse (생성 + critique) | 입력·출력 토큰당 과금, 호출 시에만 발생 | 상시 리소스 없음. `NUM_SYNTHETIC`으로 총량 제어 |
| SageMaker 학습 잡 | 인스턴스 시간당 과금, 잡 종료 시 자동 중지 | 자동 종료. 실패 잡은 콘솔에서 확인 |
| SageMaker endpoint | 인스턴스 시간당 **상시** 과금 | `99_cleanup.ipynb`로 반드시 삭제 |
| S3 (합성 JSONL, 모델 아티팩트) | 저장 용량당 과금 | `99_cleanup.ipynb` |

합성 생성 자체는 상시 리소스를 남기지 않습니다. 다만 같은 노트북 흐름의 endpoint와 학습 잡은 별개이므로 `99_cleanup.ipynb`로 정리하세요.

**모델 ID는 env로 주입하고 하드코딩하지 마세요.** `BEDROCK_CLAUDE_MODEL_ID`에는 [inference profile](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html) prefix(`us.`/`eu.`/`apac.`/`global.`)가 필수입니다. 킷 기본값은 `global.anthropic.claude-sonnet-5`(이 계정 `list_inference_profiles` 실측 2026-07)이며, 최신(5+) Claude는 dateless pinned-snapshot 형식일 수 있습니다. 모델 로스터는 자주 바뀌므로 다른 계정/리전에서는 Bedrock 콘솔에서 현행 ID를 확인한 뒤 env로 넣으세요.

AWS 마케팅 수치("최대 N% 절감" 등)는 **AWS 주장**으로 표기하고 출처를 붙이세요(이 문서에서는 인용하지 않습니다).
