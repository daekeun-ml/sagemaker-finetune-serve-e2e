# 요약 트랙 — 미국 법안 원문 → 사람이 읽을 요약

!!! info "Scope"
    긴 문서 하나를 넣으면 짧은 요약이 나오는 SLM을 만들고 싶은 분을 위한 트랙 소개입니다(`tracks/03_summarization`).
    선행 조건은 AWS 자격증명과 SageMaker 실행 role뿐입니다(`00_setup`이 확인합니다). SageMaker가 처음이면 [SageMaker 기초](01_sagemaker_basics.md)를 먼저 읽으세요.
    다루는 것은 이 트랙의 task 정의·시드 데이터셋·성공 기준·노트북 구성·트랙별 설정값입니다. 학습 방식 자체는 [파인튜닝](03_finetuning.md), 배포·서빙은 [SageMaker 추론](04_sagemaker_inference.md), 완주 절차는 [실행 runbook](RUN_E2E.md)이 담당하므로 여기서 반복하지 않습니다.
    **다루지 않는 것**: 대화체 요약(회의록·상담 로그)은 시드를 바꿔야 하고(아래 시드 절 참고), 자유서술 답변은 [도메인 QA 트랙](track_domain_qa.md)입니다.

이 트랙과 관련된 리포지토리 파일:

- `tracks/03_summarization/track_data.py` — 시드 로드, `{input, output}` 어댑터, `SYSTEM_PROMPT`
- `tracks/03_summarization/*.ipynb` — 이 트랙의 노트북 9개
- `common/config.py` — `TRACKS['summarization']` 레지스트리(시드 데이터셋, `max_seq_length=2048`)
- `common/eval_utils.py` — `eval_rouge()` + `llm_judge()`
- `tracks/build_all_tracks.py` — 이 트랙의 `TrackSpec`(엔드포인트 prefix, 서빙·생성 길이, GRPO reward 종류)

---

## 이 트랙이 푸는 문제

!!! abstract "쉽게 말하면"
    "이 긴 문서 요약해 줘"를 **전담하는 작은 모델**을 만듭니다. 입력은 문서 본문 하나, 출력은 요약 문장들뿐입니다. 라벨도 카테고리도 JSON 스키마도 없습니다.
    학습 데이터가 미국 연방 법안이므로 결과 모델의 어투는 "법안 요지 브리핑"에 가깝습니다. 사내 규정·계약서·보고서처럼 **형식이 정해진 긴 문서**를 다루는 작업에 가장 잘 맞고, 대화 로그 요약이나 회의록 같은 대화체에는 그대로 맞지 않습니다(아래 시드 절 참고).

`track_data.py`의 어댑터는 한 row를 두 필드로만 줄입니다 — 원본 `text` → `input`, 원본 `summary` → `output`. 실제 `data/train.jsonl` 첫 예시입니다(발췌).

```text
# 원본 row (FiscalNote/billsum)
title:   A bill to limit the civil liability of business entities providing use of facilities...
text:    SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES
             TO NONPROFIT ORGANIZATIONS.
             (a) Definitions.--In this section:
             (1) Business entity.--The term ``business entity'' means a firm, corporation, ...
summary: Shields a business entity from civil liability relating to any injury or death
         occurring at a facility of that entity in connection with a use of such facility ...

# to_messages() 변환 후 — user 1턴 + assistant 1턴
user:      "You are a precise document summarizer. ... Output ONLY the summary.
            Summarize the following document:
            SECTION 1. LIABILITY OF BUSINESS ENTITIES ..."      ← 5,237자(고정 접두 211자 + 본문 5,026자)
assistant: "Shields a business entity from civil liability ..."  ← 1,561자
```

`SYSTEM_PROMPT`가 system 턴이 아니라 **첫 user 턴 앞에 병합(fold)** 되어 있는 것이 눈에 걸릴 수 있습니다. Gemma chat template이 system role을 거부하기 때문이며(`track_data.py`의 주석과 `common/gemma_format.fold_system_into_user`), 5개 트랙이 모두 같은 방식입니다.

!!! warning "학습 프롬프트와 추론 프롬프트가 정확히 같지 않습니다"
    학습 시 user 턴의 고정 접두는 `SYSTEM_PROMPT`(174자)만이 아닙니다. `to_messages()`가 그 뒤에 `Summarize the following document:` 한 줄을 **하드코딩으로 끼워** 총 211자를 붙입니다(위 예시 블록의 두 번째 줄이 그것입니다).

    ```python
    # tracks/03_summarization/track_data.py
    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nSummarize the following document:\n\n{example['input']}"}
    ```

    반면 배포 후 호출부는 이 한 줄을 보내지 않습니다 — `04_evaluate`·`05_agentic_strands`는 `gf.build_inference_messages(ex['input'], system_content=td.SYSTEM_PROMPT)`로 **system + 본문**만 보내고, `03_deploy_endpoint`는 `f'{td.SYSTEM_PROMPT}\n\n{user}'`를 씁니다. 지시문은 `SYSTEM_PROMPT`에 이미 들어 있어 실무상 큰 문제는 아니지만, "학습과 완전히 동일한 프롬프트"는 아니라는 뜻입니다. 파인튜닝 효과를 최대로 재려면 호출 시에도 같은 한 줄을 붙여 A/B로 비교해 보세요.

    `02b_local_serve`의 예시 입력(`serve_example_user`)은 본문만, `03_deploy_endpoint`의 스모크 입력(`deploy_smoke_user`)은 같은 문서에 `"Summarize: "`를 붙인 형태라 두 표면형의 응답 차이를 `02b`의 3-D 셀에서 A/B로 볼 수 있습니다(`tracks/build_all_tracks.py`).

---

## 시드 데이터셋

[`FiscalNote/billsum`](https://huggingface.co/datasets/FiscalNote/billsum) — **cc0-1.0(public domain), ungated**라 HF 토큰이 필요 없습니다. `text`(법안 본문) + `summary`(사람이 쓴 요약) + `title` 구조이고, `load_seed_examples()`는 `train` 스플릿을 앞에서부터 순서대로 읽어 `input`/`output`이 모두 비어 있지 않은 것만 담습니다(셔플 없음 → 같은 인덱스가 항상 같은 예시).

이 트랙에서 실제로 물리는 지점은 **길이**입니다.

- `track_data.MAX_DOC_CHARS = 6000` — 법안 본문이 매우 길 수 있어 문자 단위로 자릅니다. 리포에 커밋된 `data/train.jsonl`의 시드 300건 중 **260건(87%)이 이 상한에 걸려** user 턴이 정확히 6,211자(고정 접두 211자 + 본문 6,000자)입니다. 즉 이 트랙의 학습 입력 길이는 데이터가 정하는 게 아니라 **이 상수가 정합니다.** 위에 인용한 row 0은 본문이 5,026자라 상한에 닿지 않은 나머지 13% 쪽이고, 상한에 걸리는 첫 행은 index 1입니다.
- 정답 요약 길이는 시드 중앙 1,110자(최대 4,950자)인데, 합성으로 만든 200건은 중앙 515자(최대 850자)로 더 짧습니다. 합성이 시드보다 짧고 균질해지는 경향이 있으니 `01` 노트북의 EDA 표를 그냥 넘기지 마세요.
- 합성 단계가 다른 트랙보다 **느립니다.** 시드 1건이 중앙 1,651자(추출 트랙은 475자)라 배치 프롬프트가 약 10,900자가 됩니다 — 잘림이 아니라 순수 지연이며, 그래서 이 kit의 `.env`는 `NUM_SYNTHETIC=100`으로 낮춰 두었습니다([생성 건수 결정](02_synthetic_data.md#생성-건수-결정--num_synthetic-기본값)).

!!! warning "법안 원문의 이중 백틱이 노트북 출력을 삼킵니다"
    billsum은 구식 인용부호로 ``` ``and'' ``` 처럼 **이중 백틱**을 씁니다. 실측된 한 held-out 입력에는 백틱이 **79개** 있었고, 이 텍스트를 노트북 마크다운에 그대로 넣자 인라인 코드스팬이 열려 뒤따르는 예측 블록까지 사라졌습니다(응답 1,596자를 정상 수신한 상태였는데도 화면에 아무것도 없었습니다).
    `common/display_utils.py`가 입력·예측을 모두 `<pre>`로 감싸 막아 두었습니다. 직접 셀을 쓸 때도 같은 규칙을 지키세요 — 자세한 재현은 [입력 텍스트가 노트북 마크다운을 깨뜨릴 때](05_serving_containers.md#입력-텍스트가-노트북-마크다운을-깨뜨릴-때)에 있습니다.

대화체 요약(회의록·상담 로그)을 원한다면 시드를 바꿔야 합니다. 완전 permissive한 대화 요약 공개셋이 마땅치 않아 이 kit은 **문서 요약 시드 + grounded 합성**으로 확장하는 쪽을 택했습니다(`track_data.py` 독스트링).

---

## 성공 기준

`eval_kind="summarization"`이라 `04_evaluate`는 두 지표를 함께 냅니다.

| 지표 | 구현 | 무엇을 잡나 | 무엇을 못 잡나 |
|---|---|---|---|
| **ROUGE-L**(primary) | `eval_utils.eval_rouge()` — `rouge_score`, stemmer 사용, rouge1/2/L의 F-measure 평균 | 참조 요약과의 표면적 겹침. 값이 싸고 결정론적 | ❌ 원문에 없는 사실을 넣었는지(faithfulness) |
| **LLM-judge** | `eval_utils.llm_judge()` — Bedrock Converse, `groundedness`/`coverage` 각 1–5점 | 원문 근거 여부와 핵심 누락 | ❌ 무료가 아님 — 호출 과금, judge 편향 |

요약은 정답이 하나가 아니어서 exact match가 성립하지 않고, 반대로 ROUGE만 보면 "원문 문구를 많이 베낀 요약"이 유리해집니다. 그래서 자동 지표를 주 지표로 두고 judge를 보완으로 붙이는 조합입니다. judge는 비용 때문에 held-out **앞 20건**에만 돌리고, 프롬프트에 넣는 원문은 4,000자로 잘립니다(`eval_utils.py`).

held-out은 `01`이 학습에 쓴 앞 `NUM_SEED_SAMPLES`건(기본 300)을 **명시적으로 건너뛴 뒤** `N_EVAL`건(기본 50, `DRY_RUN`이면 20)을 잘라 씁니다. 합성 데이터로 평가하면 teacher 모방을 재는 것이 되므로 금지입니다 — 규율의 배경은 [held-out 규율](02_synthetic_data.md#held-out-규율--합성으로-평가-금지)에 있습니다. 평가 전용 문서는 없고 구현이 곧 명세입니다(`common/eval_utils.py` + 각 트랙 `04_evaluate`).

---

## 노트북 순서

이 트랙의 노트북은 **9개**입니다. 텍스트 트랙 표준 세트에서 `02a_train_grpo_sagemaker`만 빠져 있습니다(`02a`를 갖는 것은 추출·분류 트랙 둘뿐이라 그 두 트랙만 10개입니다).

| 노트북 | 산출물 |
|---|---|
| `00_setup` | 자격증명·리전·role 확인, 의존성 설치 |
| `01_data_and_synthetic` | 시드 300건 + 합성 → `data/train.jsonl`(리포 커밋본은 500건), 토큰 길이 EDA |
| `02_train_sft_sagemaker` | SageMaker Training Job(`scripts/train.py`, QLoRA) → 머지된 모델 아티팩트(S3), `%store md_summarization` |
| `02b_local_serve` | **(선택)** 내 GPU의 vLLM로 프리플라이트 — 배포 5~15분 왕복을 줄입니다 |
| `03_deploy_endpoint` | `gemma-summarization-vllm-<timestamp>` real-time endpoint + invoke 스모크. `%store ep_summarization` |
| `04_evaluate` | held-out ROUGE-L + LLM-judge 점수 |
| `05_agentic_strands` | `summarize_document` tool을 가진 Strands 에이전트(reasoning은 Bedrock Claude) |
| `06_agentcore_deploy` | AgentCore Runtime 배포 |
| `99_cleanup` | endpoint → endpoint-config → model 삭제, 로컬 모델 정리 |

??? question "오개념 — “GRPO 노트북이 빠진 건 미완성이라서인가요?”"
    아닙니다. 의도적으로 없습니다. GRPO에는 **프로그램으로 채점되는 reward**가 필요하고 `scripts/train_grpo.py --reward_kind`가 받는 값은 `extraction`과 `classification` 둘뿐입니다. "좋은 요약"은 규칙으로 채점할 수 없어서 이 트랙의 `TrackSpec`은 `grpo_reward_kind`가 비어 있고, 그러면 빌더가 `02a`를 생성하지 않습니다.
    LLM-judge를 reward로 쓸 수는 있지만 rollout마다 judge를 호출해야 해 비용·시간이 급증하고 judge 편향이 학습에 섞입니다 — 판단 근거는 [왜 추출·분류 트랙에만 GRPO가 있나](03_finetuning.md#왜-추출분류-트랙에만-grpo가-있나)에 있습니다.

!!! tip "먼저 DRY_RUN=1로 한 바퀴 도세요"
    `DRY_RUN=1`이면 시드 8건 · 합성 6건 · held-out 20건으로 줄어들어 파이프라인 형태만 빠르게 검증합니다. 단계별 핸드오프와 비용 가드는 [실행 runbook](RUN_E2E.md#단계별-실행과-데이터-핸드오프)에 정리돼 있습니다.

---

## 트랙별 설정값

다른 트랙과 다른 값만 모았습니다. 값의 출처는 `common/config.py`의 `TRACKS`와 `tracks/build_all_tracks.py`의 `TrackSpec`입니다. 입력이 길고 정답도 길어서 **서빙 컨텍스트와 생성 길이를 학습 길이와 분리**해 둔 트랙이고(도메인 QA 트랙도 같은 이유로 분리합니다), `serve_max_model_len`은 이 kit에서 가장 큰 값입니다(생성 상한만은 멀티모달 트랙의 768이 더 큽니다 — 정답 JSON이 592토큰까지 가기 때문입니다).

| 설정 | 이 트랙 | 다른 텍스트 트랙 | 근거 |
|---|---|---|---|
| `max_seq_length` | 2048 | 추출 2048 / 분류 512 / QA 1024 | 학습 시 "입력+정답"이 들어가야 하는 길이 |
| `serve_max_model_len` | **4096** | QA 2048 / 추출·분류는 미지정(= `max_seq_length × 2`) | held-out 프롬프트 median 1,370 / **max 2,006 토큰**이라 학습값 2048로는 부족 — 두 값을 분리하는 이유는 [학습 길이와 서빙 길이](00_overview.md#학습-길이와-서빙-길이는-다른-값입니다) |
| `gen_max_tokens` | **512** | QA 512 / 추출·분류 256 | 정답 요약 median 209 / p90 475 / max 964 토큰. 256이면 held-out **40%(20/50건)** 가 잘려 ROUGE-L이 구조적으로 과소 측정 |
| `grpo_reward_kind` | (빈 문자열) | 추출·분류만 값 있음 | 프로그램적 reward 불가 → `02a` 노트북이 생성되지 않습니다 |
| `eval_kind` | `summarization` | 분류 `classification` | `04_evaluate`가 `eval_rouge()` + `llm_judge()`를 부르고, 실시간 추론 셀의 **스트리밍이 기본 on**이 됩니다 |
| `endpoint_prefix` | `gemma-summarization` | 분류 `gemma-classification` | 학습 잡·endpoint 이름과 `%store` 키(`ep_summarization`·`md_summarization`)의 접두어 |
| `multimodal` | `False` | 05만 `True` | 텍스트 전용이라는 표식입니다(레지스트리 기본값). 노트북 세트를 정하는 것은 이 값이 아니라 빌더이며, 이 트랙은 `01_data_and_synthetic`을 씁니다 |

`gen_max_tokens=512`의 근거는 endpoint 실측입니다(입력 5,996자) — `max_tokens=256`은 `finish_reason='length'`로 902자에서 문장 중간에 끊겼고, 512는 `stop`으로 397토큰·1,446자, 1024는 `stop`으로 571토큰이었습니다. 즉 512부터 모델이 스스로 종료합니다. 정답 전량(max 964토큰)까지 덮으려면 1024로 올리면 됩니다. 절단은 예외도 경고도 없이 200 응답으로 오므로 [max_tokens 절단과 finish_reason](05_serving_containers.md#max_tokens-절단과-finish_reason)의 `finish_reason` 확인 습관을 권합니다.

이 트랙은 응답이 긴 자유서술이라 실시간 추론 셀에서 **스트리밍이 기본으로 켜져** 있습니다(`_stream_default()`가 `eval_kind`로 판단). 실측에서 첫 응답 0.42초 vs 완성 대기 16.16초로 체감이 38배 좋아지지만, 완료 시각은 15.9초 vs 16.2초로 사실상 같습니다 — 총 생성 시간과 처리량은 개선되지 않습니다([응답 스트리밍](04_sagemaker_inference.md#응답-스트리밍--invoke_endpoint_with_response_stream)).

!!! warning "이 트랙에서 먼저 터진 두 가지"
    (1) **학습 잡이 머지 도중 잘렸습니다.** `gemma-summarization-train-20260731084146`(`ml.g6.2xlarge`)이 189/189 step을 다 끝낸 뒤 1시간 기본 한도에 걸려 `Stopped`로 종료됐고, 아티팩트에 어댑터만 남아 배포가 불가능했습니다. seq 2048은 약 17s/step이라 이 트랙이 가장 먼저 한도에 부딪힙니다 → `stopping_condition`을 반드시 명시하세요([MaxRuntimeExceeded 함정](03_finetuning.md#maxruntimeexceeded--학습-뒤-머지에서-잘리는-함정)).
    (2) **엉뚱한 endpoint를 불렀습니다.** `%store`의 전역 `endpoint_name`이 다른 트랙 값으로 덮여, 요약 노트북이 멀티모달 endpoint(`max_model_len=2048`)를 호출해 "maximum context length is 2048" 400 에러가 났습니다 — 요약 endpoint는 4096이라 정상인데도입니다. 그래서 이 트랙은 `ep_summarization` 키를 우선 사용합니다([%store 전역 오염](05_serving_containers.md#store-전역-오염--엉뚱한-endpoint-호출)).

---

## 이어서 볼 문서

- [00 전체 지도](00_overview.md#5개-독립-트랙과-공통-레이어) — 5개 트랙 비교와 공통 `common/` 레이어
- [02 합성 데이터](02_synthetic_data.md) — grounded 합성과 held-out 규율
- [03 파인튜닝](03_finetuning.md) — LoRA/QLoRA, Gemma 관용구, 머지·re-export
- [04 SageMaker 추론](04_sagemaker_inference.md) — endpoint 3층 구조와 호출 스키마
- [05 서빙 컨테이너](05_serving_containers.md) — 엔진 선택, OOM·절단 실측 함정
- [실행 runbook](RUN_E2E.md#단계별-실행과-데이터-핸드오프) — 단계별 실행 순서, 비용 가드, 완료 기준

!!! danger "비용과 cleanup"
    학습 잡은 실행 시간만큼 과금되고 **endpoint는 호출하지 않아도 삭제할 때까지 시간당 계속 과금**됩니다. 트랙을 마쳤으면 `99_cleanup`을 반드시 실행해 endpoint·endpoint-config·model을 모두 지우세요.
    합성 데이터 생성과 `04_evaluate`의 LLM-judge, `05_agentic_strands`는 Bedrock 호출로 별도 과금됩니다(상주 리소스는 없습니다). 삭제 순서와 잔여 리소스 확인은 [비용과 cleanup](04_sagemaker_inference.md#비용과-cleanup)에 있습니다.
