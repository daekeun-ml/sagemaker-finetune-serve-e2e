# 도메인 QA 코스: 질문(+참고 문서)에서 자유서술 답변

!!! info "Scope"
    자유서술형 답변을 내는 모델을 만들려는 분을 위한 코스입니다(`tracks/04_domain_qa`).
    "질문(+참고 문서)을 주면 사람이 쓴 것 같은 답을 돌려준다"가 목표라면 이 코스가 맞습니다.

    - **선행 조건**: AWS 자격증명과 Amazon SageMaker AI 실행 role (`00_setup`이 확인).
      SageMaker AI가 처음이면 [SageMaker AI 기초](../01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: task 정의, 시드 데이터셋, 성공 기준, 노트북 구성, 코스별 설정값
    - **여기서 다루지 않는 것**: 학습 방식은 [파인튜닝](../03_finetuning.md),
      배포와 서빙은 [SageMaker AI 추론](../04_sagemaker_inference.md), 전체 실행 절차는
      [E2E 실행 가이드](../RUN_E2E.md)
    - **다른 코스**: 긴 문서 요약은 [요약](summarization.md).
      검색기를 붙이는 RAG는 이 코스 위에 얹는 별도 작업입니다(아래 "오해" 노트 참고)

이 코스와 관련된 파일입니다. 디렉터리 이름 `tracks/`와 `TRACKS`, `track_data.py` 같은 식별자는 초기 이름을 유지합니다.

- `tracks/04_domain_qa/track_data.py`: 시드 로드, `{input, output}` 어댑터, `SYSTEM_PROMPT`
- `tracks/04_domain_qa/*.ipynb`: 이 코스의 노트북 9개
- `common/config.py`: `TRACKS['domain_qa']` 레지스트리(시드 데이터셋, `max_seq_length=1024`)
- `common/eval_utils.py`: `eval_rouge()` + `llm_judge()`
- `tracks/build_all_tracks.py`: 이 코스의 `TrackSpec`(endpoint prefix, serving과 생성 길이, GRPO reward 종류)

---

## 이 코스가 푸는 문제

!!! abstract "입력과 출력"
    "이 문서 읽고 내 질문에 답해 줘"를 잘하는 모델을 만드는 코스입니다. 정답 형태가 JSON도 라벨도 아니고 **자유로운 문장**이라, 채점도 문자열 비교가 아니라 심사(judge)로 합니다.

입력은 **instruction 하나**이고, 참고할 문서가 있으면 그 뒤에 `[Context]` 헤더로 붙습니다. 출력은 답변 텍스트 그대로입니다. `track_data.py`의 `_compose_input()`이 만드는 표면형은 정확히 이렇습니다.

원본 row (dolly 첫 번째 예시, 실측):

```text
instruction: When did Virgin Australia start operating?
context:     Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an
             Australian-based airline. ... It commenced services on 31 August 2000 as Virgin Blue,
             with two aircraft on a single route. ...
response:    Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two
             aircraft on a single route.
category:    closed_qa
```

어댑터를 통과한 뒤 (`load_seed_examples()` 반환값):

```text
input:  When did Virgin Australia start operating?

        [Context]
        Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based
        airline. ... It commenced services on 31 August 2000 as Virgin Blue, ...

output: Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a
        single route.
```

`context`가 빈 문자열인 row는 헤더 없이 instruction만 `input`이 됩니다. 실제로 다음과 같은 짧은 예시가 그대로 학습셋에 들어갑니다.

```text
input:  Which is a species of fish? Tope or Rope
output: Tope
```

학습 시점에는 `to_messages()`가 이 쌍을 2턴 `messages`로 바꿉니다. 이때 `SYSTEM_PROMPT`("You are a helpful domain assistant. Answer the user's instruction. If context is provided, ground your answer in it and do not contradict it.")는 **system role이 아니라 첫 user 턴 앞에 병합**됩니다. Gemma chat template이 system role을 거부하기 때문입니다([chat template과 system fold](../03_finetuning.md#chat-template과-system-fold)).

??? question "오해: “context가 있으니 RAG 코스 아닌가요?”"
    아닙니다. 이 코스는 **검색 단계가 없습니다.** context는 데이터셋이 이미 붙여 준 문단이고, 모델이 배우는 것은 "주어진 문단에 근거해 답하기"입니다. 검색기를 붙이는 것은 이 코스 위에 얹는 별도 작업이며, 이 프로젝트의 노트북에는 포함되지 않았습니다.

---

## 시드 데이터셋

시드는 [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)(cc-by-sa-3.0, ungated)입니다. Databricks 직원들이 직접 작성한 instruction-following 셋으로, 컬럼은 `instruction` / `context` / `response` / `category` 넷입니다.

실측 로드 결과(`train` 스플릿, 15,011건):

| 항목 | 실측값 |
|---|---|
| 전체 건수 | 15,011 |
| 스플릿 | `train` 하나뿐 (test/validation 없음) |
| `context`가 채워진 row | 4,467건 (29.8%): 나머지 70%는 문서 없이 답해야 하는 질문 |
| `category` 분포 | `open_qa` 3,742, `general_qa` 2,191, `classification` 2,136, `closed_qa` 1,773, `brainstorming` 1,766, `information_extraction` 1,506, `summarization` 1,188, `creative_writing` 709 |

`load_seed_examples(n)`은 앞에서부터 순서대로 훑으며 `instruction`과 `response`가 **둘 다 비어 있지 않은** row만 채택하고, `n`건이 모이면 멈춥니다. 셔플하지 않습니다. 이 데이터셋은 라벨 정렬 순서가 아니고 category가 앞부분에도 섞여 있어, 분류 코스처럼 셔플을 강제할 이유가 없습니다.

!!! warning "스플릿이 하나뿐이라 held-out을 직접 잘라야 합니다"
    dolly에는 test 스플릿이 없습니다. 그래서 `04_evaluate`는 학습이 사용한 **앞 `NUM_SEED_SAMPLES`건(기본 300)을 명시적으로 건너뛰고** 그 뒤 `N_EVAL`건(기본 50, `DRY_RUN`이면 20)을 held-out으로 씁니다. `load_seed_examples`가 항상 같은 순서를 돌려주므로 이 분리는 재현 가능합니다.
    `pool[-N_EVAL:]` 같은 방식은 쓰지 않습니다. 넉넉히 로드하지 않으면 held-out이 학습 구간(0~299) **안쪽**에 들어가 점수가 부풀려집니다. 배경은 [held-out 규율](../02_synthetic_data.md#held-out-규율-합성으로-평가-금지)에 있습니다.

!!! danger "CC-BY-SA는 파생물로 전파됩니다"
    dolly는 cc-by-sa-3.0(share-alike)입니다. 이 데이터로 학습한 어댑터와 머지 모델과 합성 데이터를 **배포**할 때 share-alike 의무가 따라붙습니다. 사내 실습으로 끝낼 것인지, 외부 배포까지 갈 것인지에 따라 시드를 다시 고르는 편이 나을 수 있습니다(이 프로젝트의 다른 코스 시드는 apache-2.0 / mit / cc0-1.0 / cc-by-4.0입니다).

---

## 성공 기준

주 지표는 **Bedrock LLM-judge**이고, ROUGE-L은 보조 proxy입니다.

- **LLM-judge (primary)**: `eval_utils.llm_judge()`가 Bedrock Converse로 `correctness`, `helpfulness`, `groundedness`를 각 1~5점으로 채점합니다. judge 모델은 `config.BEDROCK_CLAUDE_MODEL_ID`(기본 `global.anthropic.claude-sonnet-5`)이고, `temperature=0.0`, STRICT JSON 출력으로 고정합니다. 비용 때문에 held-out **앞 20건만** 채점합니다.
- **ROUGE-L (보조)**: `eval_utils.eval_rouge()`가 held-out 전량에 대해 rouge1/rouge2/rougeL F-measure 평균을 냅니다.

**왜 judge가 주 지표인가**: 이 코스의 정답은 자유형 문장입니다. "31 August 2000"과 "August 31, 2000"은 같은 정답인데 exact-match는 0점을 주고, ROUGE-L도 표현이 다르면 정답을 오답처럼 깎습니다. 반대로 원문 단어만 잔뜩 베껴 온 무의미한 답이 ROUGE는 높게 나옵니다. 그래서 정확성, 유용성, 근거성을 각각 보는 judge를 주 지표로 두고, ROUGE-L은 "judge 호출 없이도 회귀를 감지하는 값싼 센서"로만 씁니다.

추출 코스의 `arg_f1`이나 분류 코스의 macro-F1처럼 순수 python으로 채점 가능한 지표가 이 코스에는 없다는 점이, 뒤에 나오는 GRPO 부재와 같은 원인입니다.

---

## 노트북 순서

이 코스의 노트북은 **9개**입니다. 텍스트 코스의 표준 세트와 같고, **`02a_train_grpo_sagemaker`만 없습니다**(`02a`를 갖는 것은 추출과 분류 코스 둘뿐이라 그 두 코스만 10개입니다).

| 노트북 | 결과 |
|---|---|
| `00_setup` | 자격증명, 리전, role 확인, 의존성 설치 |
| `01_data_and_synthetic` | dolly 시드 300건 + grounded 합성 → `data/train.jsonl`(`messages` 포맷) |
| `02_train_sft_sagemaker` | `scripts/train.py`를 SageMaker AI 학습 Job으로 실행 → 머지된 모델 artifact(S3), `%store md_domain_qa` |
| `02b_local_serve` | **(선택)** 로컬 vLLM으로 preflight: 클라우드 배포 전 30초 검증 |
| `03_deploy_endpoint` | `gemma-domainqa-vllm-<timestamp>` real-time endpoint + invoke smoke test. `%store ep_domain_qa` |
| `04_evaluate` | held-out ROUGE-L + LLM-judge 점수 |
| `05_agentic_strands` | `answer_domain_question` tool을 가진 Strands 에이전트(reasoning은 Bedrock Claude) |
| `06_agentcore_deploy` | AgentCore Runtime 배포 |
| `99_cleanup` | endpoint → endpoint-config → model 삭제, 로컬 모델 정리 |

**`02a`(GRPO)가 없는 이유**는 reward를 프로그램으로 계산할 수 없기 때문입니다. `scripts/train_grpo.py`의 `--reward_kind`가 받는 값은 `extraction`과 `classification` 둘뿐이고, "좋은 답변"을 규칙으로 채점할 방법이 없습니다. LLM-judge를 reward로 쓰는 것은 가능하지만 rollout마다 judge를 호출해야 해 비용과 시간이 급증하고 judge 편향이 학습에 섞입니다. 판단 근거는 [왜 추출과 분류 코스에만 GRPO가 있나](../03_finetuning.md#왜-추출과-분류-코스에만-grpo가-있나)에 있습니다.

!!! tip "먼저 DRY_RUN=1로 데이터 흐름을 확인하세요"
    `DRY_RUN=1`이면 시드 8건, 합성 6건, held-out 20건으로 줄어듭니다. SageMaker AI Training Job의 규모는 자동으로 줄지 않으므로 `MAX_TRAIN_SAMPLES`와 `EPOCHS`도 직접 조정하세요. 단계별 핸드오프와 비용은 [E2E 실행 가이드](../RUN_E2E.md)에 정리돼 있습니다.

---

## 코스별 설정값

다른 코스와 다른 값만 모았습니다. 값의 출처는 `common/config.py`의 `TRACKS`와 `tracks/build_all_tracks.py`의 `TrackSpec`이고, 길이 근거는 시드 150건을 gemma-4 E4B 토크나이저로 실측한 분포입니다.

| 설정 | 이 코스 | 다른 텍스트 코스 | 근거 |
|---|---|---|---|
| `max_seq_length` | **1024** | 추출, 요약 2048 / 분류 512 | 학습 전체(입력+정답) median 141 / p90 420 / max 1,945 → 1024면 4건(2.7%)만 절단. 대부분이 훨씬 짧아 더 키우면 메모리만 낭비합니다 |
| `serve_max_model_len` | **2048** | 요약 4096 / 추출, 분류는 미지정(= `max_seq_length × 2`) | 추론 프롬프트 median 58 / p90 291 / max 1,140 → 프롬프트 1,140 + 생성 512에 여유(1024로 두면 1건이 초과): 두 값을 분리하는 이유는 [학습 길이와 서빙 길이](../00_overview.md#학습-길이와-서빙-길이는-다른-값입니다) |
| `gen_max_tokens` | **512** | 요약 512 / 추출, 분류 256 | 정답 median 39 / p90 218 / **max 1,781**. 256으로 두면 정답 13건(150건 중 **8.7%**)이 잘려 ROUGE와 judge 점수가 구조적으로 과소 측정됩니다 |
| `grpo_reward_kind` | (빈 문자열) | 추출, 분류만 값 있음 | 프로그램적 reward 불가 → `02a` 노트북이 생성되지 않습니다 |
| `eval_kind` | `domain_qa` | 분류 `classification` | `04_evaluate`가 `llm_judge()` + `eval_rouge()`를 부르고, 실시간 추론 셀의 **스트리밍이 기본 on**이 됩니다 |
| `endpoint_prefix` | `gemma-domainqa` | 분류 `gemma-classification` | 학습 Job, endpoint 이름과 `%store` 키(`ep_domain_qa`, `md_domain_qa`)의 접두어 |
| `multimodal` | `False` | 05만 `True` | 텍스트 전용이라는 표식입니다(레지스트리 기본값). 노트북 세트를 정하는 것은 이 값이 아니라 빌더이며, 이 코스는 `01_data_and_synthetic`을 씁니다 |

스트리밍이 기본 on인 이유는 답변이 긴 자유서술이라 첫 토큰을 먼저 보여 주는 편이 체감상 훨씬 낫기 때문입니다(짧은 JSON, 라벨을 내는 추출과 분류 코스에서는 끕니다). 단 첫 토큰 체감만 줄고 전체 생성 시간이나 처리량은 바뀌지 않습니다([스트리밍이 개선하지 않는 것](../05_serving_containers.md#스트리밍이-개선하지-않는-것)).

!!! warning "절단은 에러 없이 200 응답으로 옵니다"
    잘렸는지 확인하려면 응답의 `finish_reason`을 봐야 합니다(`length`면 절단). 코스별 `gen_max_tokens` 전체 표와 확인 코드는 [max_tokens 절단과 finish_reason](../05_serving_containers.md#max_tokens-절단과-finish_reason)에 있습니다.
    배포, 평가, 에이전트 셀이 모두 같은 `gen_max_tokens=512`를 쓰도록 맞춰져 있으니, 한 곳만 256으로 되돌리지 마세요. 그 셀에서만 답이 잘려 원인 파악이 어려워집니다.

---

## 이어서 볼 문서

- [00 전체 지도](../00_overview.md#5개-독립-코스와-공통-레이어): 5개 코스 비교와 공통 `common/` 레이어
- [02 합성 데이터](../02_synthetic_data.md): grounded 합성과 held-out 규율
- [03 파인튜닝](../03_finetuning.md): LoRA/QLoRA, Gemma 관용구, 머지와 re-export
- [04 SageMaker AI 추론](../04_sagemaker_inference.md): endpoint 3층 구조와 호출 스키마
- [05 서빙 컨테이너](../05_serving_containers.md): 엔진 선택, OOM, 절단 실측 함정
- [E2E 실행 가이드](../RUN_E2E.md#단계별-실행과-데이터-핸드오프): 단계별 실행 순서, 비용 안내, 완료 기준

!!! danger "비용과 cleanup"
    학습 Job은 실행 시간만큼 과금되고 **endpoint는 호출하지 않아도 삭제할 때까지 시간당 계속 과금**됩니다. 코스를 마쳤으면 `99_cleanup`을 반드시 실행해 endpoint, endpoint-config, model을 모두 지우세요.
    합성 데이터 생성과 `04_evaluate`의 LLM-judge, `05_agentic_strands`는 Bedrock 호출로 별도 과금됩니다(상주 리소스는 없습니다). `06`으로 AgentCore를 배포했다면 `bash agentcore/cleanup_agent.sh --aws`도 필요합니다.
