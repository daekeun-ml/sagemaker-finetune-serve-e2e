# 추출 → JSON 코스: 텍스트에서 스키마에 맞는 JSON 뽑기

!!! info "Scope"
    자연어 텍스트에서 **스키마가 정해진 JSON**을 뽑아내야 하는 개발자를 위한
    코스입니다(`tracks/01_extraction_to_json`, 이 프로젝트의 플래그십).

    - **선행 조건**: [시작하기](../getting_started.md)의 설치와 AWS 자격증명과 Amazon SageMaker AI 실행 role
      (`00_setup`이 확인). SageMaker AI가 처음이면 [SageMaker AI 기초](../01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: task 정의, 시드 데이터셋, 성공 기준, 노트북 구성, 코스별 설정값
    - **여기서 다루지 않는 것**: 학습 방식은 [파인튜닝](../03_finetuning.md),
      배포와 서빙은 [SageMaker AI 추론](../04_sagemaker_inference.md), 완주 절차는
      [실행 runbook](../RUN_E2E.md)
    - **다른 코스**: 이미지 입력(영수증 등)은 [멀티모달 추출](multimodal.md),
      라벨 하나만 고르는 문제는 [분류](classification.md)

이 코스와 관련된 repository 파일입니다(디렉터리 이름 `tracks/`와 `TRACKS`, `track_data.py` 같은 식별자는 역사적 이유로 그대로 둡니다):

- `tracks/01_extraction_to_json/track_data.py`: 시드 로드, `{input, output}` 어댑터, `SYSTEM_PROMPT`
- `tracks/01_extraction_to_json/scripts/train.py`, `train_grpo.py`: SFT / GRPO 학습(로컬 dry-run ↔ SageMaker AI 겸용)
- `tracks/01_extraction_to_json/*.ipynb`: 이 코스의 노트북 10개
- `common/config.py`: `TRACKS['extraction']` 레지스트리 항목(시드 데이터셋, `max_seq_length=2048`)
- `common/eval_utils.py`: `eval_extraction()`(valid_json_rate, name_accuracy, arg_f1)
- `tracks/01_extraction_to_json/_build_notebooks.py`: 이 코스의 `TrackSpec`(endpoint prefix, serving과 생성 길이, GRPO reward 종류)

---

## 이 코스가 푸는 문제

!!! abstract "쉽게 말하면"
    "사용자가 한 말"과 "쓸 수 있는 함수 목록(JSON 스키마)"을 함께 주면, 모델이 **어느 함수를 어떤 인자로 부를지**를 JSON 한 덩어리로만 답하게 만듭니다.
    설명 문장은 쓰지 않습니다. 코드가 그대로 `json.loads()` 할 수 있는 출력이 목표입니다.

시드 원본 row는 멀티턴 대화 텍스트지만, `track_data._parse_glaive_row()`가 이를 학습용 한 쌍으로 접습니다. 아래는 실제 학습 JSONL(`data/train.jsonl`)에 들어가는 형태입니다.

**입력**(`SYSTEM_PROMPT` + 사용자 발화 + `[Available tools]` 스키마)

```text
You are a precise information-extraction engine. Read the user text and the available
tool schema, then output ONLY a valid JSON object of the form
{"name": "<function>", "arguments": {<args>}}. No prose, JSON only.

Hi, I would like to calculate my BMI. I weigh 70 kilograms and my height is 1.75 meters.

[Available tools]
You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "calculate_bmi",
    "description": "Calculate the Body Mass Index (BMI) of a person",
    "parameters": {"type": "object", "properties": {
        "weight": {"type": "number", "description": "The weight of the person in kilograms"},
        "height": {"type": "number", "description": "The height of the person in meters"}},
        "required": ["weight", "height"]}
}
```

**출력**(정답: 이것만 생성하도록 학습합니다)

```json
{"name": "calculate_bmi", "arguments": {"weight": 70, "height": 1.75}}
```

`to_messages()`가 이 쌍을 `[{"role":"user"}, {"role":"assistant"}]`로 만들 때 `SYSTEM_PROMPT`를 **첫 user 턴에 합칩니다**(fold). Gemma instruct chat template이 system role을 거부하기 때문입니다. 근거는 [chat template과 system fold](../03_finetuning.md#chat-template과-system-fold)에 있습니다.

!!! warning "학습과 같은 형태로 물어야 학습 효과가 보입니다"
    같은 파인튜닝 모델에 같은 질문("What's the weather in Busan tomorrow?")을 세 가지로 물은 실측 결과입니다(`02b_local_serve.ipynb`의 3-D 셀).

    | 프롬프트 구성 | 응답 |
    |---|---|
    | system 없음, 스키마 없음 | `"I do not have real-time access to weather..."`: 일반 챗봇처럼 답합니다 |
    | system 있음, 스키마 없음 | `{"name": "get_current_weather", ...}`: 함수명을 추측합니다 |
    | system + 스키마 | `{"name": "get_weather", "arguments": {...}}` ✅ 정확 |

    배포 후 호출부(`04_evaluate`, `05_agentic_strands`)는 모두 세 번째 형태로 보냅니다. 스키마를 빼고 테스트한 뒤 "학습이 안 됐다"고 판단하는 것이 이 코스에서 가장 흔한 오진입니다.

---

## 시드 데이터셋

[`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)(apache-2.0, ungated)를 씁니다. 토큰 없이 로드되고, 라이선스 전파 부담이 없어 파생 모델 배포가 자유롭습니다(`04_domain_qa`의 dolly는 cc-by-sa-3.0으로 share-alike 의무가 있습니다. [라이선스 요약](../00_overview.md#라이선스-요약)).

원본 스키마는 `system`(사용 가능한 함수 JSON 스키마) + `chat`(멀티턴 텍스트, assistant가 `<functioncall> {json}` 방출) 두 컬럼입니다. `_parse_glaive_row()`는 정규식으로 **첫 USER 발화**와 **첫 `<functioncall>` JSON**만 뽑고, `system`의 스키마 힌트를 `[Available tools]` 헤더와 함께 입력에 붙입니다. 파싱이나 `json.loads`가 실패한 row는 `None`으로 버려지므로 `load_seed_examples(n)`은 **성공한 것만 n개** 채워 돌려줍니다.

!!! warning "인자 없는 함수가 대다수입니다"
    이 시드는 `{"arguments": {}}`처럼 **인자가 비어 있는 호출**이 많습니다. `common/grpo_data.py`가 쓰는 glaive 뒷부분 구간에서 94%입니다. SFT에는 문제가 없지만, GRPO에서는 채점이 사실상 "함수명 맞았나"로 축소되어 rollout이 전부 만점이 되고 학습 신호가 사라집니다.
    그래서 `02a`의 기본 prompt 소스(`synth`)는 생성 프롬프트에 난이도 제약을 걸어 **인자 2개 이상**을 강제합니다(제약 적용 후 실측: 인자 없음 0건 / 평균 인자 2.1개). 자세한 근거는 [RL prompt 소스 3가지](../03_finetuning.md#rl-prompt-소스-3가지)에 있습니다.

시드는 앞의 300건(`config.NUM_SEED_SAMPLES`)만 쓰고, 나머지는 Bedrock grounded 합성으로 늘립니다(`config.NUM_SYNTHETIC` 기본 200, repository에 포함된 `.env`는 100). 시드 1건의 길이는 요약 코스(중앙 1,651자)보다 짧은 약 475자라 합성 호출 지연이 작습니다([생성 건수 결정](../02_synthetic_data.md#생성-건수-결정-num_synthetic-기본값) 참고).

---

## 성공 기준

채점은 `common/eval_utils.eval_extraction()` 한 함수로 끝납니다. 모델 호출도 LLM-judge도 필요 없는 **순수 파이썬 채점**이라 재현성이 높고 비용이 0입니다.

| 지표 | 정의 | 역할 |
|---|---|---|
| `valid_json_rate` | 출력이 dict이고 `name` 키가 있고 `arguments`가 dict인 비율 | 형식 게이트 |
| `name_accuracy` | 함수명 정확 일치 비율 | 함수 선택 능력 |
| **`arg_f1`** | `(키, 정규화된 값)` 쌍 집합의 micro precision/recall F1 | **주 지표**(단 아래 경고 참고) |
| `exact_match` | 함수명 + 인자 집합이 완전히 일치 | 가장 엄격한 참고치 |

**왜 `arg_f1`이 이 태스크에 맞나.** 인자가 3개인 호출에서 2개를 맞혔다면 그것은 완전 실패가 아닙니다. `exact_match`만 보면 이 차이가 지워지고, `valid_json_rate`만 보면 "JSON 형식은 맞지만 값이 다 틀린" 모델이 만점을 받습니다. `arg_f1`은 인자 단위 부분 점수를 주면서, 없는 인자를 만들어내면 precision으로, 빠뜨리면 recall로 벌점을 줍니다. 값 비교 전에 `_norm_val()`이 dict/list를 정렬 직렬화하고 `1.0`을 `1`로 맞추므로 표현 차이 때문에 틀리지 않습니다.

파싱이 실패하면 `eval_extraction()`은 예측 인자 집합을 공집합으로 두므로, **정답에 인자가 있는 건에서는** 그 인자 전부가 false negative가 되어 `arg_f1`이 떨어집니다.

!!! warning "이 시드의 held-out에서는 arg_f1 단독으로 판단하면 안 됩니다"
    위 문장이 성립하려면 정답에 인자가 있어야 합니다. 그런데 held-out 구간(시드 인덱스 300~349, `N_EVAL=50`)의 정답은 **50건 중 46건이 `{"arguments": {}}`**입니다(위 [시드 데이터셋](#시드-데이터셋) 절의 "인자 없음 94%"와 같은 원인). `gold_args`가 공집합이면 `fn += len(gold_args - pred_args)`가 0을 더하므로, 그 46건은 파싱이 완전히 깨져도 `arg_f1`에 **아무 벌점도 남기지 않습니다.**

    실측으로 확인한 최악의 경우입니다. 인자가 빈 46건에는 산문(`"Sure! I can help with that."`)을, 인자가 있는 4건에는 완벽한 JSON을 내는 가짜 모델을 채점하면:

    ```text
    {'n': 50, 'valid_json_rate': 0.08, 'name_accuracy': 0.08, 'arg_f1': 1.0, 'exact_match': 0.08}
    ```

    92%가 망가진 모델이 `arg_f1` 만점을 받습니다. 즉 이 held-out에서 `arg_f1`은 실질적으로 **50건 중 4건으로만 계산**됩니다.

    그래서 판단 순서는 **`valid_json_rate` → `name_accuracy` → `arg_f1`**입니다. 앞의 두 지표가 형식과 함수 선택의 게이트 역할을 하고, `arg_f1`은 그 게이트를 통과한 뒤 인자 채우기 품질을 보는 값으로 읽으세요. 인자 단위 성능을 제대로 재려면 `02a`의 `synth` prompt처럼 **인자 2개 이상**을 강제한 held-out을 따로 만들어야 합니다.

평가는 `04_evaluate.ipynb`가 endpoint를 `temperature=0.0`으로 호출해 수행하고, held-out은 학습에 쓴 앞 300건을 **명시적으로 건너뛴 뒤** 그 다음 50건(`N_EVAL`, dry-run은 20)을 씁니다. 합성 데이터로 평가하면 teacher 모방도를 재는 데 그칩니다. [held-out 규율](../02_synthetic_data.md#held-out-규율-합성으로-평가-금지)을 참고하세요.

---

## 노트북 순서

이 코스의 노트북은 **10개**입니다. `02a`(GRPO)와 `02b`(로컬 서빙)를 모두 갖는 두 코스 중 하나입니다(다른 하나는 [분류 코스](classification.md), 요약과 도메인 QA는 `02a`가 없어 9개입니다).

| 노트북 | 결과 |
|---|---|
| `00_setup` | 자격증명, 리전, role 확인, `role`, `bucket` `%store` 저장 |
| `01_data_and_synthetic` | 시드 300건 파싱 + grounded 합성 → `data/train.jsonl`, 토큰 길이, 중복 EDA(JSON 파싱률 점검 포함) |
| `02_train_sft_sagemaker` | SageMaker AI 학습 Job(TRL SFT + QLoRA) → 머지된 모델 artifact, `%store md_extraction` |
| `02a_train_grpo_sagemaker` | **(선택)** SFT artifact를 base로 GRPO 정련 → 새 artifact |
| `02b_local_serve` | **(선택)** 로컬 GPU vLLM preflight: 배포 전 30초 안에 같은 오류를 재현 |
| `03_deploy_endpoint` | `gemma-extraction-vllm-<timestamp>` real-time endpoint + invoke 스모크 |
| `04_evaluate` | held-out `valid_json_rate`, `name_accuracy`, `arg_f1` |
| `05_agentic_strands` | `extract_structured_json` tool을 가진 Strands 에이전트(reasoning은 Bedrock Claude) |
| `06_agentcore_deploy` | AgentCore Runtime 배포(로컬 dev → 클라우드) |
| `99_cleanup` | endpoint → endpoint-config → model 삭제, 로컬 모델 정리 |

`02a`가 있는 이유는 **reward를 프로그램으로 채점할 수 있기 때문**입니다. `scripts/train_grpo.py`의 `reward_extraction`은 유효 JSON에 0.3, 함수명 일치에 0.3, 인자 F1에 0.4를 배분합니다. 요약과 도메인 QA에는 이런 채점 함수가 없어 GRPO 노트북이 아예 생성되지 않습니다. 판단 근거는 [왜 추출과 분류 코스에만 GRPO가 있나](../03_finetuning.md#왜-추출과-분류-코스에만-grpo가-있나).

`02b`가 있는 이유는 배포 실패 원인이 대개 모델 파일 문제인데 endpoint는 한 번 띄우는 데 5~15분이 걸리기 때문입니다. 클라우드와 같은 엔진(vLLM)으로 로컬에서 먼저 확인하면 이 왕복이 사라집니다. 로컬 GPU가 없으면 건너뛰어도 됩니다(`has_local_serve=True`인 텍스트 코스 공통, 멀티모달 05에는 없습니다). 최소 경로는 `00 → 01 → 02 → 03 → 04 → 99`입니다.

!!! tip "먼저 DRY_RUN=1로 한 바퀴 도세요"
    `DRY_RUN=1`이면 시드 8건, 합성 6건, held-out 20건으로 줄어들어 파이프라인 형태만 빠르게 검증합니다. 단계별 핸드오프와 비용 가드는 [실행 runbook](../RUN_E2E.md)에 정리돼 있습니다.

---

## 코스별 설정값

다른 코스와 다른 값만 모았습니다. 출처는 `common/config.py`의 `TRACKS["extraction"]`와 `_build_notebooks.py`의 `TrackSpec`입니다.

| 설정 | 값 | 근거 |
|---|---|---|
| `seed_dataset` | `glaiveai/glaive-function-calling-v2` | apache-2.0, ungated |
| `max_seq_length` | **2048** | 입력에 툴 스키마 JSON이 통째로 들어가 길어집니다(분류 코스는 512) |
| `serve_max_model_len` | **미지정 → 4096** | spec에 값이 없어 `max_seq_length × 2`가 됩니다([학습 길이와 서빙 길이](../00_overview.md#학습-길이와-서빙-길이는-다른-값입니다)) |
| `gen_max_tokens` | **256** | 정답 JSON은 짧습니다. 길이가 문제되는 코스와의 대조는 [max_tokens 절단과 finish_reason](../05_serving_containers.md#max_tokens-절단과-finish_reason)에 있습니다 |
| `grpo_reward_kind` | `extraction` | 유효 JSON 0.3 + 함수명 일치 0.3 + 인자 F1 0.4(`train_grpo.py`의 `reward_extraction`) |
| `eval_kind` | `extraction` | `eval_extraction()` 경로 선택 |
| `endpoint_prefix` | `gemma-extraction` | `%store` 오염을 피하는 코스 전용 키(`ep_extraction`)와 cleanup 필터에 함께 쓰입니다 |
| 스트리밍 기본값 | `False` | JSON은 완성돼야 파싱이 되고 애초에 짧아 토큰을 흘려도 체감 이득이 없습니다 |

`max_seq_length`는 요약 코스(2048)와 값이 같지만 이유가 다릅니다. 요약은 **입력 문서**가 길고, 이 코스는 **스키마**가 깁니다. 그래서 요약 코스는 `serve_max_model_len`, `gen_max_tokens`를 명시적으로 올렸고(4096/512), 이 코스는 기본값으로 충분합니다.

??? question "오해: “function calling 데이터로 학습하니 이 모델은 툴을 실행하는 거죠?”"
    아닙니다. 이 코스가 학습하는 것은 **함수 호출을 표현하는 JSON을 생성하는 능력**이고, 그 JSON을 실제로 실행하는 주체는 호출자 쪽 코드입니다.
    `05_agentic_strands`에서 orchestration은 Bedrock Claude가 맡고, SLM endpoint는 `extract_structured_json` tool로 구조화 JSON만 반환합니다. SLM은 빠른 구조화 추출기, Claude는 reasoning model로 역할이 명확히 나뉘어 있어 flagship 코스로 삼았습니다([Agentic loop](../06_agentic.md)).

??? question "오해: “툴 호출이 목적이 아니면 이 코스는 안 맞나요?”"
    출력 스키마가 정해진 추출 문제 전반에 그대로 옮겨집니다. 규약서에서 당사자, 금액, 기간을 뽑거나, 문의 메일에서 주문번호와 요청유형을 뽑는 작업은 형태가 같습니다. `{"name": ..., "arguments": {...}}` 대신 원하는 스키마를 `SYSTEM_PROMPT`에 넣고, `track_data.py`의 파서를 자기 데이터에 맞게 바꾸면 됩니다(다른 코스들도 이 파일만 교체해 만들었습니다).
    다만 `eval_extraction()`은 `name`/`arguments` 구조를 가정하므로, 스키마를 바꾸면 채점 함수도 같이 손봐야 합니다.

---

## 이어서 볼 문서

- [00 전체 지도](../00_overview.md#5개-독립-코스와-공통-레이어): 5개 코스 비교와 공통 `common/` 레이어
- [02 합성 데이터](../02_synthetic_data.md): grounded 합성과 held-out 규율
- [03 파인튜닝](../03_finetuning.md): LoRA/QLoRA, Gemma 관용구, 머지와 re-export
- [04 SageMaker AI 추론](../04_sagemaker_inference.md): endpoint 3층 구조와 호출 스키마
- [05 서빙 컨테이너](../05_serving_containers.md): 엔진 선택, OOM, 절단 실측 함정
- [실행 runbook](../RUN_E2E.md#단계별-실행과-데이터-핸드오프): 단계별 실행 순서, 비용 가드, 완료 기준

!!! danger "비용과 cleanup"
    학습 Job은 실행 시간만큼 과금되고 **endpoint는 삭제할 때까지 시간당 계속 과금**됩니다. 코스를 마쳤으면 `99_cleanup`을 반드시 실행해 endpoint, endpoint-config, model을 모두 지우세요.
    `02a`의 GRPO는 prompt당 rollout을 여러 개 생성하므로 SFT보다 오래 걸립니다(노트북 기본 `MAX_RUNTIME_HOURS=6`). 합성 데이터 생성과 `05_agentic_strands`는 Bedrock 호출로 별도 과금되고, `06`으로 AgentCore를 배포했다면 `bash agentcore/cleanup_agent.sh --aws`도 필요합니다.
