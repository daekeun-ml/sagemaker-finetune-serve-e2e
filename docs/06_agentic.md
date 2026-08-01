# 06 · Agentic Loop — 파인튜닝 SLM(tool) + Bedrock Claude(reasoning)

!!! info "읽는 사람과 범위"
    이 킷으로 Gemma SLM을 학습(`02_train_sft_sagemaker`)·배포(`03_deploy_endpoint`)해 **real-time endpoint를 이미 가진 분**이 대상입니다. Strands/AgentCore는 처음이어도 괜찮습니다.
    다루는 것: SLM endpoint를 tool로 노출하는 방법, Bedrock Claude를 reasoning으로 붙이는 방법, AgentCore Runtime 배포 계약.
    다루지 않는 것: 학습·데이터 합성([파인튜닝](03_finetuning.md)·[합성 데이터](02_synthetic_data.md)), 서빙 컨테이너 선택([서빙 컨테이너 선택](05_serving_containers.md)).

이 문서는 노트북 `05_agentic_strands`와 `06_agentcore_deploy`, 그리고 `agentcore/` 디렉터리의 스크립트를 설명합니다.

이 문서와 관련된 킷 파일:

- `agentcore/app.py` — AgentCore Runtime 엔트리포인트 스캐폴드(`BedrockAgentCoreApp` + `@app.entrypoint`), 정보추출 트랙 tool 포함
- `agentcore/templates/main.py` — CLI 스캐폴딩의 데모 tool을 대체해 이식되는 `extract_structured_json` tool
- `agentcore/templates/load.py` — reasoning 모델 로더, `BEDROCK_CLAUDE_MODEL_ID` env를 읽고 없으면 킷 기본값으로 폴백
- `agentcore/Dockerfile` — ARM64 런타임 이미지(`/invocations`+`/ping`:8080 계약은 SDK가 제공)
- `agentcore/setup_agentcore_cli.sh` — nvm으로 Node 20 + `@aws/agentcore` CLI 설치(sudo 불필요)
- `agentcore/create_agent.sh` — 프로젝트를 non-interactive로 생성하고 SLM tool을 이식
- `agentcore/verify_local.sh` — 배포 전 로컬 dev 서버로 실제 추론 검증
- `agentcore/cleanup_agent.sh` — dev 프로세스·프로젝트 폴더 정리, `--aws`면 Runtime/ECR까지 삭제
- `common/aws_utils.py` — 서비스 경계의 실체(`invoke_sagemaker_chat`·`stream_sagemaker_chat`·`bedrock_converse`)
- `common/llm_gateway.py` — LiteLLM 경유로 Bedrock과 endpoint를 단일 인터페이스로 호출(대안 경로)
- `common/config.py` — `BEDROCK_CLAUDE_MODEL_ID`·`BEDROCK_REGION` 등 env 기반 설정

노트북 순서: `05_agentic_strands` → `06_agentcore_deploy`

!!! warning "빠르게 바뀌는 값"
    **모델 ID·DLC 이미지 태그·SDK 버전·리전·GA 상태**는 이 문서에서 가장 빨리 낡는 부분입니다. Bedrock 모델 로스터, AgentCore GA·지원 리전, `bedrock-agentcore` import 경로, `@aws/agentcore` CLI 버전은 **실행 직전에 다시 확인**하세요.
    코드에 `# TODO verify` 주석이 붙은 지점이 그 목록이고, 최종 확인처는 각 절에 인라인으로 달아 둔 공식 문서 링크입니다. 시크릿·계정 ID·endpoint 이름은 하드코딩하지 말고 env/param으로 주입하세요.

---

## TL;DR

**파인튜닝 Gemma SLM은 빠른 "전문가 도구(tool)", Bedrock Claude는 범용 "추론·오케스트레이션(reasoning)" 역할을 맡습니다. agent framework(Strands 우선, LangGraph 옵션)가 tool-use로 이 둘을 묶습니다. endpoint는 `sagemaker-runtime`, Bedrock은 `bedrock-runtime`으로 서로 다른 서비스이므로 절대 섞어서 부르면 안 됩니다.**

세부 결론은 다음과 같습니다.

1. SLM endpoint는 `@tool`로 감싼 `extract_structured_json`(내부는 `sagemaker-runtime invoke_endpoint`)으로 노출합니다. Claude가 "언제 이 도구를 부를지"를 판단합니다 — [Strands로 묶는 agentic loop](#strands로-묶는-agentic-loop).
2. Strands는 `BedrockModel`을 기본 provider로 사용합니다(reasoning=Claude). 멀티 프로바이더가 필요하면 `LiteLLMModel` 옵션을 쓰세요 — [provider 선택](#provider-선택--bedrockmodel과-litellmmodel).
3. Bedrock Claude 모델 ID는 **inference-profile prefix**(`us.`/`eu.`/`apac.`/`global.`) 형식이며, env/param으로 주입해야 합니다. 하드코딩은 금지입니다 — [Bedrock Claude 모델 ID 규칙](#bedrock-claude-모델-id-규칙).
4. 프로덕션 배포는 **AgentCore Runtime**(ARM64 컨테이너, `/invocations`+`/ping`:8080)으로 진행합니다 — [프로덕션 배포](#프로덕션-배포--agentcore-runtime).
5. 과금이 **두 군데 이상**(endpoint 시간당 + Bedrock 토큰당 + AgentCore Runtime)에서 발생하므로 cleanup이 필수입니다 — [비용과 cleanup](#비용과-cleanup).

---

## 기존 Pain Point

`03_deploy_endpoint`까지 진행했다면 대개 이런 상태일 것입니다.

- SLM endpoint는 있지만 **"그래서 이걸 어떻게 앱으로 쓰지?"** 하는 고민이 생깁니다. endpoint는 프롬프트를 주면 텍스트를 뱉는 함수일 뿐, 스스로 "언제 나를 부를지"를 판단하지 못하기 때문입니다.
- 반대로 Bedrock Claude는 범용 추론은 잘하지만, **우리 도메인 전용 구조화 추출/분류는 파인튜닝한 SLM이 더 싸고 빠르고 정확**합니다.
- 흔한 오해로 "그럼 Claude한테 endpoint를 Bedrock API로 부르라고 하면 되지 않나?"라고 생각하기 쉽지만, 이는 **틀린 접근**입니다. endpoint와 Bedrock은 별개 서비스입니다([서비스 경계](#서비스-경계--endpoint--bedrock)).
- 또 다른 함정은, 로컬에서 `python app.py`로 잘 돌던 에이전트를 그대로 프로덕션에 올리려다 **AgentCore Runtime의 HTTP 계약(ARM64·`/invocations`·`/ping`·8080)** 을 몰라서 막히는 경우입니다.

이 문서는 "endpoint + Claude"를 하나의 **agentic loop**로 묶고, 로컬에서 AgentCore Runtime까지 올리는 최소 경로를 정리합니다.

---

## 왜 SLM은 도구, Claude는 두뇌인가

!!! abstract "쉽게 말하면"
    - **Claude(reasoning LLM)** 는 전체 문제를 이해하고 "지금 무슨 도구가 필요한가"를 결정하는 **팀장**에 해당합니다.
    - **파인튜닝 Gemma SLM(tool)** 은 한 가지 일(예: 텍스트→JSON 추출)을 아주 빠르고 정확하게 처리하는 **전문가 사원**에 해당합니다.
    - **agent framework(Strands)** 는 팀장이 사원에게 일을 시키는 **결재 라인(tool-use 프로토콜)** 역할을 합니다.

역할을 나누는 근거는 과금 모델과 커스터마이즈 수단이 서로 다르다는 데 있습니다.

### 대조표 — reasoning LLM vs specialist SLM

| 축 | Bedrock Claude (reasoning) | 파인튜닝 Gemma SLM (tool) |
|---|---|---|
| 역할 | 계획·판단·도구 선택·결과 검증 | 도메인 특화 단일 작업 실행 |
| 호출 서비스 | `bedrock-runtime` → `converse()` | `sagemaker-runtime` → `invoke_endpoint()` |
| 과금 모델 | 토큰당(사용량) | endpoint 시간당(상시 인스턴스) |
| 강점 | 범용 추론·멀티스텝·자연어 오케스트레이션 | 전용 태스크의 속도·비용·포맷 안정성 |
| 모델 관리 | AWS 관리형(모델 ID만 지정) | 우리가 학습·배포·운영 |
| 커스터마이즈 | 프롬프트/few-shot 위주 | 가중치 자체를 파인튜닝(LoRA) |

??? question "오개념 — “Claude 하나로 다 하면 안 되나요?”"
    됩니다. 다만 도메인 전용 추출/분류에서는 (1) 작은 파인튜닝 SLM이 **더 저렴하고 지연도 낮으며** (2) 출력 포맷이 **더 안정적**입니다.
    반대로 SLM 하나로 다중 스텝 추론을 시키기는 어렵습니다. 그래서 **역할 분담**이 핵심입니다.

### 기술적 차이 3가지

1. **서비스 경계**: endpoint(`sagemaker-runtime`)와 Bedrock(`bedrock-runtime`)은 클라이언트도 API도 다릅니다([서비스 경계](#서비스-경계--endpoint--bedrock)). 프레임워크는 이 둘을 각각 다른 통합으로 호출합니다.
2. **제어 주체**: Claude가 "도구 호출 여부와 인자"를 결정하면, 프레임워크가 실제 tool 함수를 실행한 뒤 결과를 다시 Claude에게 돌려줍니다(tool-use round-trip).
3. **배포 단위**: SLM은 SageMaker endpoint(개별 리소스)로, 에이전트 전체는 AgentCore Runtime(컨테이너)으로 각각 별도 배포됩니다.

---

## 서비스 경계 — endpoint ≠ Bedrock

이 킷에서 가장 흔하게 사고가 나는 지점이므로 **절대 섞지 마세요**. `common/aws_utils.py` 상단에도 못 박아 둔 규칙은 다음과 같습니다.

```
SageMaker endpoint 호출  = boto3 "sagemaker-runtime" 클라이언트, invoke_endpoint()
Bedrock Claude 호출       = boto3 "bedrock-runtime" 클라이언트, converse()
→ 별개 서비스 · 별개 클라이언트. "endpoint를 Bedrock API로 호출"은 잘못.
```

### tool-use 왕복 흐름

```
사용자 입력
   │
   ▼
┌──────────────────────────────────┐
│  Bedrock Claude (reasoning)      │  bedrock-runtime.converse()
│  "이건 추출 작업 → 도구 호출"    │
└────────────────┬─────────────────┘
                 │ tool-use 요청 (name=extract_structured_json, args=...)
                 ▼
┌──────────────────────────────────┐
│  @tool extract_structured_json   │  sagemaker-runtime.invoke_endpoint()
│  파인튜닝 Gemma SLM endpoint     │
└────────────────┬─────────────────┘
                 │ 결과(JSON 등)
                 ▼
   Claude가 결과 검증·설명 → 최종 응답
```

- 배포된 모델을 코드에서 호출하는 경로는 [SageMaker 모델 배포 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)에 정리되어 있습니다. 이 킷의 tool은 그중 boto3 `sagemaker-runtime` 직접 호출을 씁니다.
- endpoint 호출 스키마는 서빙 컨테이너를 따릅니다. 이 킷의 기본 엔진은 vLLM(대안 SGLang·DJL LMI)이고 셋 다 **OpenAI 호환**이므로, tool은 `{"messages": [...]}` 스키마를 씁니다 — `common/aws_utils.py`의 `invoke_sagemaker_chat()`입니다. `{"inputs", "parameters"}` generation 스키마(LMI rolling-batch·HF TGI 관용)가 필요하면 같은 파일의 `invoke_sagemaker_endpoint()`를 쓰세요.
- 스트리밍이 필요하면 `invoke_endpoint_with_response_stream`을 감싼 `stream_sagemaker_chat()`을 사용하세요. **실측(요약 트랙, vLLM 0.26.0): 첫 청크 0.51s / 전체 9.24s** 로 대기 체감이 약 18배 줄었습니다. 다만 전체 생성 시간과 동시 처리량은 그대로입니다 — 자세한 실측은 [응답 스트리밍](04_sagemaker_inference.md#응답-스트리밍--invoke_endpoint_with_response_stream)에 있습니다.
- Bedrock은 `converse()`(또는 스트리밍 `converse_stream()`)를 쓰며, [Converse API 문서](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)가 규정한 메시지·`inferenceConfig` 스키마를 따릅니다. 구현은 `common/aws_utils.py`의 `bedrock_converse()`이고, 호출 예시는 [amazon-bedrock-samples](https://github.com/aws-samples/amazon-bedrock-samples)에도 있습니다.

??? question "오개념 — “LiteLLM 쓰면 둘이 같은 거 아닌가요?”"
    그렇지 않습니다. `common/llm_gateway.py`의 [LiteLLM](https://github.com/BerriAI/litellm)은 **호출 인터페이스만** OpenAI 호환 `completion()`으로 통일해 줄 뿐, 내부적으로는 여전히 `bedrock/converse/<model>`와 `sagemaker_chat/<endpoint>`로 **다른 백엔드에 라우팅**합니다.
    "같은 서비스"가 된 것이 아니라 "같은 함수 시그니처로 부를 수 있게" 감싼 것일 뿐입니다.

경계를 지켰더라도, endpoint에 보내는 **페이로드 형식**에서 한 번 더 걸리는 지점이 있습니다.

??? question "오개념 — “프롬프트를 렌더해서 inputs로 직송하면 되지 않나?”"
    안 됩니다. vLLM/SGLang/LMI는 OpenAI 호환 서버라 **chat template을 서버가 적용**합니다. 로컬 토크나이저로 렌더한 raw 문자열을 `{"inputs": ...}`로 보내면 실측으로 다음 에러가 납니다.
    `Could not find a handler for the request. Expected one of: ['ChatCompletionRequest', 'CompletionRequest']`
    그래서 tool은 `messages`를 그대로 보냅니다 — 클라이언트에 tokenizer/transformers 의존이 필요 없습니다.

### SageMaker 추론 4옵션

SLM을 어디에 배포할지에도 선택지가 있습니다(`03_deploy_endpoint`에서는 real-time을 선택했습니다). 네 옵션의 정의는 [SageMaker 배포 옵션 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-options.html)를 따르며, 축별 상세 비교는 [추론 4옵션 대조표](04_sagemaker_inference.md#추론-4옵션-대조표)에 있습니다.

| 옵션 | 특징 | LLM/SLM 적합성 |
|---|---|---|
| Real-time | 상시 endpoint, 저지연 | ✅ 적합 — 이 킷 기본. 시간당 과금 |
| Serverless | 온디맨드, 스케일-투-제로 | ❌ 부적합 — GPU 없음 |
| Asynchronous | 큐 기반, 대용량/긴 추론 | 조건부 — 긴 생성·오프라인 배치 |
| Batch Transform | 잡 단위 대량 오프라인 | ❌ 부적합 — 실시간 agentic엔 불가 |

??? question "오개념 — “비용 아끼려고 Serverless에 SLM 올리면 어떨까요?”"
    부적합합니다. 현재 SageMaker Serverless Inference는 **GPU를 제공하지 않아** Gemma 같은 SLM 서빙에 쓸 수 없습니다. agentic loop의 tool은 real-time endpoint를 전제로 합니다.
    Serverless의 GPU 지원 여부는 이 문서에서 가장 빨리 낡을 수 있는 항목이므로 **실행 직전에 재확인**하세요.

??? info "더 읽을 거리"
    - [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) — `Predictor`·`invoke_endpoint` 래퍼의 실제 구현.
    - [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples) — 배포 옵션별 노트북 예제.

---

## Strands로 묶는 agentic loop

!!! abstract "쉽게 말하면"
    - 파이썬 함수에 `@tool` 데코레이터만 붙이면 그것이 "도구"가 됩니다.
    - `Agent(model=..., tools=[...])`에 넘기면 Claude가 필요할 때 알아서 호출합니다.
    - tool의 docstring이 "언제 이 도구를 쓸지"를 Claude에게 알려 주는 유일한 근거입니다.

`@tool` 데코레이터와 `Agent` 클래스의 계약은 [Strands Agents 문서](https://strandsagents.com/)에, 구현은 [strands-agents/sdk-python](https://github.com/strands-agents/sdk-python)에 있습니다. `agentcore/app.py`와 노트북 `05_agentic_strands`의 실제 패턴은 다음과 같습니다.

```python
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def extract_structured_json(text: str) -> str:
    """Extract structured JSON from text using the fine-tuned Gemma SLM (SageMaker endpoint)."""
    # endpoint 호출은 sagemaker-runtime — Bedrock 아님.
    # messages 그대로 전송 → 서버(vLLM/SGLang/LMI)가 chat template을 적용.
    rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = {"messages": [{"role": "system", "content": SLM_SYSTEM_PROMPT},
                            {"role": "user", "content": text}],
               "max_new_tokens": 256, "temperature": 0.1}
    resp = rt.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                              ContentType="application/json", Body=json.dumps(payload))
    ...

model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)  # reasoning=Claude
agent = Agent(model=model, tools=[extract_structured_json], system_prompt="You orchestrate...")
```

노트북에서는 같은 tool을 `common/aws_utils.invoke_sagemaker_chat()`으로 구현해, 응답 파싱까지 공통 코드에 맡깁니다(`max_tokens=256`, `temperature=0.1`). SLM에 넣는 system 프롬프트는 **학습 때 쓴 것과 동일**해야 합니다(트랙별 `track_data.SYSTEM_PROMPT`).

### provider 선택 — BedrockModel과 LiteLLMModel

| 상황 | provider |
|---|---|
| Bedrock Claude만 쓴다 (이 킷 기본) | `strands.models.BedrockModel` — 기본 provider |
| 여러 프로바이더(OpenAI/Anthropic API/기타)를 갈아끼워야 한다 | [`LiteLLMModel`](https://github.com/BerriAI/litellm) 옵션 (`strands-agents[litellm]`) |

??? question "오개념 — “reasoning 모델을 LiteLLM으로 바꾸면 tool 호출 방식도 바뀌나요?”"
    아닙니다. `@tool extract_structured_json` 내부의 `sagemaker-runtime invoke_endpoint`는 그대로 유지됩니다.
    provider 교체는 **reasoning LLM 백엔드**만 바꿀 뿐, tool은 여전히 SageMaker endpoint를 그대로 호출합니다.

### LangGraph 옵션

Strands를 우선(권장하며 이 킷의 기본)으로 하되, 이미 LangGraph 그래프 오케스트레이션에 익숙하거나 복잡한 상태 머신·분기가 필요하다면 LangGraph로도 동일한 아키텍처(SLM=tool, Claude=node)를 구성할 수 있습니다. 이 킷은 Strands 경로를 완성형으로 제공하고, LangGraph는 대안으로 남겨 둡니다.

---

## Bedrock Claude 모델 ID 규칙

- Bedrock Claude는 **[`converse` API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)** 로 호출합니다.
- 모델 ID는 **inference-profile prefix**를 붙인 형식을 씁니다: `us.`/`eu.`/`apac.`/`global.` (지역 라우팅). AWS는 이를 [cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)로 문서화합니다.
- 이 킷의 기본값은 `common/config.py`의 `BEDROCK_CLAUDE_MODEL_ID = "global.anthropic.claude-sonnet-5"`이며, 이 값은 **해당 계정에서 `list_inference_profiles`로 확인**한 것입니다(실측 2026-07). 최신 Claude는 날짜 없는 pinned-snapshot 형식을 씁니다.
- 모델 로스터는 계정·리전마다 다르므로 **실행 전 재확인**하고 env로 주입하세요. 코드에는 **절대 하드코딩하지 마세요**.
    - `common/config.py`의 `BEDROCK_CLAUDE_MODEL_ID`(env `BEDROCK_CLAUDE_MODEL_ID`)를 사용합니다. 호출 리전은 `BEDROCK_REGION`(기본값 `AWS_REGION`)입니다.
    - `agentcore/app.py`는 `os.environ["BEDROCK_CLAUDE_MODEL_ID"]`로 반드시 주입받고(미설정이면 즉시 실패), `agentcore/templates/load.py`는 env가 없을 때만 킷 기본값으로 폴백합니다.

??? question "오개념 — “모델 ID에 왜 `us.`가 붙나요? 그냥 `anthropic.claude-...`면 안 되나요?”"
    안 될 수 있습니다. 최신 Claude는 대부분 **cross-region inference profile**을 통해 호출하며, 그때 지역 prefix(`us.`/`eu.`/`apac.`/`global.`)가 필요합니다.
    prefix 없는 순수 모델 ID로는 호출이 거부될 수 있습니다. 어떤 형식이 유효한지는 리전·모델별로 다르니 **실행 전 재확인**하세요.

---

## 프로덕션 배포 — AgentCore Runtime

!!! abstract "쉽게 말하면"
    - 로컬에서 `python app.py`로 잘 도는 에이전트를 AWS 관리형 런타임에 컨테이너로 올립니다.
    - 런타임은 정해진 **HTTP 계약**으로만 에이전트를 호출합니다.
    - 계약을 지키는 서버 코드는 SDK가 대신 만들어 주므로, 우리가 쓸 것은 진입점 함수 하나입니다.

### HTTP 계약과 SDK

`agentcore/app.py`와 `agentcore/Dockerfile` 기준이며, 계약의 원본은 [Bedrock AgentCore 개발자 가이드](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)입니다.

- **ARM64** 컨테이너를 사용합니다(`FROM --platform=linux/arm64`).
- HTTP는 **`POST /invocations`**(호출)과 **`GET /ping`**(헬스체크)을 **port 8080**에서 제공합니다.
- SDK는 [`bedrock-agentcore`](https://github.com/aws/bedrock-agentcore-sdk-python)를 쓰며, `BedrockAgentCoreApp()` + `@app.entrypoint` + `app.run()` 조합으로 구성합니다.
- CLI는 현행 권장 배포 흐름인 **`@aws/agentcore` (npm CLI)** 를 사용합니다 — `agentcore create/dev/deploy/invoke`. 구 [`bedrock-agentcore-starter-toolkit`](https://github.com/aws/bedrock-agentcore-starter-toolkit)(`agentcore configure/launch`)는 더 이상 권장되지 않으며 참고용입니다.

```python
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict) -> dict:
    result = _agent(payload.get("prompt", ""))   # Strands agent
    return {"result": str(result)}

if __name__ == "__main__":
    app.run()   # /invocations + /ping on :8080
```

CLI가 생성하는 스캐폴딩은 import 경로가 `from bedrock_agentcore.runtime import BedrockAgentCoreApp`이고 진입점이 `async def invoke(payload, context)`로 스트리밍(`agent.stream_async`)입니다. 킷의 `agentcore/templates/main.py`가 그 형태입니다.

!!! warning "배포 전 재확인 목록"
    AgentCore의 **GA 상태·지원 리전**, `bedrock-agentcore` SDK의 **import 경로/데코레이터 시그니처**, `@aws/agentcore` CLI **버전**(실측 2026-07: v0.24.2), base 이미지 태그는 빠르게 바뀝니다.
    `agentcore/app.py`와 `agentcore/Dockerfile`에 `# TODO verify` 주석으로 표시된 지점을 배포 전에 반드시 확인하세요.

### CLI 배포 절차

노트북 셀이 아니라 **터미널**에서 진행합니다(대화형 프롬프트·장시간 dev 서버·PATH 연속성 때문입니다. 셀의 `!명령`은 매번 새 셸이라 nvm PATH가 안 이어져 `agentcore: command not found`가 납니다).

```bash
bash agentcore/setup_agentcore_cli.sh          # Node >= 20 + @aws/agentcore (nvm, sudo 불필요)
source $HOME/.nvm/nvm.sh && nvm use 20
bash agentcore/create_agent.sh                 # non-interactive 생성 + SLM tool 자동 이식 + uv sync
bash agentcore/verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION]   # 로컬 dev 서버로 실제 추론 검증
cd agentcore/gemmaextraction && agentcore deploy                  # ARM64 → ECR → Runtime (CDK)
agentcore invoke --prompt '...'                                   # 배포된 Runtime 호출
```

- `@aws/agentcore`는 **Node.js 20 이상**이 필요합니다. 18 이하면 `EBADENGINE` 경고와 런타임 오류가 나고, `/usr/local` 전역 설치는 `EACCES` 권한 오류가 납니다. `setup_agentcore_cli.sh`가 nvm으로 홈에 Node 20을 깔아 두 문제를 모두 피합니다.
- `create_agent.sh`는 agent-path flag(`--framework`)와 harness-only flag(`--model-id`)를 **섞을 수 없다는 CLI 제약**(실측)을 반영해, 모델 ID는 생성된 `model/load.py`에서 env로 받게 이식합니다. 프로젝트 이름은 영숫자만 허용하고 **23자 이하**여야 합니다(CLI 제약, 실측).
- `verify_local.sh`는 dev 서버를 `setsid`로 띄우면서 stdin을 `/dev/null`로 분리하고(터미널 점유·stdin 문제 회피), 종료는 `kill <pid>`로 정밀하게 합니다. `pkill -f 'agentcore dev'`는 실행 중인 셸까지 죽입니다(실측).
- CLI를 쓰지 않는 경로는 ARM64 이미지를 ECR에 푸시한 뒤 `bedrock-agentcore-control`의 `create_agent_runtime`을 직접 호출하는 것입니다. 호출은 `bedrock-agentcore`의 `invoke_agent_runtime(agentRuntimeArn=..., runtimeSessionId=<33자 이상>, payload=..., qualifier="DEFAULT")`입니다. 파라미터 스키마가 바뀔 수 있으니 최신 boto3 레퍼런스에서 확인하세요.

??? question "오개념 — “x86 이미지로 빌드하면 안 되나요?”"
    안 됩니다. AgentCore Runtime은 **ARM64**를 요구합니다. `docker buildx build --platform linux/arm64`로 빌드해야 하며, x86 이미지는 거부될 수 있습니다.

계약의 나머지 절반인 헬스체크도 자주 오해를 받습니다.

??? question "오개념 — “`/ping`은 왜 필요한가요?”"
    런타임이 컨테이너의 헬스를 체크하는 엔드포인트입니다. `BedrockAgentCoreApp.run()`이 `/invocations`와 함께 자동으로 제공하므로 직접 구현할 필요는 없습니다.

자세한 배포 절차는 노트북 **`06_agentcore_deploy`** 와 `agentcore/app.py`·`agentcore/Dockerfile`을 참고하세요. `agentcore/app.py`의 스캐폴드는 **정보추출 트랙 전용**(tool = `extract_structured_json`)이므로, 다른 트랙에 쓸 때는 tool 함수와 system 프롬프트를 그 트랙 것으로 바꿔야 합니다.

---

## 자주 나오는 오개념

아래 두 항목은 앞 절에서 다루지 않은, 서비스 계층을 헷갈릴 때 생기는 착각입니다.

??? question "오개념 — “SageMaker deployment guardrail(blue/green·canary·rolling)을 이 agentic 배포에 쓴다”"
    그렇지 않습니다. 이 배포 가드레일은 **SageMaker classic endpoint 업데이트** 기능이지 AgentCore Runtime이나 Strands의 기능이 아닙니다.
    SLM endpoint를 무중단 갱신할 때는 쓸 수 있지만, 에이전트 컨테이너 배포와는 무관합니다.

프레임워크와 게이트웨이를 같은 층으로 보는 착각도 같은 유형입니다.

??? question "오개념 — “Strands = LiteLLM”"
    아닙니다. Strands는 agent framework이고 LiteLLM은 모델 게이트웨이입니다.
    Strands의 provider 중 하나로 `LiteLLMModel`을 **쓸 수 있는** 관계일 뿐입니다([provider 선택](#provider-선택--bedrockmodel과-litellmmodel)).

---

## 비용과 cleanup

agentic loop는 과금 소스가 **둘 이상**입니다.

!!! danger "비용과 cleanup"
    endpoint를 켜둔 채 잊어버리는 것이 이 킷에서 가장 흔한 비용 사고입니다. GPU 인스턴스는 **삭제할 때까지 시간당 과금**되며, AgentCore Runtime은 그와 **별개로** 과금됩니다.
    실습이 끝나면 반드시 `99_cleanup`을 실행하고, AgentCore를 배포했다면 `cleanup_agent.sh --aws`까지 실행하세요. `sm.list_endpoints()`로 남은 endpoint가 없는지 확인하세요.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker real-time endpoint | 삭제 전까지 시간당(GPU 인스턴스) | `99_cleanup` 또는 `predictor.delete_endpoint()` |
| Bedrock Claude | 토큰당(호출량) | 상시 리소스 없음. 대량 호출 시 비용 |
| AgentCore Runtime + ECR 이미지 | Runtime 리소스 + 이미지 스토리지 | `bash agentcore/cleanup_agent.sh --aws` (`agentcore destroy`) |

- 로컬만 정리하려면 `bash agentcore/cleanup_agent.sh`(dev 서버 프로세스 + 프로젝트 폴더)를 쓰고, AWS 배포 리소스까지 지우려면 `--aws`를 붙이세요. `agentcore destroy`가 실패하면 생성된 프로젝트 안의 `agentcore/cdk`(예: `agentcore/gemmaextraction/agentcore/cdk`)에서 `npx cdk destroy`로, 그것도 안 되면 콘솔에서 Runtime·ECR·CloudFormation 스택을 직접 삭제합니다.
- 프로젝트 폴더를 남겨 두려면 `KEEP_PROJECT=1`을 붙이세요. dev 서버 종료도 `kill <pid>` 방식이라 실행 중인 셸을 죽이지 않습니다.
- 상태와 비용은 `common/aws_utils.py`의 `print_cost_warning()`과 `cw_links()`로 확인할 수 있습니다.
