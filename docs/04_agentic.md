# 04 · Agentic Loop — 파인튜닝 SLM(tool) + Bedrock Claude(reasoning)

> **대상 독자**: 이 킷으로 이미 Gemma SLM을 학습(`02`)·배포(`03`)해 real-time endpoint를 가진 분을 위한 문서입니다. Strands/AgentCore는 처음이어도 괜찮습니다.
> **⚠️ 주의**: 🔴 표시는 **빠르게 바뀌는 사실**(모델 ID·이미지 태그·SDK 버전·리전·GA 상태)을 뜻하므로 **실행 직전에 반드시 다시 확인**하세요. 시크릿·계정 ID·endpoint 이름은 코드에 하드코딩하지 말고 env/param으로 주입하세요.
> **라이브 검증 2026-07** (세부 조합은 실행 전에 재확인하세요).

---

## 0. TL;DR (한 줄)

**파인튜닝 Gemma SLM은 빠른 "전문가 도구(tool)", Bedrock Claude는 범용 "추론·오케스트레이션(reasoning)" 역할을 맡습니다. agent framework(Strands 우선, LangGraph 옵션)가 tool-use로 이 둘을 묶습니다. endpoint는 `sagemaker-runtime`, Bedrock은 `bedrock-runtime`으로 서로 다른 서비스이므로 절대 섞어서 부르면 안 됩니다.**

세부 결론은 다음과 같습니다.
1. SLM endpoint는 `@tool`로 감싼 `call_slm`(내부는 `sagemaker-runtime invoke_endpoint`)으로 노출합니다. Claude가 "언제 이 도구를 부를지"를 판단합니다.
2. Strands는 `BedrockModel`을 기본 provider로 사용합니다(reasoning=Claude). 멀티 프로바이더가 필요하면 `LiteLLMModel` 옵션을 쓰세요.
3. Bedrock Claude 모델 ID는 **inference-profile prefix**(`us.`/`eu.`/`apac.`/`global.`) 형식이며, env/param으로 주입해야 합니다. 하드코딩은 금지입니다.
4. 프로덕션 배포는 **AgentCore Runtime**(ARM64 컨테이너, `/invocations`+`/ping`:8080)으로 진행합니다. 🔴 GA와 리전은 배포 전에 다시 확인하세요.
5. 과금이 **두 군데**(endpoint 시간당 + Bedrock 토큰당)에서 발생하므로 cleanup이 필수입니다.

관련 파일과 문서는 다음과 같습니다: `agentcore/app.py`(엔트리포인트 스캐폴드), `common/aws_utils.py`(서비스 경계), `common/llm_gateway.py`(LiteLLM 단일 인터페이스), 그리고 노트북 `05_agentic_strands` → `06_agentcore_deploy`.

---

## 0.5 기존 Pain Point

`03`까지 진행했다면 대개 이런 상태일 것입니다.

- SLM endpoint는 있지만 **"그래서 이걸 어떻게 앱으로 쓰지?"** 하는 고민이 생깁니다. endpoint는 프롬프트를 주면 텍스트를 뱉는 함수일 뿐, 스스로 "언제 나를 부를지"를 판단하지 못하기 때문입니다.
- 반대로 Bedrock Claude는 범용 추론은 잘하지만, **우리 도메인 전용 구조화 추출/분류는 파인튜닝한 SLM이 더 싸고 빠르고 정확**합니다.
- 흔한 오해로 "그럼 Claude한테 endpoint를 Bedrock API로 부르라고 하면 되지 않나?"라고 생각하기 쉽지만, 이는 **틀린 접근**입니다. endpoint와 Bedrock은 별개 서비스입니다(§2).
- 또 다른 함정은, 로컬에서 `python app.py`로 잘 돌던 에이전트를 그대로 프로덕션에 올리려다 **AgentCore Runtime의 HTTP 계약(ARM64·`/invocations`·`/ping`·8080)** 을 몰라서 막히는 경우입니다.

이 문서는 "endpoint + Claude"를 하나의 **agentic loop**로 묶고, 로컬에서 AgentCore Runtime까지 올리는 최소 경로를 정리합니다.

---

## 1. Why — 왜 SLM을 "도구", Claude를 "두뇌"로 나누나?

### 쉽게 말하면
- **Claude(reasoning LLM)** 는 전체 문제를 이해하고 "지금 무슨 도구가 필요한가"를 결정하는 **팀장**에 해당합니다.
- **파인튜닝 Gemma SLM(tool)** 은 한 가지 일(예: 텍스트→JSON 추출)을 아주 빠르고 정확하게 처리하는 **전문가 사원**에 해당합니다.
- **agent framework(Strands)** 는 팀장이 사원에게 일을 시키는 **결재 라인(tool-use 프로토콜)** 역할을 합니다.

### 대조표 — reasoning LLM vs specialist SLM

| 축 | Bedrock Claude (reasoning) | 파인튜닝 Gemma SLM (tool) |
|---|---|---|
| 역할 | 계획·판단·도구 선택·결과 검증 | 도메인 특화 단일 작업 실행 |
| 호출 서비스 | `bedrock-runtime` → `converse()` | `sagemaker-runtime` → `invoke_endpoint()` |
| 과금 모델 | 토큰당(사용량) | endpoint 시간당(상시 인스턴스) |
| 강점 | 범용 추론·멀티스텝·자연어 오케스트레이션 | 전용 태스크의 속도·비용·포맷 안정성 |
| 모델 관리 | AWS 관리형(모델 ID만 지정) | 우리가 학습·배포·운영 |
| 커스터마이즈 | 프롬프트/few-shot 위주 | 가중치 자체를 파인튜닝(LoRA) |

> ❓ **"Claude 하나로 다 하면 안 되나요?"** — 됩니다. 다만 도메인 전용 추출/분류에서는 (1) 작은 파인튜닝 SLM이 **더 저렴하고 지연도 낮으며** (2) 출력 포맷이 **더 안정적**입니다. 반대로 SLM 하나로 다중 스텝 추론을 시키기는 어렵습니다. 그래서 **역할 분담**이 핵심입니다.

### 기술적 차이 3가지
1. **서비스 경계**: endpoint(`sagemaker-runtime`)와 Bedrock(`bedrock-runtime`)은 클라이언트도 API도 다릅니다(§2). 프레임워크는 이 둘을 각각 다른 통합으로 호출합니다.
2. **제어 주체**: Claude가 "도구 호출 여부와 인자"를 결정하면, 프레임워크가 실제 `call_slm`을 실행한 뒤 결과를 다시 Claude에게 돌려줍니다(tool-use round-trip).
3. **배포 단위**: SLM은 SageMaker endpoint(개별 리소스)로, 에이전트 전체는 AgentCore Runtime(컨테이너)으로 각각 별도 배포됩니다.

---

## 2. 🔴 서비스 경계 — endpoint ≠ Bedrock (절대 섞지 말 것)

이 킷에서 가장 흔하게 사고가 나는 지점입니다. `common/aws_utils.py` 상단에도 못 박아 둔 규칙은 다음과 같습니다.

```
SageMaker endpoint 호출  = boto3 "sagemaker-runtime" 클라이언트, invoke_endpoint()
Bedrock Claude 호출       = boto3 "bedrock-runtime" 클라이언트, converse()
→ 별개 서비스 · 별개 클라이언트. "endpoint를 Bedrock API로 호출"은 잘못.
```

```
사용자 입력
   │
   ▼
┌─────────────────────────────┐
│  Bedrock Claude (reasoning)  │  bedrock-runtime.converse()
│  "이건 추출 작업 → 도구 호출" │
└──────────────┬──────────────┘
               │ tool-use 요청 (name=call_slm, args=...)
               ▼
┌─────────────────────────────┐
│  @tool call_slm              │  sagemaker-runtime.invoke_endpoint()
│  파인튜닝 Gemma SLM endpoint  │
└──────────────┬──────────────┘
               │ 결과(JSON 등)
               ▼
   Claude가 결과 검증·설명 → 최종 응답
```

- endpoint 호출 스키마는 서빙 컨테이너를 따릅니다: `{"inputs", "parameters"}` (HF TGI/DJL 관용). 스트리밍이 필요하면 `invoke_endpoint_with_response_stream`을 사용하세요.
- Bedrock은 `converse()`(또는 스트리밍 `converse_stream()`)를 쓰며, 메시지·`inferenceConfig` 스키마를 따릅니다.
- 실제 구현은 `common/aws_utils.py`의 `invoke_sagemaker_endpoint()`와 `bedrock_converse()`를 참고하세요.

> ❓ **"LiteLLM 쓰면 둘이 같은 거 아닌가요?"** — `common/llm_gateway.py`의 LiteLLM은 **호출 인터페이스만** OpenAI 호환 `completion()`으로 통일해 줄 뿐, 내부적으로는 여전히 `bedrock/...`와 `sagemaker_chat/<ep>`로 **다른 백엔드에 라우팅**합니다. "같은 서비스"가 된 것이 아니라 "같은 함수 시그니처로 부를 수 있게" 감싼 것일 뿐입니다.

### SageMaker 추론 4옵션 (참고)
SLM을 어디에 배포할지에도 선택지가 있습니다(`03`에서는 real-time을 선택했습니다).

| 옵션 | 특징 | LLM/SLM 적합성 |
|---|---|---|
| Real-time | 상시 endpoint, 저지연 | 이 킷 기본. 시간당 과금 |
| Serverless | 온디맨드, 스케일-투-제로 | ⚠️ **GPU 없음 → LLM/SLM 부적합** |
| Asynchronous | 큐 기반, 대용량/긴 추론 | 배치성 추론에 |
| Batch Transform | 잡 단위 대량 오프라인 | 실시간 agentic엔 부적합 |

> ❓ **"비용 아끼려고 Serverless에 SLM 올리면 어떨까요?"** — 현재 SageMaker Serverless Inference는 **GPU를 제공하지 않아** Gemma 같은 SLM 서빙에는 부적합합니다. agentic loop의 tool은 real-time endpoint를 전제로 합니다. (🔴 GPU 지원 여부는 정책이 바뀔 수 있으니 다시 확인하세요.)

---

## 3. Strands로 묶기 — `@tool`로 `call_slm` 감싸기

### 쉽게 말하면
Strands에서는 (1) 파이썬 함수에 `@tool` 데코레이터만 붙이면 그것이 "도구"가 되고, (2) `Agent(model=..., tools=[...])`에 넘기면 Claude가 필요할 때 알아서 호출합니다. `agentcore/app.py`의 실제 패턴은 다음과 같습니다.

```python
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def call_slm(text: str) -> str:
    """파인튜닝 Gemma SLM(SageMaker endpoint)로 구조화 추출."""
    rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)  # 🔴 Bedrock 아님
    resp = rt.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="application/json",
                              Body=json.dumps({"inputs": text, "parameters": {...}}))
    ...

model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)  # reasoning=Claude
agent = Agent(model=model, tools=[call_slm], system_prompt="You orchestrate...")
```

### provider 선택: BedrockModel(기본) vs LiteLLMModel(옵션)

| 상황 | provider |
|---|---|
| Bedrock Claude만 쓴다 (이 킷 기본) | `strands.models.BedrockModel` — 기본 provider |
| 여러 프로바이더(OpenAI/Anthropic API/기타)를 갈아끼워야 한다 | `LiteLLMModel` 옵션 |

> ❓ **"reasoning 모델을 LiteLLM으로 바꾸면 tool 호출 방식도 바뀌나요?"** — 아니요. `@tool call_slm` 내부의 `sagemaker-runtime invoke_endpoint`는 그대로 유지됩니다. provider 교체는 **reasoning LLM 백엔드**만 바꿀 뿐, tool은 여전히 SageMaker endpoint를 그대로 호출합니다.

### LangGraph 옵션
Strands를 우선(권장하며 이 킷의 기본)으로 하되, 이미 LangGraph 그래프 오케스트레이션에 익숙하거나 복잡한 상태 머신·분기가 필요하다면 LangGraph로도 동일한 아키텍처(SLM=tool, Claude=node)를 구성할 수 있습니다. 이 킷은 Strands 경로를 완성형으로 제공하고, LangGraph는 대안으로 남겨 둡니다.

---

## 4. 🔴 Bedrock Claude 모델 ID — inference-profile prefix, 하드코딩 금지

- Bedrock Claude는 **`converse` API**로 호출합니다.
- 모델 ID는 **inference-profile prefix**를 붙인 형식을 씁니다: `us.`/`eu.`/`apac.`/`global.` (지역 라우팅).
  - 형식만 예로 들면 `us.anthropic.claude-...` 와 같습니다. 🔴 정확한 문자열은 호출 시점에 **Bedrock 모델 상세 페이지**에서 확인하세요.
- 코드에는 **절대 하드코딩하지 말고** 다음과 같이 env/param으로 주입하세요.
  - `common/config.py`의 `BEDROCK_CLAUDE_MODEL_ID`(env `BEDROCK_CLAUDE_MODEL_ID`)를 사용합니다.
  - `agentcore/app.py`는 `os.environ["BEDROCK_CLAUDE_MODEL_ID"]`로 주입받습니다.

> ❓ **"모델 ID에 왜 `us.`가 붙나요? 그냥 `anthropic.claude-...`면 안 되나요?"** — 최신 Claude는 대부분 **cross-region inference profile**을 통해 호출하며, 그때 지역 prefix(`us.`/`eu.`/`apac.`/`global.`)가 필요합니다. prefix 없는 순수 모델 ID로는 호출이 안 될 수 있습니다. 🔴 어떤 형식이 유효한지는 리전·모델별로 다르니 다시 확인하세요.

---

## 5. 프로덕션 배포 — AgentCore Runtime

### 쉽게 말하면
로컬에서 `python app.py`로 잘 도는 에이전트를, AWS 관리형 런타임(AgentCore Runtime)에 컨테이너로 올립니다. 런타임은 정해진 **HTTP 계약**을 통해 에이전트를 호출합니다.

### 계약 (agentcore/app.py + Dockerfile 기준)
- **ARM64** 컨테이너를 사용합니다(`--platform=linux/arm64`).
- HTTP는 **`POST /invocations`**(호출)과 **`GET /ping`**(헬스체크)을 **port 8080**에서 제공합니다.
- SDK는 `bedrock-agentcore`를 쓰며, `BedrockAgentCoreApp()` + `@app.entrypoint` + `app.run()` 조합으로 구성합니다.
- CLI는 현행 권장 배포 흐름인 **`@aws/agentcore` (npm CLI)** 를 사용합니다 — `agentcore create/dev/deploy/invoke`.

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

> 🔴 **재확인 필수**: AgentCore의 **GA 상태·지원 리전**, `bedrock-agentcore` SDK의 **import 경로/데코레이터**, `@aws/agentcore` CLI **버전**, base 이미지 태그는 빠르게 바뀝니다. `agentcore/app.py`와 `Dockerfile`에 `# TODO verify` 주석으로 표시된 지점을 배포 전에 반드시 확인하세요.

> ❓ **"x86 이미지로 빌드하면 안 되나요?"** — AgentCore Runtime은 **ARM64**를 요구합니다. 따라서 `docker buildx build --platform linux/arm64`로 빌드해야 하며, x86 이미지는 거부될 수 있습니다.

> ❓ **"`/ping`은 왜 필요한가요?"** — 런타임이 컨테이너의 헬스를 체크하는 엔드포인트입니다. `BedrockAgentCoreApp.run()`이 `/invocations`와 함께 자동으로 제공하므로 직접 구현할 필요는 없습니다.

자세한 배포 절차는 노트북 **`06_agentcore_deploy`** 와 `agentcore/app.py`·`agentcore/Dockerfile`을 참고하세요.

---

## 6. 🔴 비용 & cleanup — 과금이 두 군데서 발생

agentic loop는 과금 소스가 **둘 이상**이므로 특히 주의해야 합니다.

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker real-time endpoint | **삭제 전까지 시간당**(GPU 인스턴스) | `99_cleanup` 또는 `predictor.delete_endpoint()` |
| Bedrock Claude | 토큰당(호출량) | 상시 리소스 없음. 대량 호출 시 비용 |
| AgentCore Runtime | Runtime 리소스 과금 | 미사용 시 정리(배포 리소스 삭제) |

- endpoint를 켜둔 채 잊어버리는 것이 가장 흔한 비용 사고이므로, 실습이 끝나면 **반드시** `99_cleanup`을 실행하세요.
- 상태와 비용은 `common/aws_utils.py`의 `print_cost_warning()`과 `cw_links()`로 확인할 수 있습니다.

---

## ❓ 오개념 노트 (모아보기)

> **1. "endpoint도 Bedrock API로 부를 수 있다."** — ❌. `sagemaker-runtime`(endpoint)와 `bedrock-runtime`(Claude)는 별개의 서비스이자 클라이언트입니다(§2).

> **2. "SageMaker deployment guardrail(blue/green·canary·rolling)을 이 agentic 배포에 쓴다."** — 이 배포 가드레일은 **SageMaker classic endpoint 업데이트** 기능이지 AgentCore Runtime이나 Strands의 기능이 아닙니다. 혼동하지 마세요.

> **3. "Serverless endpoint에 SLM 올려 비용 절감."** — ❌. 현재 SageMaker Serverless는 **GPU를 지원하지 않아** SLM 서빙에 부적합합니다(§2). 🔴 재확인이 필요한 항목입니다.

> **4. "Strands = LiteLLM."** — ❌. Strands는 agent framework이고 LiteLLM은 모델 게이트웨이입니다. Strands의 provider 중 하나로 `LiteLLMModel`을 **쓸 수 있는** 관계일 뿐입니다.

> **5. "AgentCore 컨테이너는 아무 아키텍처나 된다."** — ❌. **ARM64**가 필수이며, `/invocations`+`/ping`:8080 계약도 반드시 지켜야 합니다(§5).

> **6. "모델 ID는 코드에 박아두면 편하다."** — ❌. inference-profile prefix가 리전/버전마다 달라지고 시크릿 관리 원칙에도 어긋납니다. env/param으로만 주입하세요(§4).

---

## 라이브 검증 2026-07 (근거 · 실행 전 재확인)

| 주제 | docs.aws.amazon.com | 공식 GitHub |
|---|---|---|
| SageMaker Runtime `invoke_endpoint` | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html | https://github.com/aws/sagemaker-python-sdk |
| SageMaker 추론 옵션(Real-time/Serverless/Async/Batch) | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-options.html | https://github.com/aws/amazon-sagemaker-examples |
| Bedrock `Converse` API | https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html | https://github.com/aws-samples/amazon-bedrock-samples |
| Bedrock cross-region inference profiles | https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html | — |
| Strands Agents | https://strandsagents.com/ | https://github.com/strands-agents/sdk-python |
| Bedrock AgentCore Runtime | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ | https://github.com/aws/bedrock-agentcore-sdk-python |
| AgentCore CLI (`@aws/agentcore`) | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ | https://github.com/aws/bedrock-agentcore-starter-toolkit |
| LiteLLM (Bedrock/SageMaker 라우팅) | — | https://github.com/BerriAI/litellm |

> 🔴 위 URL·기능·모델 ID·SDK/CLI 버전·GA·리전은 시점에 따라 달라집니다. **실행 직전에 다시 확인하세요** (`# TODO verify`).

---

**이전**: `03_deploy_endpoint` (endpoint 배포) → `04_evaluate` (held-out 평가) · **관련**: `05_agentic_strands`(노트북 실습), `agentcore/app.py`(엔트리포인트) · **다음**: `06_agentcore_deploy` (프로덕션 배포) → `99_cleanup`
