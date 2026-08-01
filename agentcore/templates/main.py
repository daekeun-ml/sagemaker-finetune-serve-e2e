"""
main.py — AgentCore 에이전트 엔트리포인트 (이 킷용: Bedrock Claude reasoning + SLM endpoint tool).

create_agent.sh가 `agentcore create` 스캐폴딩 생성 후, 이 파일을 app/<agent>/main.py 로 복사한다.
CLI 기본 스캐폴딩의 데모 tool(add_numbers)을, 파인튜닝 SLM(SageMaker endpoint)을 호출하는
extract_structured_json tool로 교체한 버전이다.

🔴 모델 ID·endpoint 이름은 하드코딩 금지 → env로 주입:
   SLM_ENDPOINT_NAME, AWS_REGION, BEDROCK_CLAUDE_MODEL_ID (model/load.py 참고).
"""
from typing import Any
import json
import os
from collections import OrderedDict
import boto3
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model

app = BedrockAgentCoreApp()
log = app.logger

DEFAULT_SYSTEM_PROMPT = """
You orchestrate. When the user gives text needing structured extraction, call
extract_structured_json to get the JSON from the fine-tuned SLM, then validate/explain it.
"""

# 🔴 SLM에 보낼 때 쓰는 system 프롬프트(학습 때와 동일해야 함). endpoint의 chat template로 렌더된다.
SLM_SYSTEM_PROMPT = (
    "You are a precise information-extraction engine. Read the user text and the available "
    'tool schema, then output ONLY a valid JSON object {"name": ..., "arguments": {...}}.'
)

# --- 환경변수 (배포 시 AgentCore/컨테이너 env로 주입; 하드코딩 금지) ---
SLM_ENDPOINT_NAME = os.environ.get("SLM_ENDPOINT_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

tools = []
_INLINE_FUNCTION_NAMES = set()


# 🔴 파인튜닝 SLM(SageMaker endpoint)을 tool로 래핑. endpoint 호출 = sagemaker-runtime(Bedrock 아님).
#    🔴 messages 형식으로 보낸다 → endpoint 핸들러(inference.py)가 서버측에서 chat template을 적용한다.
#    (raw 텍스트를 inputs로 직송하면 template 미적용 → degenerate/빈 응답. 로컬 tokenizer/torch 불필요.)
@tool
def extract_structured_json(text: str) -> str:
    """Extract structured JSON from text using the fine-tuned Gemma SLM (SageMaker endpoint)."""
    if not SLM_ENDPOINT_NAME:
        return '{"error": "SLM_ENDPOINT_NAME env not set"}'
    rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    # 🔴 messages 스키마의 생성 한도 키는 max_tokens (OpenAI 호환). max_new_tokens는
    #    {"inputs","parameters"} 스키마 쪽 이름이라 vLLM/SGLang/LMI가 무시한다 → 한도가 안 걸린다.
    #    256은 추출·분류 트랙 값(요약·도메인 QA는 512).
    payload = {"messages": [{"role": "system", "content": SLM_SYSTEM_PROMPT},
                            {"role": "user", "content": text}],
               "max_tokens": 256, "temperature": 0.1}
    resp = rt.invoke_endpoint(EndpointName=SLM_ENDPOINT_NAME,
                              ContentType="application/json", Body=json.dumps(payload))
    body = json.loads(resp["Body"].read().decode("utf-8"))
    # HF DLC 핸들러 응답: [ '{"generated_text": "..."}', "application/json" ] 형태 방어적 파싱
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return body
    if isinstance(body, dict):
        gt = body.get("generated_text")
        if isinstance(gt, str):
            s = gt.strip()
            if s.startswith("{") and '"generated_text"' in s:  # 이중 래핑 방지
                try:
                    return json.loads(s).get("generated_text", gt)
                except (json.JSONDecodeError, ValueError):
                    pass
            return gt
    return json.dumps(body, ensure_ascii=False)
tools.append(extract_structured_json)


def _make_conversation_manager():
    return NullConversationManager()


def agent_factory():
    cache = OrderedDict()
    def get_or_create_agent(session_id):
        if session_id in cache:
            cache.move_to_end(session_id)
            return cache[session_id]
        if len(cache) >= 128:
            cache.popitem(last=False)
        cache[session_id] = Agent(
            model=load_model(),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools=tools,
            conversation_manager=_make_conversation_manager(),
            hooks=[],
        )
        return cache[session_id]
    return get_or_create_agent
get_or_create_agent = agent_factory()


def _extract_prompt(payload: dict):
    """Accept harness-style messages[], tool_results[], or plain prompt string payloads."""
    if "messages" in payload:
        return payload["messages"]
    if "tool_results" in payload:
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in payload["tool_results"]]}]
    return payload.get("prompt", "")


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")
    session_id = getattr(context, "session_id", "default-session")
    agent = get_or_create_agent(session_id)
    prompt = _extract_prompt(payload)
    async for event in agent.stream_async(prompt):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event


if __name__ == "__main__":
    app.run()
