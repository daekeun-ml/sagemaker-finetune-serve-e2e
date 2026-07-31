"""
agentcore/app.py — AgentCore Runtime 엔트리포인트 스캐폴드

Strands 에이전트(Bedrock Claude reasoning + 파인튜닝 SLM endpoint tool)를 AgentCore Runtime에
호스팅하기 위한 HTTP 계약(/invocations POST + /ping GET, port 8080)을 제공한다.

근거 (litellm/agentic 정찰 2026-07 검증):
  - bedrock-agentcore SDK: BedrockAgentCoreApp() + @app.entrypoint + app.run().
  - 현행 권장 배포 = @aws/agentcore npm CLI (agentcore create/dev/deploy/invoke).
  - ARM64 컨테이너, /invocations + /ping, port 8080.

⚠️ AgentCore·Strands·Bedrock 모델 ID는 빠르게 바뀜 → 배포 전 리전·GA·버전 재확인.
🔴 시크릿·모델 ID·endpoint 이름은 환경변수로 주입 (하드코딩 금지).
"""
from __future__ import annotations

import json
import os

import boto3
from strands import Agent, tool
from strands.models import BedrockModel

# --- 환경변수 (배포 시 AgentCore/컨테이너 env로 주입) ---
ENDPOINT_NAME = os.environ["SLM_ENDPOINT_NAME"]                      # 파인튜닝 SLM endpoint
BEDROCK_MODEL_ID = os.environ["BEDROCK_CLAUDE_MODEL_ID"]             # inference-profile ID (하드코딩 금지)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = (
    "You are a precise information-extraction engine. Read the user text and the available "
    'tool schema, then output ONLY a valid JSON object {"name": ..., "arguments": {...}}.'
)


@tool
def extract_structured_json(text: str) -> str:
    """Extract structured JSON from text using the fine-tuned Gemma SLM (SageMaker endpoint)."""
    # 🔴 endpoint 호출 = sagemaker-runtime (Bedrock 아님).
    # 🔴 messages 형식으로 보낸다 → endpoint 핸들러(inference.py)가 서버측에서 chat template을 적용.
    #    raw 텍스트를 inputs로 직송하면 template 미적용 → degenerate/빈 응답(실측). 로컬 tokenizer 불필요.
    rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        "max_new_tokens": 256, "temperature": 0.1,
    }
    resp = rt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(resp["Body"].read().decode("utf-8"))
    if isinstance(body, list) and body and "generated_text" in body[0]:
        return body[0]["generated_text"]
    return json.dumps(body)


# reasoning = Bedrock Claude
_model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
_agent = Agent(
    model=_model,
    tools=[extract_structured_json],
    system_prompt=(
        "You orchestrate. When the user gives text needing structured extraction, call "
        "extract_structured_json, then validate/explain the returned JSON."
    ),
)


# --- AgentCore Runtime 호스팅 ---
# ⚠️ import 경로/데코레이터는 bedrock-agentcore SDK 버전에 맞춰 재확인 (# TODO verify).
try:
    from bedrock_agentcore import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload: dict) -> dict:
        """AgentCore Runtime이 호출하는 진입점. payload={'prompt': ...} 가정."""
        prompt = payload.get("prompt", "")
        result = _agent(prompt)
        # Strands 결과 객체 → 문자열
        return {"result": str(result)}

    if __name__ == "__main__":
        app.run()  # /invocations + /ping on :8080

except ImportError:
    # 로컬 테스트 폴백 (bedrock-agentcore 미설치 시)
    if __name__ == "__main__":
        import sys

        prompt = sys.argv[1] if len(sys.argv) > 1 else "Extract a tool call: book_table for 2 at 8pm."
        print(_agent(prompt))
