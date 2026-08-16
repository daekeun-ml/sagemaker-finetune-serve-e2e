"""Strands 에이전트를 AgentCore Runtime에 호스팅하는 진입점입니다.

모델 ID, 리전, SageMaker 엔드포인트 이름은 환경변수로 전달합니다.
"""
from __future__ import annotations

import json
import os

import boto3
from strands import Agent, tool
from strands.models import BedrockModel

# 배포 환경변수
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
    # SageMaker Runtime에 messages 형식으로 보내 서버에서 채팅 템플릿을 적용합니다.
    rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        # OpenAI 호환 messages 스키마는 max_tokens를 사용합니다.
        "max_tokens": 256, "temperature": 0.1,
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


# AgentCore Runtime 호스팅
try:
    from bedrock_agentcore import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload: dict) -> dict:
        """AgentCore Runtime이 호출하는 진입점. payload={'prompt': ...} 가정."""
        prompt = payload.get("prompt", "")
        result = _agent(prompt)
        # Strands 결과 객체를 문자열로 변환합니다.
        return {"result": str(result)}

    if __name__ == "__main__":
        app.run()  # /invocations + /ping on :8080

except ImportError:
    # 로컬 테스트 폴백 (bedrock-agentcore 미설치 시)
    if __name__ == "__main__":
        import sys

        prompt = sys.argv[1] if len(sys.argv) > 1 else "Extract a tool call: book_table for 2 at 8pm."
        print(_agent(prompt))
