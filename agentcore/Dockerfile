# AgentCore Runtime 컨테이너 — 🔴 반드시 ARM64 (linux/arm64)
# /invocations (POST) + /ping (GET) on port 8080 계약을 bedrock-agentcore SDK가 제공.
# ⚠️ base 이미지·SDK 버전은 배포 전 재확인 (# TODO verify).
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

# 의존성 (버전 pin — 실행 전 재확인)
RUN pip install --no-cache-dir \
    bedrock-agentcore \
    'strands-agents>=1.48.0' \
    boto3

COPY app.py /app/app.py

EXPOSE 8080

# BedrockAgentCoreApp.run() 이 8080에서 /invocations + /ping 서빙
CMD ["python", "app.py"]
