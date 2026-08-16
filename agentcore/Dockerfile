# AgentCore Runtime은 linux/arm64 이미지를 사용합니다.
# bedrock-agentcore SDK가 포트 8080의 /invocations와 /ping을 제공합니다.
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

# 런타임 의존성
RUN pip install --no-cache-dir \
    bedrock-agentcore \
    'strands-agents>=1.48.0' \
    boto3

COPY app.py /app/app.py

EXPOSE 8080

# BedrockAgentCoreApp.run() 이 8080에서 /invocations + /ping 서빙
CMD ["python", "app.py"]
