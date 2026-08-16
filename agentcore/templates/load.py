import os

from strands.models.bedrock import BedrockModel


def load_model() -> BedrockModel:
    """환경변수와 IAM 자격증명으로 Bedrock 모델 클라이언트를 만듭니다."""
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")
    region = os.environ.get("AWS_REGION")  # None이면 SDK가 기본 자격증명 리전 사용
    if region:
        return BedrockModel(model_id=model_id, region_name=region)
    return BedrockModel(model_id=model_id)
