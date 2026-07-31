import os

from strands.models.bedrock import BedrockModel


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials.

    🔴 모델 ID는 하드코딩하지 않고 env(BEDROCK_CLAUDE_MODEL_ID)에서 읽는다(킷 원칙).
    env 미설정 시 킷 기본값(global.anthropic.claude-sonnet-5)로 폴백.
    inference-profile prefix(us./global.) 필수 — 배포 리전에서 list_inference_profiles로 재확인.
    """
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")
    region = os.environ.get("AWS_REGION")  # None이면 SDK가 기본 자격증명 리전 사용
    if region:
        return BedrockModel(model_id=model_id, region_name=region)
    return BedrockModel(model_id=model_id)
