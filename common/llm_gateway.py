"""Bedrock과 SageMaker 엔드포인트를 LiteLLM 인터페이스로 호출합니다.

Bedrock 모델 ID와 리전은 호출부에서 전달하며 코드에 고정하지 않습니다.
"""
from __future__ import annotations

from typing import Any


def bedrock_model_str(model_id: str, force_converse: bool = True) -> str:
    """inference-profile ID → LiteLLM bedrock 라우팅 문자열.

    force_converse=True 면 'bedrock/converse/' prefix로 Converse API를 명시적으로 강제
    (미지정 'bedrock/'도 지원 모델은 converse를 쓰지만 버전 의존적이라 명시 권장).
    """
    core = model_id[len("bedrock/"):] if model_id.startswith("bedrock/") else model_id
    return f"bedrock/converse/{core}" if force_converse else f"bedrock/{core}"


def sagemaker_model_str(endpoint_name: str, chat: bool = True) -> str:
    """SageMaker endpoint → LiteLLM 라우팅 문자열.

    chat=True  → 'sagemaker_chat/<ep>' (endpoint가 messages/OpenAI 호환 서빙, 서버측 chat template).
    chat=False → 'sagemaker/<ep>' (텍스트 경로; 호출 시 hf_model_name= 로 chat template 지정 필요).
    """
    return f"sagemaker_chat/{endpoint_name}" if chat else f"sagemaker/{endpoint_name}"


def chat(
    model: str,
    messages: list[dict[str, str]],
    region: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    hf_model_name: str | None = None,
    **kwargs: Any,
) -> str:
    """LiteLLM completion 래퍼. model 은 위 helper가 만든 라우팅 문자열.

    예:
        chat(bedrock_model_str(cfg.BEDROCK_CLAUDE_MODEL_ID), msgs, region=cfg.AWS_REGION)
        chat(sagemaker_model_str(EP, chat=False), msgs, region=REGION, hf_model_name="google/gemma-3-4b-it")
    """
    from litellm import completion

    params: dict[str, Any] = dict(
        model=model,
        messages=messages,
        aws_region_name=region,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    # 'sagemaker/' 텍스트 경로에서만 hf_model_name으로 올바른 chat template 적용
    if hf_model_name and model.startswith("sagemaker/") and not model.startswith("sagemaker_chat/"):
        params["hf_model_name"] = hf_model_name

    resp = completion(**params)
    return resp["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 편의: config 기반 원샷 헬퍼
# ---------------------------------------------------------------------------
def bedrock_chat(user_text: str, region: str, model_id: str, system_text: str | None = None, **kw) -> str:
    """Bedrock Claude 원샷 (LiteLLM 경유). aws_utils.bedrock_converse(boto3 직접)의 LiteLLM 대안."""
    msgs = ([{"role": "system", "content": system_text}] if system_text else []) + [
        {"role": "user", "content": user_text}
    ]
    return chat(bedrock_model_str(model_id), msgs, region=region, **kw)


def endpoint_chat(user_text: str, endpoint_name: str, region: str, chat_route: bool = True,
                  hf_model_name: str | None = None, **kw) -> str:
    """파인튜닝 SLM endpoint 원샷 (LiteLLM 경유)."""
    msgs = [{"role": "user", "content": user_text}]
    return chat(sagemaker_model_str(endpoint_name, chat=chat_route), msgs, region=region,
                hf_model_name=hf_model_name, **kw)
