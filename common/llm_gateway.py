"""
common/llm_gateway.py — LiteLLM 통합 게이트웨이 (Bedrock + SageMaker endpoint 단일 인터페이스)

왜 LiteLLM인가:
  - Bedrock Claude(reasoning)와 SageMaker endpoint(파인튜닝 SLM)를 **하나의 OpenAI 호환
    completion() 인터페이스**로 호출 → 프로바이더 교체가 model 문자열만 바꾸면 됨(이식성).
  - 합성 데이터/평가/agentic에서 모델 백엔드를 통일.

라우팅 규약 (LiteLLM 실측 검증 2026-07, docs.litellm.ai):
  - Bedrock Claude : "bedrock/us.anthropic.claude-..."  (converse 강제: "bedrock/converse/us.anthropic...")
                     geo prefix(us./eu./apac./global.)는 model 문자열에 인라인.
  - SageMaker EP   : "sagemaker_chat/<endpoint>"  (endpoint가 messages/OpenAI 호환일 때 — 서버측 chat template)
                     또는 "sagemaker/<endpoint>" + hf_model_name=... (클라이언트측 프롬프트 포매팅)
  - creds/region  : env(AWS_ACCESS_KEY_ID/SECRET/AWS_REGION_NAME) 또는 completion() 파라미터(aws_region_name 등).

⚠️ litellm은 빠르게 바뀜(검증 시 1.93.0, 1.94 프리릴리스 존재) → requirements에서 pin, 실행 전 재확인.
🔴 Bedrock 모델 ID는 하드코딩 금지 — inference-profile ID를 env/param으로.
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
