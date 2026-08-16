"""Gemma 학습과 추론에 사용할 표준 messages 형식을 만듭니다.

채팅 마커는 직접 조립하지 않고 토크나이저의 `apply_chat_template`에 맡깁니다.
"""
from __future__ import annotations

from typing import Any


def build_messages(
    user_content: str,
    assistant_content: str,
    system_content: str | None = None,
) -> list[dict[str, str]]:
    """단일 학습 예시를 표준 messages 리스트로 만듭니다.

    SFT 학습용. tokenizer.apply_chat_template(messages, tokenize=False)에 넣으면
    Gemma 턴 마커가 자동 렌더링된다. role 이름은 표준(user/assistant/system)을 쓰고,
    Gemma 템플릿이 내부에서 'model'로 매핑한다.
    """
    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_inference_messages(
    user_content: str,
    system_content: str | None = None,
) -> list[dict[str, str]]:
    """assistant 턴이 없는 추론용 messages를 만듭니다."""
    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """apply_chat_template로 추론 프롬프트 문자열 생성 (add_generation_prompt=True).

    endpoint에 raw text를 보낼 때 사용. tokenizer는 학습에 쓴 것과 동일해야 함.
    system role을 지원하지 않는 Gemma 템플릿이면 여기서 예외가 날 수 있으므로,
    호출부에서 system을 첫 user 턴에 병합하는 폴백을 둔다(fold_system_into_user).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def fold_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """system role 미지원 템플릿용 폴백: system 내용을 첫 user 턴 앞에 병합.

    apply_chat_template이 system role에서 에러를 내면 이걸로 재시도.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system = messages[0]["content"]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            merged = dict(m)
            merged["content"] = f"{system}\n\n{m['content']}"
            return rest[:i] + [merged] + rest[i + 1 :]
    # user 턴이 없으면 그대로
    return rest
