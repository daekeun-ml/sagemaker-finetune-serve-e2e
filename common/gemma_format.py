"""
common/gemma_format.py — Gemma chat 포맷 유틸 (4개 트랙 공용)

핵심 원칙 (정찰 2026-07 검증):
  - Gemma는 <start_of_turn>user ... <end_of_turn> / <start_of_turn>model ... <end_of_turn>
    턴 기반 chat template을 -it 토크나이저에 내장하고 있다. 출력 role은 'assistant'가 아니라 'model'.
  - 🔴 수동으로 <start_of_turn> 마커를 문자열 조립하지 말 것.
    tokenizer.apply_chat_template()에 위임한다 (TRL SFTTrainer는 conversational
    'messages' 데이터셋을 받으면 자동으로 chat template을 적용한다).
  - 초기 Gemma instruct는 전용 system role이 없어, system 프롬프트를 첫 user 턴에 병합하는
    템플릿이 많다. 정확한 처리는 각 모델의 tokenizer_config에 따르므로 apply_chat_template에 맡긴다.

따라서 이 모듈은 "마커를 만든다"가 아니라 각 트랙의 raw row를 **표준 messages 포맷**
( [{"role": "system"/"user"/"assistant", "content": "..."}] )으로 바꾸는 어댑터를 제공한다.
학습 데이터는 이 messages 리스트를 담은 컬럼("messages")으로 저장하면 SFTTrainer가 알아서 처리.
"""
from __future__ import annotations

from typing import Any


def build_messages(
    user_content: str,
    assistant_content: str,
    system_content: str | None = None,
) -> list[dict[str, str]]:
    """단일 turn (system?)+user+assistant → 표준 messages 리스트.

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
    """추론용 messages (assistant 턴 없음). endpoint 호출 프롬프트 조립에 사용."""
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
