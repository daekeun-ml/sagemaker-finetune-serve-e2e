"""정보 추출 코스의 시드 로더와 messages 변환기를 제공합니다."""
from __future__ import annotations

import json
import re
from typing import Any

# 이 트랙의 task 지시문 (합성 생성/critique 프롬프트에 주입)
TASK_INSTRUCTION = (
    "Structured information extraction: given a user request and an available function/tool "
    "schema, extract the correct function call as strict JSON (function name + typed arguments). "
    "The domain is API/tool function-calling. Labels = valid JSON tool calls grounded in the schema."
)

# SLM에 학습시킬 system 프롬프트 (추론 시에도 동일하게 사용)
SYSTEM_PROMPT = (
    "You are a precise information-extraction engine. Read the user text and the available "
    "tool schema, then output ONLY a valid JSON object of the form "
    '{"name": "<function>", "arguments": {<args>}}. No prose, JSON only.'
)


def _parse_glaive_row(row: dict[str, Any]) -> dict[str, str] | None:
    """glaive-function-calling-v2 행을 입력과 출력으로 변환합니다.

    스키마: system(툴 JSON 스키마 포함), chat(멀티턴 텍스트; assistant가 <functioncall> JSON 방출).
    USER: ... ASSISTANT: <functioncall> {json} 형태를 파싱.
    """
    system = row.get("system", "") or ""
    chat = row.get("chat", "") or ""

    # 첫 USER 발화
    um = re.search(r"USER:\s*(.*?)(?:\nASSISTANT:|\Z)", chat, re.DOTALL)
    # 첫 functioncall
    fm = re.search(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|ASSISTANT:|USER:|\Z)", chat, re.DOTALL)
    if not um or not fm:
        return None
    user_text = um.group(1).strip()
    try:
        call = json.loads(fm.group(1).strip())
    except json.JSONDecodeError:
        return None
    # 스키마 힌트를 user 입력에 포함 (grounded)
    schema_hint = system.replace("SYSTEM:", "").strip()
    user_input = f"{user_text}\n\n[Available tools]\n{schema_hint}" if schema_hint else user_text
    return {"input": user_input, "output": json.dumps(call, ensure_ascii=False)}


def to_messages(example: dict[str, str]) -> list[dict[str, str]]:
    """system 지시문을 첫 user 턴에 병합해 표준 messages를 만듭니다."""
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def load_seed_examples(n: int, token: str | None = None) -> list[dict[str, str]]:
    """시드 스모크 샘플 n개를 {"input","output"} 리스트로 반환."""
    from datasets import load_dataset

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", token=token)
    out: list[dict[str, str]] = []
    for row in ds:
        parsed = _parse_glaive_row(row)
        if parsed:
            out.append(parsed)
        if len(out) >= n:
            break
    return out


def seed_texts_for_synth(seed_examples: list[dict[str, str]]) -> list[str]:
    """합성 grounding용 seed 문자열 (input+output 요약)."""
    return [f"REQUEST: {e['input'][:400]}\nEXTRACTED_JSON: {e['output']}" for e in seed_examples]
