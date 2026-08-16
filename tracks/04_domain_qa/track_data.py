"""도메인 QA 코스의 시드 로더와 messages 변환기를 제공합니다.

시드 데이터는 CC BY-SA 3.0이므로 파생물 배포 시 라이선스 조건을 확인해야 합니다.
"""
from __future__ import annotations

TASK_INSTRUCTION = (
    "Domain question-answering / instruction following. Given an instruction and optional context, "
    "produce a helpful, correct answer grounded in the context when provided. Output = the answer text."
)

SYSTEM_PROMPT = (
    "You are a helpful domain assistant. Answer the user's instruction. If context is provided, "
    "ground your answer in it and do not contradict it."
)


def to_messages(example: dict[str, str]) -> list[dict[str, str]]:
    # system 지시문을 첫 user 턴에 병합합니다.
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def _compose_input(instruction: str, context: str) -> str:
    context = (context or "").strip()
    if context:
        return f"{instruction}\n\n[Context]\n{context}"
    return instruction


def load_seed_examples(n: int, token: str | None = None) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", token=token)
    out: list[dict[str, str]] = []
    for row in ds:
        instr = row.get("instruction") or ""
        resp = row.get("response") or ""
        if instr and resp:
            out.append({"input": _compose_input(instr, row.get("context", "")), "output": resp})
        if len(out) >= n:
            break
    return out


def seed_texts_for_synth(seed_examples: list[dict[str, str]]) -> list[str]:
    return [f"INSTRUCTION: {e['input'][:500]}\nANSWER: {e['output'][:400]}" for e in seed_examples]
