"""
tracks/04_domain_qa/track_data.py — 도메인 QA / instruction 트랙 데이터 어댑터

시드: databricks/databricks-dolly-15k (cc-by-sa-3.0, ungated) — instruction+context+response+category.
성공 기준: (context 있으면 근거해) instruction에 정확·유용하게 응답.
⚠️ CC-BY-SA: 파생물 share-alike 의무 — 배포 시 라이선스 전파.
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
    # 🔴 Gemma는 system role 거부 → system 지시문을 첫 user 턴에 병합(fold).
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
