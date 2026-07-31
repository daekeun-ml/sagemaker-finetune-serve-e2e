"""
tracks/03_summarization/track_data.py — 문서 요약 트랙 데이터 어댑터

시드: FiscalNote/billsum (cc0-1.0, ungated, public domain) — text(법안)+summary.
성공 기준: 문서의 핵심을 정확·간결하게 요약 (ROUGE + LLM-judge).

참고: 대화체 요약은 완전 permissive 공개셋이 부족 → 문서요약 시드 + grounded 합성으로 확장.
"""
from __future__ import annotations

TASK_INSTRUCTION = (
    "Document summarization. Given a document (e.g. a legislative bill or report), produce a concise, "
    "faithful summary that captures the key points without adding facts. Output = the summary text."
)

SYSTEM_PROMPT = (
    "You are a precise document summarizer. Read the document and produce a concise, faithful summary. "
    "Do not add information not present in the document. Output ONLY the summary."
)

MAX_DOC_CHARS = 6000  # 시드 문서가 매우 길 수 있어 컨텍스트 절단


def to_messages(example: dict[str, str]) -> list[dict[str, str]]:
    # 🔴 Gemma는 system role 거부 → system 지시문을 첫 user 턴에 병합(fold).
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nSummarize the following document:\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def load_seed_examples(n: int, token: str | None = None) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("FiscalNote/billsum", split="train", token=token)
    out: list[dict[str, str]] = []
    for row in ds:
        doc = (row.get("text") or "")[:MAX_DOC_CHARS]
        summ = row.get("summary") or ""
        if doc and summ:
            out.append({"input": doc, "output": summ})
        if len(out) >= n:
            break
    return out


def seed_texts_for_synth(seed_examples: list[dict[str, str]]) -> list[str]:
    # 요약 스타일/길이를 grounding (문서 앞부분 + 요약)
    return [f"DOCUMENT(excerpt): {e['input'][:600]}\nSUMMARY: {e['output']}" for e in seed_examples]
