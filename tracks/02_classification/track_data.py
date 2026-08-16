"""의도 분류 코스의 시드 로더와 messages 변환기를 제공합니다.

스크립트 기반 원본 대신 parquet 형식의 banking77 미러를 사용하며,
클래스 편향을 막기 위해 고정 시드로 셔플합니다.
"""
from __future__ import annotations

TASK_INSTRUCTION = (
    "Text classification (banking customer-intent). Given a customer message, output the single "
    "most appropriate intent label from the banking domain (e.g. card_arrival, balance_not_updated, "
    "lost_or_stolen_card). Labels = one intent string per message."
)

SYSTEM_PROMPT = (
    "You are an intent classifier for banking customer messages. Output ONLY the single most "
    "appropriate intent label (snake_case), nothing else."
)

# datasets 5 이상에서 읽을 수 있는 parquet 미러를 순서대로 시도합니다.
_CANDIDATES = (
    "mteb/banking77",          # mit, parquet, label_text 컬럼 있음 (기본)
    "gtfintechlab/banking77",  # cc-by-4.0, parquet, label이 ClassLabel (폴백)
)

# banking77 라벨은 데이터셋 features에서 로드 (하드코딩 대신 런타임 확인)
_LABEL_NAMES: list[str] = []


def _load_label_names(ds) -> list[str]:
    """ClassLabel의 라벨 이름을 반환하고 없으면 빈 리스트를 반환합니다."""
    global _LABEL_NAMES
    try:
        _LABEL_NAMES = list(ds.features["label"].names)
    except Exception:
        _LABEL_NAMES = []
    return _LABEL_NAMES


def _label_str(row: dict, labels: list[str]) -> str:
    """지원하는 데이터셋 스키마에서 의도 라벨을 추출합니다.

    - mteb/banking77       : label(int) + label_text(str)  → label_text 우선 사용
    - gtfintechlab/banking77: label(ClassLabel int)        → features.names 로 변환
    """
    txt = row.get("label_text")
    if isinstance(txt, str) and txt:
        return txt
    lbl = row.get("label")
    if labels and isinstance(lbl, int) and 0 <= lbl < len(labels):
        return labels[lbl]
    return str(lbl)


def to_messages(example: dict[str, str]) -> list[dict[str, str]]:
    # system 지시문을 첫 user 턴에 병합합니다.
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def load_seed_examples(n: int, token: str | None = None, *, seed: int = 42) -> list[dict[str, str]]:
    """banking77 시드를 고정된 순서로 셔플해 n건 반환합니다."""
    from datasets import load_dataset

    ds, errs = None, []
    for name in _CANDIDATES:
        try:
            ds = load_dataset(name, split="train", token=token)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__} {str(e)[:120]}")
    if ds is None:
        raise RuntimeError(
            "banking77 시드를 로드하지 못했습니다. 시도한 후보:\n  " + "\n  ".join(errs)
            + "\n  PolyAI/banking77은 스크립트 기반이라 datasets 5 이상에서 지원되지 않습니다."
        )

    labels = _load_label_names(ds)
    ds = ds.shuffle(seed=seed)          # 라벨 정렬 순서를 섞습니다.
    out: list[dict[str, str]] = []
    for row in ds:
        out.append({"input": row["text"], "output": _label_str(row, labels)})
        if len(out) >= n:
            break
    return out


def load_label_names(token: str | None = None) -> list[str]:
    """평가에 사용할 banking77 라벨 이름을 반환합니다."""
    from datasets import load_dataset

    for name in _CANDIDATES:
        try:
            ds = load_dataset(name, split="test", token=token)
        except Exception:  # noqa: BLE001
            continue
        try:                                    # ClassLabel 이면 그대로
            return list(ds.features["label"].names)
        except Exception:                       # noqa: BLE001  int64 + label_text
            pass
        if "label_text" in ds.column_names:
            # 정수 라벨과 label_text를 매핑해 인덱스 순서로 정렬합니다.
            m: dict[int, str] = {}
            for row in ds:
                lbl = row.get("label")
                if isinstance(lbl, int) and lbl not in m:
                    m[lbl] = str(row.get("label_text"))
            if m:
                return [m[i] for i in sorted(m)]
    raise RuntimeError(
        "banking77 라벨 목록을 얻지 못했습니다. 시도한 후보: " + ", ".join(_CANDIDATES))


def seed_texts_for_synth(seed_examples: list[dict[str, str]]) -> list[str]:
    return [f"MESSAGE: {e['input']}\nINTENT: {e['output']}" for e in seed_examples]
