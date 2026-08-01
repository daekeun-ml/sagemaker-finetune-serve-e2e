"""
tracks/02_classification/track_data.py — 텍스트 분류(intent) 트랙 데이터 어댑터

시드: mteb/banking77 (mit, ungated) — text + label(77 banking intent) + label_text.
JumpStart엔 분류 전용 헤드가 없으므로 **instruction-based 텍스트생성**(라벨을 텍스트로) 방식으로 학습.
성공 기준: 주어진 문의를 올바른 intent 라벨로 분류 (accuracy/macro-F1).

🔴 데이터셋 선택 (실측 2026-07-30):
  원본 `PolyAI/banking77`은 **스크립트 기반**(banking77.py) 리포라서 이 kit이 핀한 datasets>=5.0.0에서
  `RuntimeError: Dataset scripts are no longer supported, but found banking77.py`로 **로드 자체가 실패**한다.
  parquet 자동변환본(refs/convert/parquet)도 없어(HF dataset-viewer도 같은 에러) 되살릴 방법이 없다.
  → **`mteb/banking77`**(mit, parquet, 다운로드 15k, MTEB 조직 관리)을 기본으로 쓴다.
  ⚠️ `legacy-datasets/banking77`은 내용·라이선스가 원본과 같지만 카드에 "deprecated and will be
     deleted"가 명시돼 있어 쓰지 않는다. `gtfintechlab/banking77`(cc-by-4.0)은 동작하지만
     다운로드 93건뿐이라 폴백으로만 둔다.
  아래 _CANDIDATES 를 순서대로 시도하므로, 어느 하나가 사라져도 다음 것으로 넘어간다.

🔴 셔플 필수 (실측):
  banking77의 train 스플릿은 **라벨 정렬 순서**다(라벨이 단조증가). 앞에서부터 N건을 그냥 뽑으면
  300건에 클래스가 3개뿐이고(card_arrival 153 / card_linking 139 / exchange_rate 8),
  평가용 held-out(행 100~149)은 **전부 card_arrival 단일 라벨**이 된다.
  → 77클래스 macro-F1이 무의미해지므로, 고정 시드로 셔플해 클래스가 골고루 섞이게 한다.
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

# 🔴 시드 후보 (순서대로 시도). 원본 PolyAI/banking77 은 datasets>=5.0.0에서 로드 불가 → 제외.
_CANDIDATES = (
    "mteb/banking77",          # mit, parquet, label_text 컬럼 있음 (기본)
    "gtfintechlab/banking77",  # cc-by-4.0, parquet, label이 ClassLabel (폴백)
)

# banking77 라벨은 데이터셋 features에서 로드 (하드코딩 대신 런타임 확인)
_LABEL_NAMES: list[str] = []


def _load_label_names(ds) -> list[str]:
    """label이 ClassLabel이면 그 names를, 아니면 빈 리스트(→ label_text 컬럼 사용)."""
    global _LABEL_NAMES
    try:
        _LABEL_NAMES = list(ds.features["label"].names)
    except Exception:
        _LABEL_NAMES = []
    return _LABEL_NAMES


def _label_str(row: dict, labels: list[str]) -> str:
    """row → intent 라벨 문자열. 두 데이터셋 스키마를 모두 지원한다.

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
    # 🔴 Gemma는 system role 거부 → system 지시문을 첫 user 턴에 병합(fold).
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def load_seed_examples(n: int, token: str | None = None, *, seed: int = 42) -> list[dict[str, str]]:
    """banking77 시드 n건. 🔴 **고정 시드로 셔플**해 클래스가 골고루 섞이게 한다.

    셔플이 없으면 train이 라벨 정렬 순서라 앞 N건에 클래스가 몇 개뿐이고(실측: 300건에 3클래스),
    평가 held-out이 단일 라벨로 붕괴해 macro-F1이 무의미해진다.
    seed를 고정하므로 학습/평가에서 같은 인덱스를 부르면 같은 결과가 나온다(재현성).
    """
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
            + "\n  ⚠️ 원본 PolyAI/banking77 은 스크립트 기반이라 datasets>=5.0.0에서 지원되지 않습니다."
        )

    labels = _load_label_names(ds)
    ds = ds.shuffle(seed=seed)          # 🔴 라벨 정렬 순서 해소
    out: list[dict[str, str]] = []
    for row in ds:
        out.append({"input": row["text"], "output": _label_str(row, labels)})
        if len(out) >= n:
            break
    return out


def load_label_names(token: str | None = None) -> list[str]:
    """77개 intent 라벨 이름 전체. 04_evaluate 의 macro-F1 계산에 필요하다.

    🔴 04_evaluate 가 `load_dataset('PolyAI/banking77', split='test').features['label'].names` 로
       직접 조회하면 같은 스크립트-데이터셋 문제로 죽는다. 이 함수를 쓰면 로더가 한 곳에 모이고
       미러가 바뀌어도 여기만 고치면 된다.

    mteb/banking77 은 label 이 ClassLabel 이 아니라 int64 + label_text 이므로,
    ClassLabel 이 없으면 label_text 를 (label 정수 순서대로) 모아 이름 목록을 만든다.
    """
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
            # label(정수) → label_text 매핑을 만들어 인덱스 순서대로 정렬
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
