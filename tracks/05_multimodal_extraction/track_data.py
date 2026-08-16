"""멀티모달 추출 코스의 시드 로더와 VLM 학습 예시 변환기를 제공합니다."""
from __future__ import annotations

import json
import os
from typing import Any

SEED_DATASET = "naver-clova-ix/cord-v2"

# 이미지와 함께 전달할 user 지시문입니다.
INSTRUCTION = (
    "You are a receipt-parsing engine. Extract the receipt into strict JSON with a 'menu' array of "
    "{name, count, price} items. Output ONLY valid JSON, no prose."
)


def _simplify_gt(ground_truth: str) -> str:
    """cord-v2 정답을 간결한 학습 대상 JSON으로 변환합니다.

    원본: {"gt_parse": {"menu": [{"nm","cnt","price"}, ...], "sub_total":..., "total":...}}
    타깃: {"menu": [{"name","count","price"}, ...]} 로 정규화(핵심 필드만, 학습 안정).
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return ground_truth
    parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else {}
    menu = parse.get("menu", []) if isinstance(parse, dict) else []
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for m in menu:
        if not isinstance(m, dict):
            continue
        items.append({"name": m.get("nm", ""), "count": m.get("cnt", ""), "price": m.get("price", "")})
    return json.dumps({"menu": items}, ensure_ascii=False)


def to_example(example: dict[str, Any]) -> dict:
    """cord-v2 행을 TRL VLM 학습 형식으로 변환합니다."""
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": _simplify_gt(example.get("ground_truth", ""))},
        ],
    }


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def load_sample_receipts() -> list[dict]:
    """데이터셋 다운로드 없이 저장소의 영수증 샘플을 로드합니다."""
    from PIL import Image
    meta_path = os.path.join(SAMPLES_DIR, "ground_truth.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    out = []
    for m in meta:
        img = Image.open(os.path.join(SAMPLES_DIR, m["file"]))
        img.load()                                   # 파일 핸들 즉시 해제
        out.append({"name": m["file"], "image": img, "ground_truth": m["ground_truth"],
                    "menu_items": m["menu_items"]})
    return out


def load_seed_examples(n: int, token: str | None = None, offset: int = 0) -> list[dict]:
    """cord-v2 시드 n개를 학습과 평가 형식으로 반환합니다.

    반복 실행 시 캐시를 재사용할 수 있도록 streaming 대신 split 슬라이스를 사용합니다.
    """
    from datasets import load_dataset
    ds = load_dataset(SEED_DATASET, split=f"train[{offset}:{offset + n}]", token=token)
    out = []
    for row in ds:
        ex = to_example(row)
        ex["_image"] = row["image"]   # 미리보기/추론 편의용(학습 컬럼 아님)
        out.append(ex)
    return out
