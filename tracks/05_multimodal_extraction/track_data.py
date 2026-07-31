"""
tracks/05_multimodal_extraction/track_data.py — 멀티모달 추출(이미지→JSON) 트랙 데이터 어댑터

시드: naver-clova-ix/cord-v2 (cc-by-4.0, ungated) — 영수증 이미지 + 구조화 ground_truth(JSON).
성공 기준: 영수증 이미지에서 메뉴/수량/가격을 구조화 JSON으로 정확히 추출.
🔴 텍스트 트랙과 달리 입력에 '이미지'가 들어가므로, 메시지 content가 [{type:image}, {type:text}] 리스트다.

이 트랙은 합성 데이터를 만들지 않는다(이미지 합성은 별개 문제) — 시드를 직접 학습/평가에 쓴다.
"""
from __future__ import annotations

import json
import os
from typing import Any

SEED_DATASET = "naver-clova-ix/cord-v2"

# 이미지와 함께 주는 지시문(첫 user 턴 텍스트). Gemma는 system role 거부 → user 텍스트에 fold.
INSTRUCTION = (
    "You are a receipt-parsing engine. Extract the receipt into strict JSON with a 'menu' array of "
    "{name, count, price} items. Output ONLY valid JSON, no prose."
)


def _simplify_gt(ground_truth: str) -> str:
    """cord-v2 ground_truth(문자열 JSON) → 간결한 학습 타깃 JSON 문자열.

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
    """cord-v2 row → TRL VLM 학습 포맷 {"images":[PIL], "messages":[텍스트만]}.

    🔴 TRL VLM collator는 이미지를 별도 `images` 컬럼으로 받고, messages는 텍스트만 둔다
       (collator가 이미지 placeholder를 주입). messages content에 이미지를 직접 넣으면
       'images ≠ placeholders' 에러가 난다(실측 확인).
    """
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": _simplify_gt(example.get("ground_truth", ""))},
        ],
    }


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def load_sample_receipts() -> list[dict]:
    """`samples/`에 미리 저장해 둔 영수증 2장을 즉시 로드(데이터셋 다운로드 없음).

    반환: [{"name","image"(PIL),"ground_truth"(dict),"menu_items"(int)}, ...]

    🔴 왜 있는가 (실측 2026-07-31): cord-v2는 이미지가 parquet에 내장돼 캐시가 없으면
       1건 로드에 ~40초가 걸린다(추론 데모엔 과한 비용). 배포 검증은 이미지 1~2장이면
       충분하므로 리포에 넣어 두고 즉시 쓴다. cord-v2는 cc-by-4.0이라 재배포 가능하다
       (출처: naver-clova-ix/cord-v2, samples/ground_truth.json에 원본 인덱스 기록).
       학습·평가처럼 전량이 필요할 때는 load_seed_examples()를 쓴다.
    """
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
    """cord-v2 시드 n개를 {"images","messages"} 형태로 반환(학습·평가용 — 데이터셋 다운로드).

    노트북 미리보기/추론용으로는 원본 PIL 이미지도 필요하니 '_image' 키에 함께 담아 준다.
    배포 스모크처럼 1~2장만 필요하면 load_sample_receipts()가 훨씬 빠르다.

    🔴 streaming=True 를 쓰지 않는 이유 (실측 2026-07-31): 이 데이터셋은 이미지가 parquet에
       내장돼 있어 **첫 row 하나를 꺼내는 데만 23초**가 걸리고, streaming은 로컬 디스크에
       캐시하지 않아 **셀을 다시 실행할 때마다 그 비용을 또 낸다**(재호출 24초 실측).
       split 슬라이스("train[:n]")로 받으면 첫 회는 비슷하지만(~36초, 900건 전량 준비)
       이후 캐시 히트로 **0.15초**가 된다. 노트북은 같은 셀을 여러 번 돌리므로 이쪽이 낫다.
    """
    from datasets import load_dataset
    ds = load_dataset(SEED_DATASET, split=f"train[{offset}:{offset + n}]", token=token)
    out = []
    for row in ds:
        ex = to_example(row)
        ex["_image"] = row["image"]   # 미리보기/추론 편의용(학습 컬럼 아님)
        out.append(ex)
    return out
