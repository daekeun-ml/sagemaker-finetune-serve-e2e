"""
Bedrock 합성 없이 dry-run용 소량 학습 JSONL을 생성합니다.

GPU dry-run은 '파이프라인이 도는가'만 검증하므로 합성 데이터가 필요 없다. 시드 데이터셋에서
few-shot 만 뽑아 messages JSONL 을 만든다. (ungated 시드/모델이면 HF 토큰 불필요.)

사용:
    python tests/prepare_dryrun_data.py --track extraction --n 24 --out /tmp/dryrun_train.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRACK_DIRS = {
    "extraction": "01_extraction_to_json",
    "classification": "02_classification",
    "summarization": "03_summarization",
    "domain_qa": "04_domain_qa",
}


def load_track_data(track: str):
    d = TRACK_DIRS[track]
    path = os.path.join(ROOT, "tracks", d, "track_data.py")
    spec = importlib.util.spec_from_file_location(f"track_data_{track}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="extraction", choices=list(TRACK_DIRS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="/tmp/dryrun_train.jsonl")
    args = ap.parse_args()

    td = load_track_data(args.track)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[{args.track}] loading seed dataset...")
    seeds = td.load_seed_examples(args.n, token=token)
    print(f"Parsed seeds: {len(seeds)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for s in seeds:
            msgs = td.to_messages(s)
            # Gemma 입력에는 system 역할을 남기지 않습니다.
            assert "system" not in [m["role"] for m in msgs], "Gemma does not support the system role"
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(seeds)} examples -> {args.out}")


if __name__ == "__main__":
    main()
