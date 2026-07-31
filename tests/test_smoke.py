"""
tests/test_smoke.py — 무거운 의존성 없이 순수 로직만 빠르게 검증하는 스모크 테스트.

목적: 개발환경 GPU에서 실제 학습/AWS 호출 전에, 데이터 어댑터·포맷터·합성 파서 같은
      순수 파이썬 로직이 정상인지 수 초 내 확인. (boto3/torch/transformers 불필요)

실행:
    cd ~/sagemaker-finetune-serve-e2e
    python -m pytest tests/test_smoke.py -v
    # 또는 pytest 없이:
    python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import sys

# 리포 루트를 path에 추가 (common import)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gemma_format_messages():
    from common import gemma_format as gf

    msgs = gf.build_messages("입력", "출력", system_content="시스템")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == "출력"

    inf = gf.build_inference_messages("입력", system_content="시스템")
    assert [m["role"] for m in inf] == ["system", "user"]


def test_fold_system_into_user():
    from common import gemma_format as gf

    folded = gf.fold_system_into_user(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert folded[0]["role"] == "user"
    assert folded[0]["content"].startswith("S")


def test_pii_and_dedup():
    from common.synth import bedrock_synth as bs

    assert bs.has_pii("연락처 test@example.com 입니다")
    assert not bs.has_pii("그냥 평범한 문장입니다")
    # 회귀(Finding 4): function-call JSON의 순수 긴 숫자는 PII 오탐 금지
    assert not bs.has_pii('{"name":"book","arguments":{"ts":1721400000,"amount":123456789}}')
    assert not bs.has_pii('{"id": 987654321012345}')
    # 진짜 전화/카드(구분자 포함)는 여전히 탐지
    assert bs.has_pii("call me at +1 415-555-0198")
    assert bs.has_pii("card 4111 1111 1111 1111")
    a = {"input": "Hello World", "output": "x"}
    b = {"input": "hello   world", "output": "x"}
    assert bs._dedup_key(a) == bs._dedup_key(b)  # 공백/대소문자 정규화


def test_extract_json_variants():
    from common.synth import bedrock_synth as bs

    assert bs._extract_json('[{"a":1}]') == [{"a": 1}]
    # 코드펜스 + 앞뒤 prose 방어
    assert bs._extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert bs._extract_json('설명... [{"x": 3}] 끝') == [{"x": 3}]


def test_extraction_track_adapter():
    """glaive row 파서 + to_messages 검증 (네트워크 없이 합성 row로)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tracks", "01_extraction_to_json",
    ))
    import track_data as td

    fake_row = {
        "system": "SYSTEM: You have access to: get_weather(city)",
        "chat": (
            "USER: What's the weather in Seoul? "
            "ASSISTANT: <functioncall> {\"name\": \"get_weather\", "
            "\"arguments\": {\"city\": \"Seoul\"}} <|endoftext|>"
        ),
    }
    parsed = td._parse_glaive_row(fake_row)
    assert parsed is not None
    assert "Seoul" in parsed["input"]
    call = json.loads(parsed["output"])
    assert call["name"] == "get_weather"

    # to_messages 는 system-folded 2턴 (Gemma 계약)
    msgs = td.to_messages(parsed)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[-1]["role"] == "assistant"


def test_all_track_adapters_importable():
    """4개 트랙의 track_data.py 가 모두 import되고 필수 심볼/시그니처를 갖는지."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracks = {
        "extraction": "01_extraction_to_json",
        "classification": "02_classification",
        "summarization": "03_summarization",
        "domain_qa": "04_domain_qa",
    }
    for key, d in tracks.items():
        path = os.path.join(root, "tracks", d, "track_data.py")
        spec = importlib.util.spec_from_file_location(f"track_data_{key}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 필수 심볼
        assert hasattr(mod, "TASK_INSTRUCTION") and mod.TASK_INSTRUCTION
        assert hasattr(mod, "SYSTEM_PROMPT") and mod.SYSTEM_PROMPT
        assert callable(mod.to_messages)
        assert callable(mod.load_seed_examples)
        assert callable(mod.seed_texts_for_synth)
        # to_messages 계약: 🔴 Gemma는 system role 거부 → system 없이 user+assistant,
        # system 지시문은 첫 user 턴에 folded 되어야 한다.
        msgs = mod.to_messages({"input": "in", "output": "out"})
        roles = [m["role"] for m in msgs]
        assert "system" not in roles, f"{key}: Gemma는 system role 미지원 — fold 필요"
        assert roles == ["user", "assistant"], f"{key}: {roles}"
        assert msgs[-1]["content"] == "out"
        # system 지시문이 user 턴에 병합됐는지
        assert mod.SYSTEM_PROMPT.split("\n")[0][:20] in msgs[0]["content"]


def test_multimodal_track_adapter():
    """멀티모달 트랙(05_multimodal_extraction)의 track_data.py 계약 검사.

    텍스트 트랙과 인터페이스가 다르다: to_example → {"images","messages"} (TRL VLM 포맷).
    """
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "tracks", "05_multimodal_extraction", "track_data.py")
    spec = importlib.util.spec_from_file_location("track_data_mm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "INSTRUCTION") and mod.INSTRUCTION
    assert callable(mod.to_example)
    assert callable(mod.load_seed_examples)
    # _simplify_gt: cord-v2 ground_truth → {menu:[{name,count,price}]}
    out = mod._simplify_gt('{"gt_parse": {"menu": [{"nm":"Coffee","cnt":"2 x","price":"5,000"}]}}')
    assert '"menu"' in out and "Coffee" in out
    # to_example 계약: images 컬럼(리스트) + messages(텍스트만, user+assistant)
    class _Img:  # PIL 대체(계약만 검사)
        size = (10, 10)
    ex = mod.to_example({"image": _Img(), "ground_truth": '{"gt_parse":{"menu":[]}}'})
    assert set(ex.keys()) >= {"images", "messages"}
    assert isinstance(ex["images"], list) and len(ex["images"]) == 1
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["user", "assistant"], f"mm roles: {roles}"
    # 🔴 messages content는 텍스트(문자열)여야 한다 — 이미지는 별도 images 컬럼(TRL VLM collator 계약)
    assert isinstance(ex["messages"][0]["content"], str)


def test_multimodal_track_registered():
    """config.TRACKS 에 멀티모달 트랙이 multimodal=True 로 등록됐는지."""
    from common import config
    assert "mm_extraction" in config.TRACKS
    assert config.TRACKS["mm_extraction"].multimodal is True


def test_config_dry_run_switch():
    from common import config

    os.environ["DRY_RUN"] = "1"
    assert config.is_dry_run() is True
    os.environ["DRY_RUN"] = "0"
    assert config.is_dry_run() is False
    # 트랙 레지스트리 무결성 (텍스트 4 + 멀티모달 1)
    assert set(config.TRACKS) == {"extraction", "classification", "summarization", "domain_qa", "mm_extraction"}


def _run_all():
    """pytest 없이 직접 실행 지원."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
