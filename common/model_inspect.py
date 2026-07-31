"""
common/model_inspect.py — 로컬 모델 디렉토리 준비 + 서빙 가능 여부 점검

02b_local_serve 노트북이 쓰는 두 가지 일을 담당한다(노트북 셀을 짧게 유지하기 위해 분리):
  1) prepare_local_model()  — 검증할 로컬 모델 디렉토리를 확보(로컬 산출물 or S3 아티팩트 해제)
  2) inspect_servability()  — 이 체크포인트가 vLLM으로 뜰 수 있는지 '실제 텐서 키'로 판정

🔴 왜 config만 보고 판단하면 안 되는가 (실측 2026-07-30):
   gemma-4 E2B/E4B는 num_kv_shared_layers>0이고, transformers는 KV-shared 레이어에
   k_norm/k_proj/v_proj 모듈을 아예 만들지 않는다("Layers sharing kv states don't need any
   weight matrices"). 그래서 save_pretrained를 거치면 원본에 있던 그 텐서가 소실된다(E4B 54개).
   vLLM은 k_norm을 전 레이어에 등록하므로 없으면 'weights not initialized' ValueError로
   엔진 초기화가 실패한다(vLLM issue #44788).
   → config의 num_kv_shared_layers 값만으로는 "복원됐는지"를 알 수 없다. 그래서 safetensors
     헤더에서 키를 직접 읽어 확인한다(가중치는 읽지 않으므로 빠르다).
"""
from __future__ import annotations

import glob
import json
import os
import struct


# ---------------------------------------------------------------------------
# 1) 체크포인트 키 읽기 (헤더만 — 가중치 로드 없음)
# ---------------------------------------------------------------------------
def checkpoint_keys(model_dir: str) -> set[str]:
    """모델 디렉토리의 텐서 키 집합. 샤딩(index.json)과 단일 파일 모두 지원.

    safetensors 헤더는 앞 8바이트가 헤더 길이(uint64 LE), 이어서 그 길이만큼 JSON이다.
    수십 GB 파일이라도 헤더만 읽으므로 즉시 끝난다.
    """
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            return set(json.load(f)["weight_map"].keys())
    keys: set[str] = set()
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys |= {k for k in json.loads(fh.read(n).decode()) if k != "__metadata__"}
    return keys


def _cfg_int(cfg: dict, *names: str) -> int:
    """config 또는 config.text_config에서 정수 필드를 찾아 반환(멀티모달 config 대응)."""
    for src in (cfg, cfg.get("text_config") or {}):
        for n in names:
            if isinstance(src.get(n), int):
                return src[n]
    return 0


# ---------------------------------------------------------------------------
# 2) 서빙 가능 여부 판정
# ---------------------------------------------------------------------------
def inspect_servability(model_dir: str, *, verbose: bool = True) -> dict:
    """체크포인트를 열어 vLLM 서빙 가능 여부를 판정한다.

    Returns dict:
      arch, model_type, is_text_only, kv_shared, n_layers, vllm_ok, missing(list), engine

    engine: 'vllm'(뜬다) | 'transformers'(KV-shared 텐서 누락 → vLLM 거부)
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"{model_dir}에 config.json이 없습니다(tar 구조 확인). 서빙 가능한 모델 디렉토리를 지정하세요.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # 🔴 텍스트 서빙이면 model_type이 *_text (예 gemma4_text)라 vision/audio 없이 로드된다.
    model_type = str(cfg.get("model_type", ""))
    is_text_only = model_type.endswith("_text") or "vision_config" not in cfg

    kv_shared = _cfg_int(cfg, "num_kv_shared_layers", "kv_shared_layers")
    n_layers = _cfg_int(cfg, "num_hidden_layers")

    missing: list[str] = []
    if kv_shared > 0 and n_layers > 0:
        keys = checkpoint_keys(model_dir)
        first = n_layers - kv_shared          # E4B: 42-18=24 → 레이어 24~41이 shared
        missing = [k for k in (
            f"model.layers.{i}.self_attn.{n}.weight"
            for i in range(first, n_layers) for n in ("k_norm", "k_proj", "v_proj")
        ) if k not in keys]

    vllm_ok = not missing
    info = {
        "arch": cfg.get("architectures"),
        "model_type": model_type,
        "is_text_only": is_text_only,
        "kv_shared": kv_shared,
        "n_layers": n_layers,
        "vllm_ok": vllm_ok,
        "missing": missing,
        "engine": "vllm" if vllm_ok else "transformers",
    }
    if verbose:
        print_servability(info, model_dir)
    return info


def print_servability(info: dict, model_dir: str = "") -> None:
    """inspect_servability 결과를 사람이 읽는 형태로 출력(조치까지 안내)."""
    if model_dir:
        print("MODEL_DIR :", model_dir)
    print("arch      :", info["arch"])
    print("model_type:", info["model_type"])
    print("text-only servable:", info["is_text_only"])
    print(f"num_kv_shared_layers: {info['kv_shared']} / {info['n_layers']} layers"
          f"  →  vLLM 서빙: " + ("가능 ✅" if info["vllm_ok"] else "불가 🔴"))

    if info["missing"]:
        print(f"🔴 KV-shared 텐서 {len(info['missing'])}개 누락 → 이 체크포인트는 vLLM이 거부합니다(#44788).")
        print(f"   예: {info['missing'][:2]}")
        print("   해결: 최신 train.py로 다시 학습/재-export → 저장 시 자동 복원됩니다.")
        print("         (train.py의 _revive_kv_shared_from_base가 base에서 그 텐서를 되살립니다)")
        print("   🔴 이미 재학습했는데 이 메시지가 보이면, 로컬 캐시가 옛 아티팩트일 수 있습니다")
        print("      → prepare_local_model(force=True) 로 다시 내려받으세요.")
    elif info["kv_shared"] > 0:
        print("KV-shared 텐서가 체크포인트에 있습니다 → vLLM으로 서빙됩니다.")
    else:
        print("KV-sharing 없음(12B/26B/31B) → vLLM으로 서빙됩니다.")


# ---------------------------------------------------------------------------
# 3) 로컬 모델 디렉토리 준비
# ---------------------------------------------------------------------------
_STAMP = ".source_model_data"   # 이 디렉토리가 어느 S3 아티팩트에서 나왔는지 기록


def prepare_local_model(model_data: str | None, region: str, *,
                        local_out: str = "out", cache_dir: str = "local_model",
                        force: bool = False) -> str:
    """검증에 쓸 로컬 모델 디렉토리를 확보해 경로를 반환한다.

    우선순위:
      (A) local_out('out')에 config.json이 있으면 그것을 사용(로컬 dry-run 산출물).
      (B) 없으면 model_data(S3 아티팩트)를 cache_dir에 내려받아 압축 해제.

    🔴 캐시 무효화: cache_dir 안에 어떤 아티팩트를 풀었는지 `.source_model_data`로 기록하고,
       model_data가 달라지면 **자동으로 다시 내려받는다**. 이게 없으면 재학습 후에도 옛 체크포인트를
       계속 검증하게 되어(실측) "왜 아직 KV-shared 텐서가 없지?" 하는 혼란이 생긴다.
       force=True면 기록과 무관하게 강제로 다시 받는다.
    """
    import shutil
    import tarfile

    local_out = os.path.abspath(local_out)
    if os.path.isfile(os.path.join(local_out, "config.json")):
        print("로컬 dry-run 산출물 사용:", local_out)
        return local_out

    if not (model_data and str(model_data).startswith("s3://")):
        raise ValueError(
            f"로컬 '{local_out}'도 없고 model_data도 유효한 S3 URI가 아닙니다({model_data!r}).\n"
            "  → 02_train_sft_sagemaker를 먼저 완료하거나, MODEL_DIR을 서빙 가능한 로컬 폴더로 직접 지정하세요.")

    import boto3

    cache_dir = os.path.abspath(cache_dir)
    stamp_path = os.path.join(cache_dir, _STAMP)
    prev = None
    if os.path.isfile(stamp_path):
        with open(stamp_path) as f:
            prev = f.read().strip()

    fresh = os.path.isfile(os.path.join(cache_dir, "config.json"))
    if fresh and not force and prev == model_data:
        print("로컬 캐시 재사용:", cache_dir)
        print("  (source:", model_data, ")")
        return cache_dir

    if fresh:
        why = "force=True" if force else f"아티팩트가 바뀜\n    이전: {prev}\n    현재: {model_data}"
        print(f"🔄 캐시를 다시 만듭니다 — {why}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir, exist_ok=True)
    bucket, key = model_data.replace("s3://", "").split("/", 1)
    # 🔴 tar는 cache_dir '안'에 받는다(밖에 두면 여러 트랙/실행이 같은 파일을 덮어써 혼동이 생김).
    #    해제 후 지우므로 디스크에 12GB가 두 벌 남지 않는다.
    tar_path = os.path.join(cache_dir, "_model.tar.gz")
    print(f"downloading {model_data} ...  (수 GB — 처음엔 몇 분 걸립니다)")
    boto3.client("s3", region_name=region).download_file(bucket, key, tar_path)
    print("압축 해제 중...")
    with tarfile.open(tar_path) as t:
        t.extractall(cache_dir)
    os.remove(tar_path)                     # 해제 후 tar 삭제(디스크 절약)
    with open(stamp_path, "w") as f:
        f.write(model_data)
    print("압축 해제 완료:", cache_dir)
    return cache_dir
