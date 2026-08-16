"""로컬 모델을 준비하고 vLLM 서빙 가능 여부를 검사합니다.

KV 공유 텐서 복원 여부는 config만으로 알 수 없으므로 safetensors 헤더의 실제 키를 확인합니다.
"""
from __future__ import annotations

import glob
import json
import os
import struct


# ---------------------------------------------------------------------------
# 1) 체크포인트 키 읽기
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

    engine: 'vllm' 또는 KV 공유 텐서가 누락된 경우 'transformers'
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"{model_dir}에 config.json이 없습니다(tar 구조 확인). 서빙 가능한 모델 디렉토리를 지정하세요.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # *_text 모델은 vision과 audio 모듈 없이 로드됩니다.
    model_type = str(cfg.get("model_type", ""))
    is_text_only = model_type.endswith("_text") or "vision_config" not in cfg

    kv_shared = _cfg_int(cfg, "num_kv_shared_layers", "kv_shared_layers")
    n_layers = _cfg_int(cfg, "num_hidden_layers")

    missing: list[str] = []
    if kv_shared > 0 and n_layers > 0:
        keys = checkpoint_keys(model_dir)
        first = n_layers - kv_shared          # E4B에서는 레이어 24~41이 공유됩니다.
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
          f"  |  vLLM 서빙: " + ("가능" if info["vllm_ok"] else "불가"))

    if info["missing"]:
        print(f"KV-shared 텐서 {len(info['missing'])}개가 없어 vLLM이 체크포인트를 거부합니다.")
        print(f"예: {info['missing'][:2]}")
        print("현재 train.py로 다시 저장하면 base 모델에서 누락된 텐서를 복원합니다.")
        print("재학습 후에도 같으면 prepare_local_model(force=True)로 캐시를 갱신하세요.")
    elif info["kv_shared"] > 0:
        print("KV-shared 텐서가 있어 vLLM으로 서빙할 수 있습니다.")
    else:
        print("KV-sharing을 사용하지 않는 모델이므로 vLLM으로 서빙할 수 있습니다.")


# ---------------------------------------------------------------------------
# 3) 로컬 모델 디렉토리 준비
# ---------------------------------------------------------------------------
_STAMP = ".source_model_data"   # 이 디렉토리가 어느 S3 아티팩트에서 나왔는지 기록


def prepare_local_model(model_data: str | None, region: str, *,
                        local_out: str = "out", cache_dir: str = "local_model",
                        force: bool = False) -> str:
    """검증에 사용할 로컬 모델 디렉토리를 반환합니다.

    우선순위:
      1) local_out에 config.json이 있으면 로컬 산출물을 사용합니다.
      2) 없으면 model_data를 cache_dir에 내려받아 압축을 풉니다.

    model_data가 바뀌거나 force=True이면 캐시를 다시 만듭니다.
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
            "02_train_sft_sagemaker를 먼저 완료하거나 MODEL_DIR을 직접 지정하세요.")

    import boto3

    cache_dir = os.path.abspath(cache_dir)
    stamp_path = os.path.join(cache_dir, _STAMP)
    prev = None
    if os.path.isfile(stamp_path):
        with open(stamp_path) as f:
            prev = f.read().strip()

    fresh = os.path.isfile(os.path.join(cache_dir, "config.json"))
    if fresh and not force and prev == model_data:
        print("로컬 캐시 사용:", cache_dir)
        print("원본:", model_data)
        return cache_dir

    if fresh:
        why = "force=True" if force else f"아티팩트가 바뀜\n    이전: {prev}\n    현재: {model_data}"
        print(f"캐시 재생성: {why}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir, exist_ok=True)
    bucket, key = model_data.replace("s3://", "").split("/", 1)
    # 여러 실행이 같은 파일을 덮어쓰지 않도록 tar 파일도 cache_dir에 저장합니다.
    tar_path = os.path.join(cache_dir, "_model.tar.gz")
    print(f"다운로드 중: {model_data} (처음에는 몇 분 걸릴 수 있습니다)")
    boto3.client("s3", region_name=region).download_file(bucket, key, tar_path)
    print("압축 해제 중...")
    with tarfile.open(tar_path) as t:
        t.extractall(cache_dir)
    os.remove(tar_path)                     # 해제 후 tar 삭제(디스크 절약)
    with open(stamp_path, "w") as f:
        f.write(model_data)
    print("준비 완료:", cache_dir)
    return cache_dir
