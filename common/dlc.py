"""SageMaker DLC 이미지 URI를 환경변수와 SDK 설정에서 해석합니다.

완전 URI, repository와 tag 조합, SDK 자동 해석 순서로 사용합니다. 태그는 실행 전에
``available_images`` 페이지에서 확인해야 합니다.
"""
from __future__ import annotations

import os

# AWS DLC의 기본 ECR 계정입니다.
DLC_ACCOUNT = os.environ.get("DLC_ACCOUNT", "763104351884")

# 아래 기본값은 예시이며 실행 전에 현행 repository와 tag를 확인해야 합니다.
#    HF DLC(transformers baked-in) 예: repository=huggingface-pytorch-training,
#       tag 형식 "<pt>-transformers<tf>-gpu-py3xx-cu1xx-ubuntu22.04"
#    일반 PyTorch DLC(우리 requirements.txt가 transformers 설치) 예: repository=pytorch-training,
#       tag 형식 "2.12.1-cu130-amzn2023-sagemaker" (사용자 제보 예시) / "2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
_DEFAULT_REPOSITORY = "huggingface-pytorch-training"          # TODO verify (available_images)
_DEFAULT_TAG = "2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04"  # TODO verify (available_images)

AVAILABLE_IMAGES_URL = "https://aws.github.io/deep-learning-containers/reference/available_images/"


def build_dlc_image_uri(region: str, repository: str, tag: str, account: str = DLC_ACCOUNT) -> str:
    """검증된 패턴으로 DLC 이미지 URI 조립."""
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


# SDK 자동 해석에 사용할 PyTorch 학습 이미지 버전입니다.
# 직접 지정할 때는 리전별 private ECR URI를 사용해야 합니다.
PYTORCH_TRAIN_VERSION = os.environ.get("PYTORCH_TRAIN_VERSION", "2.8.0")
PYTORCH_TRAIN_PY = os.environ.get("PYTORCH_TRAIN_PY", "py312")


def resolve_training_image(region: str) -> str | None:
    """학습용 DLC 이미지 URI 해석.

    우선순위:
      1) DLC_IMAGE_URI 완전 URI.
      2) DLC_REPOSITORY + DLC_TAG env로 조립.
      3) sagemaker.image_uris.retrieve(framework='pytorch', image_scope='training', ...) 자동 해석
         및 SDK가 지원하는 태그 선택.
      4) 전부 실패 시 None.
    우리 train.py는 scripts/requirements.txt로 transformers/trl/peft를 직접 설치하므로 베이스가
    순수 PyTorch DLC여도 문제없다(오히려 최신 transformers를 컨테이너 안에서 맞출 수 있어 유리).
    """
    full = os.environ.get("DLC_IMAGE_URI")
    if full:
        return full
    repo = os.environ.get("DLC_REPOSITORY")
    tag = os.environ.get("DLC_TAG")
    if repo and tag:
        return build_dlc_image_uri(region, repo, tag)
    # 자동 해석 (env 없을 때)
    try:
        try:
            from sagemaker.core.image_uris import retrieve  # v3
        except ModuleNotFoundError:
            from sagemaker.image_uris import retrieve  # v2 폴백
        return retrieve(framework="pytorch", region=region, version=PYTORCH_TRAIN_VERSION,
                        image_scope="training", instance_type="ml.g5.2xlarge",
                        py_version=PYTORCH_TRAIN_PY)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: pytorch-training 이미지를 자동으로 찾지 못했습니다({e}). "
              f"DLC_IMAGE_URI 또는 DLC_REPOSITORY와 DLC_TAG를 지정하세요: {AVAILABLE_IMAGES_URL}")
        return None


def resolve_inference_image(region: str) -> str | None:
    """추론(endpoint)용 DLC 이미지 URI 해석. env 우선순위는 학습과 동일하되 INFER_* 우선."""
    full = os.environ.get("INFER_DLC_IMAGE_URI") or os.environ.get("DLC_IMAGE_URI")
    if full:
        return full
    repo = os.environ.get("INFER_DLC_REPOSITORY") or os.environ.get("DLC_REPOSITORY")
    tag = os.environ.get("INFER_DLC_TAG") or os.environ.get("DLC_TAG")
    if repo and tag:
        return build_dlc_image_uri(region, repo, tag)
    return None


def resolve_lmi_image(region: str, version: str = "0.36.0", backend: str = "vllm") -> str | None:
    """DJL LMI 컨테이너 이미지 URI를 해석합니다.

    우선순위:
      1) LMI_IMAGE_URI 완전 URI.
      2) sagemaker.image_uris.retrieve 자동 해석.
      3) 실패 시 None.
    """
    full = os.environ.get("LMI_IMAGE_URI")
    if full:
        return full
    version = os.environ.get("LMI_VERSION", version)
    framework = "djl-tensorrtllm" if backend == "trtllm" else "djl-lmi"
    try:
        # sagemaker SDK v3: image_uris는 sagemaker.core.image_uris 로 이동(v2의 sagemaker.image_uris 아님).
        try:
            from sagemaker.core.image_uris import retrieve  # v3
        except ModuleNotFoundError:
            from sagemaker.image_uris import retrieve  # v2 폴백
        return retrieve(framework=framework, version=version, region=region)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: LMI image_uris.retrieve failed ({e}). Set LMI_IMAGE_URI env directly or re-check: {AVAILABLE_IMAGES_URL}")
        return None


# vLLM DLC
VLLM_DLC_VERSION = os.environ.get("VLLM_DLC_VERSION", "0.25.1")
VLLM_DLC_TAG_SUFFIX = os.environ.get("VLLM_DLC_TAG_SUFFIX", "gpu-py312-cu130-ubuntu22.04-sagemaker")


def resolve_vllm_image(region: str) -> str | None:
    """vLLM DLC 이미지 URI 해석 (gemma-4 등 최신 모델 서빙 기본 경로).

    우선순위:
      1) VLLM_IMAGE_URI 완전 URI.
      2) sagemaker.image_uris.retrieve 자동 해석.
      3) 검증된 패턴으로 조립: <account>.dkr.ecr.<region>.amazonaws.com/vllm:<ver>-<suffix>.
    태그는 배포 직전에 available_images의 vLLM 섹션에서 확인해야 합니다.
    """
    full = os.environ.get("VLLM_IMAGE_URI")
    if full:
        return full
    version = os.environ.get("VLLM_DLC_VERSION", VLLM_DLC_VERSION)
    try:
        try:
            from sagemaker.core.image_uris import retrieve  # v3
        except ModuleNotFoundError:
            from sagemaker.image_uris import retrieve  # v2 폴백
        return retrieve(framework="vllm", version=version, region=region)
    except Exception:  # noqa: BLE001. SDK가 모르는 버전이면 URI 패턴을 사용합니다.
        uri = f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/vllm:{version}-{VLLM_DLC_TAG_SUFFIX}"
        print(f"NOTE: image_uris.retrieve가 이 vLLM 버전을 지원하지 않아 URI를 조립합니다: {uri}\n"
              f"      배포 전에 태그를 확인하세요: {AVAILABLE_IMAGES_URL}")
        return uri


# HF PyTorch Inference DLC. transformers 기반 서빙이며 gemma-4 E계열처럼
#   vLLM이 못 여는 모델을 code/inference.py 핸들러로 서빙할 때 사용. transformers 5.5.3 = gemma4 지원(실측).
HF_INFER_TRANSFORMERS_VERSION = os.environ.get("HF_INFER_TRANSFORMERS_VERSION", "5.5.3")
HF_INFER_PYTORCH_VERSION = os.environ.get("HF_INFER_PYTORCH_VERSION", "2.6.0")


def resolve_hf_inference_image(region: str) -> str | None:
    """transformers 서빙용 HF PyTorch Inference DLC를 해석합니다.

    우선순위:
      1) HF_INFER_IMAGE_URI 완전 URI.
      2) image_uris.retrieve(framework='huggingface', image_scope='inference',
         version=<transformers>, base_framework_version='pytorch<pt>').
    """
    full = os.environ.get("HF_INFER_IMAGE_URI")
    if full:
        return full
    tfver = os.environ.get("HF_INFER_TRANSFORMERS_VERSION", HF_INFER_TRANSFORMERS_VERSION)
    ptver = os.environ.get("HF_INFER_PYTORCH_VERSION", HF_INFER_PYTORCH_VERSION)
    try:
        try:
            from sagemaker.core.image_uris import retrieve  # v3
        except ModuleNotFoundError:
            from sagemaker.image_uris import retrieve  # v2 폴백
        return retrieve(framework="huggingface", region=region, version=tfver,
                        image_scope="inference", instance_type="ml.g5.2xlarge",
                        base_framework_version=f"pytorch{ptver}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: HF 추론 이미지를 찾지 못했습니다({e}). "
              f"HF_INFER_IMAGE_URI를 지정하세요: {AVAILABLE_IMAGES_URL}")
        return None


# SGLang DLC
#
# vLLM과 SGLang 엔트리포인트는 환경변수 접두사를 제거하고 나머지를 CLI 플래그로 변환합니다.
#   따라서 엔진이 아는 플래그면 무엇이든 이 규칙으로 넘길 수 있습니다(별도 화이트리스트 없음).
#   SGLang 기본값: --model-path 미지정 시 /opt/ml/model, --port 8080, --host 0.0.0.0.
SGLANG_DLC_VERSION = os.environ.get("SGLANG_DLC_VERSION", "0.5.15")
SGLANG_DLC_TAG_SUFFIX = os.environ.get("SGLANG_DLC_TAG_SUFFIX", "gpu-py312-cu130-ubuntu24.04-sagemaker")


def resolve_sglang_image(region: str) -> str | None:
    """SGLang DLC 이미지 URI 해석 (vLLM 대안 엔진).

    우선순위:
      1) SGLANG_IMAGE_URI 완전 URI.
      2) 검증된 패턴으로 조립: <account>.dkr.ecr.<region>.amazonaws.com/sglang:<ver>-<suffix>.
    태그는 배포 직전에 다시 확인해야 합니다:
       aws ecr describe-images --registry-id 763104351884 --repository-name sglang --region <region>
    """
    full = os.environ.get("SGLANG_IMAGE_URI")
    if full:
        return full
    version = os.environ.get("SGLANG_DLC_VERSION", SGLANG_DLC_VERSION)
    suffix = os.environ.get("SGLANG_DLC_TAG_SUFFIX", SGLANG_DLC_TAG_SUFFIX)
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/sglang:{version}-{suffix}"


# 서빙 이미지 해석은 SERVING_ENGINE과 일대일로 대응합니다.
#   'vllm': vLLM DLC
#   'sglang': SGLang DLC
#   'lmi': DJL LMI
# 셋 다 연속 배칭 + OpenAI 호환(messages)이라 호출 코드가 동일하다.
# HF Inference DLC는 단건 transformers 서빙용이므로 통합 엔진 목록에서 제외합니다.
_SERVING_IMAGE_RESOLVERS = {
    "vllm": resolve_vllm_image,
    "sglang": resolve_sglang_image,
    "lmi": resolve_lmi_image,
}
SERVING_ENGINES = tuple(_SERVING_IMAGE_RESOLVERS)


def resolve_serving_image(region: str, engine: str) -> str | None:
    """엔진 이름으로 서빙 이미지 URI를 해석한다(엔진별 env 오버라이드 그대로 존중).

    engine: 'vllm' | 'sglang' | 'lmi'  (config.SERVING_ENGINE 값)
    각 엔진의 완전-URI env(VLLM_IMAGE_URI / SGLANG_IMAGE_URI / LMI_IMAGE_URI / HF_INFER_IMAGE_URI)가
    있으면 그것이 최우선이다.
    """
    engine = (engine or "").strip().lower()
    resolver = _SERVING_IMAGE_RESOLVERS.get(engine)
    if resolver is None:
        raise ValueError(f"engine={engine!r} invalid. Choose one of {list(SERVING_ENGINES)}")
    return resolver(region)


def serving_image_table(region: str) -> dict[str, str]:
    """엔진별로 현재 env가 해석하는 이미지 URI 표(노트북에서 한눈에 확인용)."""
    out = {}
    for name in SERVING_ENGINES:
        try:
            out[name] = resolve_serving_image(region, name) or "(해석 실패: 환경변수 지정 필요)"
        except Exception as e:  # noqa: BLE001
            out[name] = f"(오류: {e})"
    return out


# ---------------------------------------------------------------------------
# 서빙 환경변수 조립
# ---------------------------------------------------------------------------
# 의미별 설정을 엔진별 키로 변환하는 매핑을 한 곳에서 관리합니다.
#
# 키 유래(라이브 검증 2026-07-31):
#   vLLM / SGLang DLC : entrypoint가 SM_<ENGINE>_ 접두사를 떼고 소문자화해 CLI 플래그로
#                       전달한다(aws/deep-learning-containers의 sagemaker_entrypoint.sh).
#   DJL LMI           : OPTION_* 를 vLLM EngineArguments로 pass-through 한다(djl-serving vLLM user guide).
_ENV_KEYS: dict[str, dict[str, str]] = {
    #  의미            vllm                              sglang                            lmi
    "model":     {"vllm": "SM_VLLM_MODEL",               "sglang": "SM_SGLANG_MODEL_PATH",         "lmi": "HF_MODEL_ID"},
    "tp":        {"vllm": "SM_VLLM_TENSOR_PARALLEL_SIZE", "sglang": "SM_SGLANG_TP_SIZE",           "lmi": "OPTION_TENSOR_PARALLEL_DEGREE"},
    "max_len":   {"vllm": "SM_VLLM_MAX_MODEL_LEN",        "sglang": "SM_SGLANG_CONTEXT_LENGTH",    "lmi": "OPTION_MAX_MODEL_LEN"},
    "max_seqs":  {"vllm": "SM_VLLM_MAX_NUM_SEQS",         "sglang": "SM_SGLANG_MAX_RUNNING_REQUESTS", "lmi": "OPTION_MAX_ROLLING_BATCH_SIZE"},
    "mem_util":  {"vllm": "SM_VLLM_GPU_MEMORY_UTILIZATION", "sglang": "SM_SGLANG_MEM_FRACTION_STATIC", "lmi": "OPTION_GPU_MEMORY_UTILIZATION"},
    # 멀티모달 입력 허용/차단. SGLang은 멀티모달을 기본 허용하므로 대응 키가 없다(None 처리).
    "mm_limit":  {"vllm": "SM_VLLM_LIMIT_MM_PER_PROMPT",  "sglang": "",                            "lmi": "OPTION_LIMIT_MM_PER_PROMPT"},
}


def serving_env(engine: str, max_model_len: int, max_num_seqs: int = 32,
                gpu_memory_utilization: float | str = "0.90", tensor_parallel: str = "1",
                model_path: str = "/opt/ml/model", mm_limit: str | None = None,
                hf_token: str | None = None) -> dict[str, str]:
    """엔진(vllm|sglang|lmi)에 맞는 서빙 컨테이너 env dict를 만든다.

    max_num_seqs와 gpu_memory_utilization 기본값은 24GB GPU의 CUDA OOM 방지용입니다.
    vLLM 기본 max_num_seqs=256은 샘플러 logits 버퍼를 256 x vocab(gemma-4는 262,144) x 4B
    = 256MiB 로 잡아, gemma-4 가중치(~15GB) + KV 캐시와 겹치면 터진다(실측 docs/05 「24GB GPU CUDA OOM」).

    mm_limit: 이미지/오디오 허용 제한 JSON 문자열(예 '{"image": 1}'). None이면 지정하지 않는다.
    hf_token: gated 모델일 때만 전달하며 없으면 환경변수에 넣지 않습니다.

    gpu_memory_utilization은 설정에 적은 표현을 유지하기 위해 문자열 "0.90"을 기본값으로 사용합니다.
    """
    engine = (engine or "").strip().lower()
    if engine not in SERVING_ENGINES:
        raise ValueError(f"engine={engine!r} invalid. Choose one of {list(SERVING_ENGINES)}")
    vals = {
        "model": model_path,
        "tp": "max" if engine == "lmi" else tensor_parallel,   # LMI 관용구는 'max'
        "max_len": str(max_model_len),
        "max_seqs": str(max_num_seqs),
        "mem_util": str(gpu_memory_utilization),
        "mm_limit": mm_limit,
    }
    if engine == "lmi":
        env = {"OPTION_ROLLING_BATCH": "vllm"}      # LMI 내부 백엔드 = vLLM
    else:
        env = {}
    for meaning, value in vals.items():
        key = _ENV_KEYS[meaning][engine]
        if key and value is not None:
            env[key] = value
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def example_uris(region: str = "us-east-1") -> dict[str, str]:
    """문서/로그용 예시 URI (실제 사용 전 페이지에서 태그 재확인)."""
    return {
        "hf_dlc_default": build_dlc_image_uri(region, _DEFAULT_REPOSITORY, _DEFAULT_TAG),
        "pytorch_dlc_example": build_dlc_image_uri(
            region, "pytorch-training", "2.12.1-cu130-amzn2023-sagemaker"
        ),
        "note": f"현행 태그는 {AVAILABLE_IMAGES_URL} 에서 확인해 DLC_IMAGE_URI 또는 DLC_TAG env로 주입",
    }
