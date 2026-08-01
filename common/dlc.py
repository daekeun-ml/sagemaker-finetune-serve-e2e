"""
common/dlc.py — SageMaker Deep Learning Container(DLC) 이미지 URI 해석 (env로 최신 태그 반영)

문제: SageMaker DLC 이미지 태그는 https://aws.github.io/deep-learning-containers/reference/available_images/
     에서 꾸준히 업데이트된다(예: pytorch 2.12.1-cu130-amzn2023-sagemaker, HF DLC 2.x-transformers5.x-...).
     estimator에 transformers_version/pytorch_version만 넘기면 SDK 버전에 매인 '오래된' 이미지 리스트로
     resolve될 수 있다.

해법: image_uri 를 **직접 지정**하고, 그 값을 env로 오버라이드 가능하게 한다.
     available_images 페이지가 갱신되면 코드 수정 없이 env(DLC_IMAGE_URI 또는 DLC_TAG)만 바꾸면 된다.

검증(2026-07-19, available_images 페이지):
  - ECR 레지스트리 계정: 763104351884 (대부분 리전 공용)
  - URI 패턴: 763104351884.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
  ⚠️ 정확한 repository/tag 는 페이지에서 확인해 env로 주입할 것 (여기 기본값은 예시·재확인 대상).

우선순위:
  1) DLC_IMAGE_URI (완전한 URI) 가 있으면 그대로 사용 — 가장 확실.
  2) DLC_REPOSITORY + DLC_TAG 로 조립.
  3) 둘 다 없으면 None → estimator가 transformers_version/pytorch_version 조합으로 resolve(구식 폴백).

우리 train.py는 requirements.txt로 transformers/trl/peft를 직접 설치하므로, 베이스 DLC의
baked-in transformers 버전보다 최신을 써도 컨테이너 안에서 상위로 업그레이드된다(torch/CUDA 호환만 유의).
"""
from __future__ import annotations

import os

# available_images 페이지 검증값(2026-07). 계정/패턴은 안정적, 태그는 자주 바뀜.
DLC_ACCOUNT = os.environ.get("DLC_ACCOUNT", "763104351884")

# ⚠️ 아래 기본값은 '예시'다 — 실행 전 available_images 페이지에서 현행 repository/tag 확인 후 env로 주입.
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


# PyTorch training DLC 자동 해석용 기본 버전(sagemaker.image_uris.retrieve로 pytorch-training 리포 해석).
# ⚠️ retrieve가 지원하는 버전이어야 함(설치된 SDK 기준: 2.8.0/py312가 현재 최신). 그보다 더 최신 태그
#    (예: 2.10.0-gpu-py313-cu130-...-sagemaker-v1.7)는 retrieve가 모를 수 있으니 DLC_IMAGE_URI 로 직접 지정.
# 🔴 CreateTrainingJob은 학습 이미지를 **리전별 private ECR**(763104351884.dkr.ecr.<region>...)에서만
#    허용한다. public.ecr.aws/... URI를 주면 "TrainingImageConfig ... VPC" 에러가 난다. DLC_IMAGE_URI로
#    직접 지정할 때도 반드시 private 리전 ECR 형식을 쓸 것. (retrieve는 자동으로 private URI를 반환함.)
PYTORCH_TRAIN_VERSION = os.environ.get("PYTORCH_TRAIN_VERSION", "2.8.0")
PYTORCH_TRAIN_PY = os.environ.get("PYTORCH_TRAIN_PY", "py312")


def resolve_training_image(region: str) -> str | None:
    """학습용 DLC 이미지 URI 해석.

    우선순위:
      1) DLC_IMAGE_URI (완전 URI) env — 가장 확실.
      2) DLC_REPOSITORY + DLC_TAG env로 조립.
      3) sagemaker.image_uris.retrieve(framework='pytorch', image_scope='training', ...) 자동 해석
         → pytorch-training 리포의 현행 태그를 SDK가 붙여줌(리포명/접미사 손으로 안 찾아도 됨).
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
        print(f"WARNING: pytorch-training image auto-resolve 실패({e}). "
              f"DLC_IMAGE_URI 또는 DLC_REPOSITORY+DLC_TAG env로 지정하세요: {AVAILABLE_IMAGES_URL}")
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
    """DJL LMI 컨테이너 이미지 URI 해석 (서빙 기본 경로).

    우선순위:
      1) LMI_IMAGE_URI (완전 URI) env — 가장 확실.
      2) sagemaker.image_uris.retrieve(framework, version, region) — 권장(리전별 계정·현행 태그 자동).
      3) 실패 시 None → 노트북이 안내.

    framework: backend='vllm' → 'djl-lmi', backend='trtllm' → 'djl-tensorrtllm' (검증 2026-07).
    version: djl-lmi.json 키(예 0.36.0 → 태그 0.36.0-lmi26.0.0-cu130). ⚠️ 실행 전 현행 최신 재확인.
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


# vLLM DLC — AWS 독립 vLLM 컨테이너(djl-lmi와 별개). gemma-4 등 최신 모델 서빙에 권장.
#   available_images의 "vLLM (Ubuntu)" 섹션. 실측 2026-07-21 최신: 0.25.1-gpu-py312-cu130-ubuntu22.04-sagemaker.
#   🔴 gemma-4 서빙엔 vLLM >= 0.19 필요 → 이 DLC(0.25.1)가 충족. LMI 0.36.0(구 vLLM)은 불가.
VLLM_DLC_VERSION = os.environ.get("VLLM_DLC_VERSION", "0.25.1")
VLLM_DLC_TAG_SUFFIX = os.environ.get("VLLM_DLC_TAG_SUFFIX", "gpu-py312-cu130-ubuntu22.04-sagemaker")


def resolve_vllm_image(region: str) -> str | None:
    """vLLM DLC 이미지 URI 해석 (gemma-4 등 최신 모델 서빙 기본 경로).

    우선순위:
      1) VLLM_IMAGE_URI (완전 URI) env — 가장 확실.
      2) sagemaker.image_uris.retrieve(framework='vllm', version, region) — SDK가 알면 사용.
      3) 검증된 패턴으로 조립: <account>.dkr.ecr.<region>.amazonaws.com/vllm:<ver>-<suffix>.
    ⚠️ 태그는 available_images의 vLLM 섹션에서 배포 직전 재확인. gemma-4는 vLLM>=0.19 필요.
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
    except Exception:  # noqa: BLE001 (SDK가 vllm framework/버전을 모를 수 있음 → 패턴 조립 폴백)
        uri = f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/vllm:{version}-{VLLM_DLC_TAG_SUFFIX}"
        print(f"NOTE: vllm image_uris.retrieve 미지원 → 패턴 조립 사용: {uri}\n"
              f"      태그는 available_images에서 재확인: {AVAILABLE_IMAGES_URL}")
        return uri


# HF PyTorch Inference DLC — transformers 기반 서빙(비-vLLM). gemma-4 E계열(KV-sharing)처럼
#   vLLM이 못 여는 모델을 code/inference.py 핸들러로 서빙할 때 사용. transformers 5.5.3 = gemma4 지원(실측).
HF_INFER_TRANSFORMERS_VERSION = os.environ.get("HF_INFER_TRANSFORMERS_VERSION", "5.5.3")
HF_INFER_PYTORCH_VERSION = os.environ.get("HF_INFER_PYTORCH_VERSION", "2.6.0")


def resolve_hf_inference_image(region: str) -> str | None:
    """HF PyTorch Inference DLC 이미지 URI 해석 (transformers 서빙용).

    우선순위:
      1) HF_INFER_IMAGE_URI (완전 URI) env.
      2) image_uris.retrieve(framework='huggingface', image_scope='inference',
         version=<transformers>, base_framework_version='pytorch<pt>') — 검증된 경로.
    실측(2026-07): transformers 5.5.3 / pytorch 2.6.0 →
      763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04
    ⚠️ gemma-4는 transformers>=5.5.3 필요(5.5.3에 gemma4 포함). 태그는 available_images에서 재확인.
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
        print(f"WARNING: HF inference image retrieve 실패({e}). HF_INFER_IMAGE_URI env로 지정: {AVAILABLE_IMAGES_URL}")
        return None


# SGLang DLC — AWS 독립 SGLang 컨테이너(vLLM DLC와 별개). vLLM 대안 서빙 엔진.
#   available_images의 "SGLang" 섹션. ECR 실조회 2026-07-30(us-west-2): 0.5.15 / 0.5.16 존재.
#   SGLang은 gemma4를 지원합니다(sgl-project/sglang에 gemma4 관련 merged PR 존재, 2026-07 확인).
#   ⚠️ 우분투 버전이 vLLM DLC(22.04)와 다릅니다(24.04) — 태그 조립 시 주의.
#
# 🔴 서빙 env 규약 (aws/deep-learning-containers의 sagemaker_entrypoint.sh 소스 확인 2026-07-30):
#   vLLM DLC  : PREFIX="SM_VLLM_"   → 접두사 제거 + 소문자 + '_'→'-' → vllm.entrypoints.openai.api_server 플래그
#   SGLang DLC: PREFIX="SM_SGLANG_" → 같은 변환 → sglang.launch_server 플래그
#   예) SM_VLLM_MAX_MODEL_LEN=2048 → --max-model-len 2048 / SM_SGLANG_TP_SIZE=1 → --tp-size 1
#   따라서 엔진이 아는 플래그면 무엇이든 이 규칙으로 넘길 수 있습니다(별도 화이트리스트 없음).
#   SGLang 기본값: --model-path 미지정 시 /opt/ml/model, --port 8080, --host 0.0.0.0.
SGLANG_DLC_VERSION = os.environ.get("SGLANG_DLC_VERSION", "0.5.15")
SGLANG_DLC_TAG_SUFFIX = os.environ.get("SGLANG_DLC_TAG_SUFFIX", "gpu-py312-cu130-ubuntu24.04-sagemaker")


def resolve_sglang_image(region: str) -> str | None:
    """SGLang DLC 이미지 URI 해석 (vLLM 대안 엔진).

    우선순위:
      1) SGLANG_IMAGE_URI (완전 URI) env — 가장 확실.
      2) 검증된 패턴으로 조립: <account>.dkr.ecr.<region>.amazonaws.com/sglang:<ver>-<suffix>.
    ⚠️ 태그는 배포 직전 재확인:
       aws ecr describe-images --registry-id 763104351884 --repository-name sglang --region <region>
    """
    full = os.environ.get("SGLANG_IMAGE_URI")
    if full:
        return full
    version = os.environ.get("SGLANG_DLC_VERSION", SGLANG_DLC_VERSION)
    suffix = os.environ.get("SGLANG_DLC_TAG_SUFFIX", SGLANG_DLC_TAG_SUFFIX)
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/sglang:{version}-{suffix}"


# 🔴 서빙 이미지 통합 해석 — SERVING_ENGINE(config)과 1:1 대응.
#   'vllm'   → vLLM DLC     (기본. 연속배칭·OpenAI 호환·스트리밍)
#   'sglang' → SGLang DLC   (vLLM 대안. RadixAttention 등)
#   'lmi'    → DJL LMI      (AWS 관리형 컨테이너, 내부 백엔드 vLLM)
# 셋 다 연속 배칭 + OpenAI 호환(messages)이라 호출 코드가 동일하다.
# (resolve_hf_inference_image는 남겨두지만 서빙 엔진 선택지에는 없다 — HF Inference DLC는
#  transformers 단건 서빙용이라 연속 배칭·스트리밍이 없어 이 kit의 서빙 경로에서 제외했다.)
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
            out[name] = resolve_serving_image(region, name) or "(해석 실패 — env로 지정 필요)"
        except Exception as e:  # noqa: BLE001
            out[name] = f"(오류: {e})"
    return out


# ---------------------------------------------------------------------------
# 서빙 env 조립 — 엔진별 키 이름만 다르고 '의미'는 같다
# ---------------------------------------------------------------------------
# 🔴 왜 함수로 묶는가: 같은 설정을 세 엔진의 서로 다른 키로 세 번 쓰면, 값을 하나 바꿀 때
#    한 곳을 빼먹기 쉽다(실측: max_num_seqs를 vLLM 분기에만 넣고 LMI 분기를 놓쳐 OOM 재발).
#    "의미 → 엔진별 키" 매핑을 여기 한 곳에 두고 노트북은 의미만 넘긴다.
#
# 키 유래(라이브 검증 2026-07-31):
#   vLLM / SGLang DLC : entrypoint가 SM_<ENGINE>_ 접두사를 떼고 소문자화 + '_'→'-' 해서 CLI 플래그로
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

    max_num_seqs / gpu_memory_utilization 기본값은 24GB GPU(L4) CUDA OOM 방지용이다 —
    vLLM 기본 max_num_seqs=256은 샘플러 logits 버퍼를 256 x vocab(gemma-4는 262,144) x 4B
    = 256MiB 로 잡아, gemma-4 가중치(~15GB) + KV 캐시와 겹치면 터진다(실측 docs/05 「24GB GPU CUDA OOM」).

    mm_limit: 이미지/오디오 허용 제한 JSON 문자열(예 '{"image": 1}'). None이면 지정하지 않는다.
    hf_token: gated 모델일 때만 전달(없으면 넣지 않는다 — 서빙 메타데이터에 평문 노출 방지).

    ⚠️ gpu_memory_utilization 기본값이 float 0.90 이 아니라 문자열 "0.90" 인 이유: str(0.90)은
       "0.9"가 되어(파이썬이 끝자리 0을 버림) 콘솔에 찍히는 env 값이 노트북 주석과 달라 보인다.
       엔진 동작은 같지만, 값이 그대로 보이는 편이 디버깅에 낫다(float를 넘겨도 정상 동작).
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
