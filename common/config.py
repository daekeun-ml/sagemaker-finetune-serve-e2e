"""노트북과 스크립트가 공유하는 전역 설정입니다.

환경 종속 값과 시크릿은 환경변수로 주입하며 코드에 하드코딩하지 않습니다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# .env 로딩
# ---------------------------------------------------------------------------
# 이 모듈은 import 시점에 환경변수를 읽으므로 저장소 루트의 .env를 먼저 불러옵니다.
# 셸에서 지정한 값은 덮어쓰지 않으며 우선순위는 셸 > .env > config.yaml > 기본값입니다.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
try:
    from dotenv import load_dotenv
except ImportError:
    if os.path.isfile(_DOTENV_PATH) and not os.environ.get("AWS_REGION"):
        print(f"NOTE: python-dotenv가 없어 {_DOTENV_PATH}를 읽지 못했습니다. "
              "`uv sync`를 실행하거나 환경변수를 직접 지정하세요.")
else:
    load_dotenv(_DOTENV_PATH)

# ---------------------------------------------------------------------------
# HF_HOME 자동 설정
# ---------------------------------------------------------------------------
# 명시된 값이 없고 ~/hf-cache가 있으면 해당 경로를 사용합니다.
# 우선순위: 기존 HF_HOME env > HF_HOME_DEFAULT env > ~/hf-cache (존재 시).
if not os.environ.get("HF_HOME"):
    _hf_home_default = os.environ.get("HF_HOME_DEFAULT") or os.path.expanduser("~/hf-cache")
    if os.path.isdir(_hf_home_default):
        os.environ["HF_HOME"] = _hf_home_default


# ---------------------------------------------------------------------------
# Gemma 4 프리셋
# ---------------------------------------------------------------------------
# 모든 프리셋은 멀티모달이며 Apache-2.0, ungated 모델입니다. 텍스트 전용 서빙은 학습 스크립트가
# 재내보낸 체크포인트를 사용합니다. E계열과 26B/31B는 transformers 5.5 이상, 12B는 5.10.1 이상이
# 필요합니다. 서빙 이미지 호환성은 common/dlc.py에서 관리합니다.
#
# 병합 단계는 base 모델을 CPU에 올리므로 GPU뿐 아니라 호스트 메모리도 고려해야 합니다. E2B와 E4B의
# KV 공유 레이어는 저장 과정에서 일부 텐서가 빠질 수 있어 학습 스크립트가 저장 직전에 복원합니다.
GEMMA4_PRESETS: dict[str, dict] = {
    "E2B": {  # 계열 최소 크기로 스모크 테스트에 적합합니다.
        "model_id": "google/gemma-4-E2B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        "train_instance": "ml.g5.2xlarge", "infer_instance": "ml.g5.2xlarge",
        # 2.3B effective라 4bit까지 갈 필요가 없다. bf16 LoRA가 더 빠르고 품질 저하도 없다.
        "use_qlora": False, "min_transformers": "5.5.0", "has_audio": True,
        # 저장 시 KV 공유 텐서를 복원해야 합니다.
        "kv_shared": True, "servable_engine": "vllm",
    },
    "E4B": {  # 단일 GPU 실습의 기본값입니다.
        "model_id": "google/gemma-4-E4B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        "train_instance": "ml.g5.2xlarge", "infer_instance": "ml.g5.2xlarge",
        "use_qlora": True, "min_transformers": "5.5.0", "has_audio": True,
        # 저장 시 KV 공유 텐서를 복원해야 합니다.
        "kv_shared": True, "servable_engine": "vllm",
    },
    "12B": {  # 11.95B dense. unified arch, transformers >= 5.10 필수. LoRA 4bit 권장.
        "model_id": "google/gemma-4-12B-it",
        "arch": "Gemma4UnifiedForConditionalGeneration", "model_type": "gemma4_unified",
        "train_instance": "ml.g5.12xlarge", "infer_instance": "ml.g5.12xlarge",
        "use_qlora": True, "min_transformers": "5.10.1", "has_audio": True,
        "kv_shared": False, "servable_engine": "vllm",
    },
    "26B-A4B": {  # MoE: total 25.2B / active 3.8B, 128 experts. audio 미지원(vision만).
        "model_id": "google/gemma-4-26B-A4B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        "train_instance": "ml.g5.12xlarge", "infer_instance": "ml.g5.12xlarge",
        "use_qlora": True, "min_transformers": "5.5.0", "has_audio": False,
        "kv_shared": False, "servable_engine": "vllm",
    },
    "31B": {  # 31.27B dense, 계열 최대. audio 미지원(vision만). KV-sharing 없음.
        "model_id": "google/gemma-4-31B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        # 비양자화 임베딩과 vision tower까지 고려해 L40S 인스턴스를 사용합니다.
        "train_instance": "ml.g6e.12xlarge", "infer_instance": "ml.g6e.12xlarge",
        "use_qlora": True, "min_transformers": "5.5.0", "has_audio": False,
        "kv_shared": False, "servable_engine": "vllm",
        # 일부 레이어에는 v_proj가 없으므로 LoRA 대상은 정규식으로 선택합니다.
        "v_proj_sparse": True,
    },
}
# 크기 선택: MODEL_SIZE env (E2B|E4B|12B|26B-A4B|31B). 기본 E4B(단일 GPU 친화).
MODEL_SIZE = os.environ.get("MODEL_SIZE", "E4B")
if MODEL_SIZE not in GEMMA4_PRESETS:
    raise ValueError(f"MODEL_SIZE={MODEL_SIZE!r} invalid. Choose one of {list(GEMMA4_PRESETS)}")
GEMMA4_PRESET = GEMMA4_PRESETS[MODEL_SIZE]

# MODEL_ID env로 프리셋 model_id를 직접 오버라이드 가능(임의 gemma-4/호환 모델).
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", GEMMA4_PRESET["model_id"])

# 서빙 엔진은 SERVING_ENGINE으로 변경할 수 있습니다.
#   | SERVING_ENGINE  | 컨테이너            | 특징                                     | 이미지 env(완전 URI) |
#   |-----------------|--------------------|------------------------------------------|---------------------|
#   | vllm (기본)      | vLLM DLC           | 연속 배칭, OpenAI 호환, 스트리밍            | VLLM_IMAGE_URI      |
#   | sglang          | SGLang DLC         | vLLM 대안(RadixAttention). gemma4 지원    | SGLANG_IMAGE_URI    |
#   | lmi             | DJL LMI            | AWS 관리형(내부 백엔드 vLLM), OPTION_* env | LMI_IMAGE_URI       |
#   세 엔진 모두 연속 배칭 + OpenAI 호환(messages) 스키마이므로 호출 코드가 동일하다.
#   버전/태그만 바꾸려면 완전 URI 대신 VLLM_DLC_VERSION / SGLANG_DLC_VERSION / LMI_VERSION 등을 쓰세요(common/dlc.py).
_VALID_ENGINES = ("vllm", "sglang", "lmi")
SERVING_ENGINE = os.environ.get("SERVING_ENGINE", GEMMA4_PRESET["servable_engine"]).strip().lower()
if SERVING_ENGINE not in _VALID_ENGINES:
    raise ValueError(f"SERVING_ENGINE={SERVING_ENGINE!r} invalid. Choose one of {list(_VALID_ENGINES)}")

# gated 모델(gemma-3/2/3n)을 당길 때만 필요. gemma-4 계열이면 비워도 됨.
# 토큰 조회 순서: 환경변수, huggingface_hub 저장 토큰
#   (`hf auth login`이 $HF_HOME/token 또는 ~/.cache/huggingface/token 에 저장한 것).
# 이렇게 해야 `hf auth login`만 해도 kit이 토큰을 인식한다(env 재설정 불필요).
# hf login을 특정 HF_HOME(예: ~/hf-cache)로 했다면 노트북도 같은 HF_HOME을 사용해야
#    huggingface_hub이 그 파일을 찾는다.
def get_hf_token() -> str | None:
    """HF 토큰을 조회 시점에 새로 해석(권장 접근자).

    import 시점에 한 번 계산하는 모듈 상수(HF_TOKEN)는 노트북 셀 실행 순서(HF_HOME을
    config import 뒤에 설정하는 경우)에 취약하다. 이 함수는 매 호출마다 새로 조회하므로
    HF_HOME/HF_TOKEN을 나중에 설정해도 정확히 반영된다.
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token  # `hf auth login` 저장 토큰($HF_HOME/token)
        return get_token()
    except Exception:  # noqa: BLE001
        return None


# 편의 스냅샷(import 시점 값). 실행 순서에 안전한 최신 값이 필요하면 get_hf_token()를 쓸 것.
HF_TOKEN = get_hf_token()


# gated 모델을 MODEL_ID로 지정할 때만 True로 설정합니다.
MODEL_IS_GATED = os.environ.get("MODEL_IS_GATED", "0") not in ("0", "", "false", "False")


def get_serving_hf_token() -> str | None:
    """gated 모델을 서빙할 때만 HF 토큰을 반환합니다."""
    return get_hf_token() if MODEL_IS_GATED else None


# ---------------------------------------------------------------------------
# AWS / SageMaker
# ---------------------------------------------------------------------------
# GPU 용량 부족 시 AWS_REGION을 바꿀 수 있습니다. 이미지 URI의 리전도 함께 맞추세요.
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# SageMaker 실행 role. env SAGEMAKER_ROLE_ARN 로 명시하거나(권장: 셸 export, .env에 박지 말 것),
# 비우면 resolve_sagemaker_role()이 자동 탐지한다.
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")


# 신규 계정 폴백: 실행 role이 하나도 없을 때 AmazonSageMaker-DefaultRole을 자동 생성할지.
# 기본값은 False입니다. 자동 생성 role에는 AmazonSageMakerFullAccess가 붙으므로 opt-in으로만 허용합니다.
SAGEMAKER_CREATE_DEFAULT_ROLE = os.environ.get("SAGEMAKER_CREATE_DEFAULT_ROLE", "0") not in ("0", "", "false", "False")


def resolve_sagemaker_role(sagemaker_session=None) -> str:
    """SageMaker 실행 role ARN을 해석합니다.

    우선순위:
      1) env SAGEMAKER_ROLE_ARN (있으면 그대로).
      2) Studio 또는 Notebook 인스턴스에 연결된 role.
      3) IAM에서 찾은 기존 SageMaker 실행 role.
      4) (opt-in) SAGEMAKER_CREATE_DEFAULT_ROLE=1 이면 AmazonSageMaker-DefaultRole 자동 생성
         (신규 계정에 role이 하나도 없을 때). AmazonSageMakerFullAccess가 붙으므로 기본 비활성.
    전부 실패하면 명확한 에러로 안내.
    """
    if SAGEMAKER_ROLE_ARN:
        return SAGEMAKER_ROLE_ARN
    # 2) Studio 또는 Notebook 인스턴스에 연결된 role
    try:
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role(sagemaker_session=sagemaker_session) if sagemaker_session \
            else get_execution_role()
    except Exception:  # noqa: BLE001 (IAM user 등에서 실패하면 3단계로 진행)
        pass
    # 3) 기존 실행 role 자동 탐지
    role = _autodiscover_sagemaker_role()
    if role:
        return role
    # 4) opt-in 최후 폴백: DefaultRole 자동 생성 (신규 계정)
    if SAGEMAKER_CREATE_DEFAULT_ROLE:
        try:
            from sagemaker.core.helper.session_helper import get_execution_role
            print("SAGEMAKER_CREATE_DEFAULT_ROLE=1: AmazonSageMaker-DefaultRole을 생성하거나 조회합니다. "
                  "AmazonSageMakerFullAccess와 IAM 역할 생성 권한이 필요합니다.")
            return get_execution_role(sagemaker_session=sagemaker_session, use_default=True) if sagemaker_session \
                else get_execution_role(use_default=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"DefaultRole 생성 실패: {e}. SAGEMAKER_ROLE_ARN env로 직접 지정하세요.") from e
    raise RuntimeError(
        "SageMaker 실행 role을 찾지 못했습니다. 다음 중 하나를 하세요:\n"
        "  1) 셸에서 `export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCT>:role/<name>` 로 기존 role 지정, 또는\n"
        "  2) 신규 계정이라 role이 없으면 `export SAGEMAKER_CREATE_DEFAULT_ROLE=1` 로 DefaultRole 자동 생성"
        "(AmazonSageMakerFullAccess 부착, iam:CreateRole 권한 필요), 또는\n"
        "  3) 콘솔에서 SageMaker 실행 role(AmazonSageMaker-ExecutionRole-*)을 만드세요.")


def _autodiscover_sagemaker_role() -> str | None:
    """IAM에서 sagemaker.amazonaws.com 을 신뢰하는 실행 role을 찾아 ARN 반환(없으면 None)."""
    import json as _json
    try:
        import boto3
        iam = boto3.client("iam", region_name=AWS_REGION)
        best = None
        for page in iam.get_paginator("list_roles").paginate():
            for r in page["Roles"]:
                name = r["RoleName"]
                trust = r.get("AssumeRolePolicyDocument", {})
                trust_s = _json.dumps(trust) if isinstance(trust, dict) else str(trust)
                trusts_sm = "sagemaker.amazonaws.com" in trust_s
                # 학습/배포용 실행 role 우선(Bedrock 소비 role 등은 후순위)
                if trusts_sm and "AmazonSageMaker-ExecutionRole" in name:
                    return r["Arn"]           # 표준 이름을 우선 사용합니다.
                if trusts_sm and best is None and "Bedrock" not in name:
                    best = r["Arn"]           # 후보로 보관
        return best
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: IAM 역할을 자동으로 찾지 못했습니다({type(e).__name__}). "
              "SAGEMAKER_ROLE_ARN을 지정하세요.")
        return None

# 학습 데이터/아티팩트 버킷. 비우면 노트북에서 sagemaker.Session().default_bucket() 사용.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
# S3 prefix를 바꾸면 기존 학습 산출물 경로를 찾지 못할 수 있습니다.
S3_PREFIX = os.environ.get("S3_PREFIX", "gemma-e2e-toolkit")

# ---------------------------------------------------------------------------
# 실험 추적 스위치. 기본값은 꺼짐입니다.
# ---------------------------------------------------------------------------
# 빈 문자열은 미설정, "1"은 켜짐, "0"은 강제 꺼짐입니다.
# 해석 규칙은 common/mlflow_utils.py에만 둡니다.
USE_MLFLOW = os.environ.get("USE_MLFLOW", "")
# MLflow client와 학습 컨테이너가 함께 읽는 표준 환경변수입니다.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# HF DLC 버전은 로컬 transformers 버전과 별개입니다.
# AWS가 게시한 태그 조합인지 실행 전에 available_images.md에서 확인하세요. 값을 비우면 estimator가 조합으로 이미지를 resolve하며,
#    컨테이너 안에서 최신 transformers가 필요하면 scripts/requirements.txt(>=5.14.1)가 업그레이드한다.
HF_TRANSFORMERS_VERSION = os.environ.get("HF_TRANSFORMERS_VERSION", "5.3.0")  # TODO verify (DLC 태그)
HF_PYTORCH_VERSION = os.environ.get("HF_PYTORCH_VERSION", "2.9.0")            # TODO verify (DLC 태그)
HF_PY_VERSION = os.environ.get("HF_PY_VERSION", "py312")                       # TODO verify (DLC 태그)

# 학습과 추론 인스턴스
#   (프리셋마다 크기에 맞는 인스턴스가 다르므로 GEMMA4_PRESET 값을 기본으로 씀. env로 언제든 오버라이드.)
TRAIN_INSTANCE_TYPE = os.environ.get("TRAIN_INSTANCE_TYPE", GEMMA4_PRESET["train_instance"])
INFER_INSTANCE_TYPE = os.environ.get("INFER_INSTANCE_TYPE", GEMMA4_PRESET["infer_instance"])


# ---------------------------------------------------------------------------
# Bedrock 모델
# ---------------------------------------------------------------------------
# inference-profile 접두사가 포함된 모델 ID를 환경변수로 지정할 수 있습니다.
BEDROCK_CLAUDE_MODEL_ID = os.environ.get(
    "BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5"  # global cross-region routing
)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", AWS_REGION)


# ---------------------------------------------------------------------------
# 합성 데이터
# ---------------------------------------------------------------------------
NUM_SYNTHETIC = int(os.environ.get("NUM_SYNTHETIC", "200"))        # 트랙당 합성 목표 건수
NUM_SEED_SAMPLES = int(os.environ.get("NUM_SEED_SAMPLES", "300"))  # 시드 스모크 샘플 수
SYNTH_MAX_WORKERS = int(os.environ.get("SYNTH_MAX_WORKERS", "10")) # 합성 시 동시 Bedrock 호출 수(throttling 시 낮추기)


# ---------------------------------------------------------------------------
# dry-run 스위치 (개발환경 GPU에서 빠른 로컬 검증)
# ---------------------------------------------------------------------------
def is_dry_run() -> bool:
    """DRY_RUN=1이면 소량 데이터로 파이프라인만 검증합니다."""
    return os.environ.get("DRY_RUN", "0") not in ("0", "", "false", "False")


@dataclass
class TrackConfig:
    """트랙별 설정. 각 트랙 노트북이 자신의 값으로 인스턴스화."""
    name: str
    seed_dataset: str            # HF dataset id (정찰 검증됨)
    seed_split: str = "train"
    seed_config: str | None = None
    hf_token_required: bool = False
    # Gemma LoRA 학습 기본값
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    multimodal: bool = False     # True면 이미지-텍스트 트랙이며 합성을 건너뜁니다.
    extra: dict = field(default_factory=dict)


# 4개 트랙 레지스트리 (Gate 2 검증 시드셋). 트랙 노트북에서 import.
TRACKS: dict[str, TrackConfig] = {
    "extraction": TrackConfig(
        name="extraction",
        seed_dataset="glaiveai/glaive-function-calling-v2",  # apache-2.0, ungated
        max_seq_length=2048,  # 툴 스키마 JSON이 길다
    ),
    "classification": TrackConfig(
        name="classification",
        # datasets 5 이상과 호환되는 parquet 미러를 사용합니다.
        seed_dataset="mteb/banking77",  # mit, ungated (parquet)
        max_seq_length=512,
    ),
    "summarization": TrackConfig(
        name="summarization",
        seed_dataset="FiscalNote/billsum",  # cc0-1.0, ungated
        max_seq_length=2048,
    ),
    "domain_qa": TrackConfig(
        name="domain_qa",
        seed_dataset="databricks/databricks-dolly-15k",  # cc-by-sa-3.0, ungated
        max_seq_length=1024,
    ),
    # 멀티모달 트랙: 영수증 이미지에서 구조화 JSON을 추출합니다.
    "mm_extraction": TrackConfig(
        name="mm_extraction",
        seed_dataset="naver-clova-ix/cord-v2",  # cc-by-4.0, 영수증 이미지와 구조화 JSON
        max_seq_length=2048,
        num_train_epochs=2,
        multimodal=True,
    ),
}


# ---------------------------------------------------------------------------
# 서드파티 로그 소음 억제 (import 시 자동)
# ---------------------------------------------------------------------------
# 모든 진입점에 적용되도록 공통 라이브러리의 로그 수준만 낮춥니다.
def _quiet_noisy_loggers() -> None:
    import logging
    for _name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                  "botocore", "boto3", "s3transfer", "filelock", "fsspec"):
        logging.getLogger(_name).setLevel(logging.WARNING)


_quiet_noisy_loggers()
