"""
common/config.py — kit 전역 설정 (플레이스홀더 + 환경변수, 시크릿 하드코딩 금지)

모든 노트북/스크립트가 여기서 설정을 읽는다. 값은 (1) 환경변수 → (2) 안전한 기본값 순.
로컬 GPU dry-run과 SageMaker 실행 양쪽에서 동일하게 동작하도록 설계.

⚠️ 이식성 규칙: AWS account id / role arn / bucket / HF 토큰은 절대 하드코딩하지 말 것.
   반드시 os.environ 또는 노트북에서 주입.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# HF_HOME 자동 설정 (VS Code/.env/.bashrc 로딩 차이에 안 취약하게)
# ---------------------------------------------------------------------------
# VS Code Jupyter는 ~/.bashrc를 상속하지 않고, .env는 워크스페이스 루트 기준이라 커스텀 캐시
# 경로(예: ~/hf-cache)의 토큰/모델을 커널이 못 찾는 일이 흔하다. 그래서 config import 시점에
# HF_HOME이 아직 없고 관례적 캐시 디렉토리가 실재하면 자동으로 지정한다(이미 설정돼 있으면 존중).
# 우선순위: 기존 HF_HOME env > HF_HOME_DEFAULT env > ~/hf-cache (존재 시).
if not os.environ.get("HF_HOME"):
    _hf_home_default = os.environ.get("HF_HOME_DEFAULT") or os.path.expanduser("~/hf-cache")
    if os.path.isdir(_hf_home_default):
        os.environ["HF_HOME"] = _hf_home_default


# ---------------------------------------------------------------------------
# 모델 — Gemma 4 프리셋 (계열 5종 전체: MODEL_SIZE env로 선택)
# ---------------------------------------------------------------------------
# ⚠️ 사실 검증 2026-07-21 (HF raw config.json + AWS available_images 실측):
#   - Gemma 4 전 사이즈 = apache-2.0 + UNGATED (토큰 불필요). 전부 멀티모달(vision; E2B/E4B/12B는 audio도).
#     → 텍스트 전용 gemma-4 공식 체크포인트는 없음. 텍스트 서빙은 re-export(gemma4_text) 또는
#       vLLM --language-model-only / OPTION_LIMIT_MM_PER_PROMPT 로 처리(deploy 노트북 참고).
#   - "E" = effective params(PLE, MoE 아님). "A4B" = active 4B(MoE, 128 experts).
#   - gemma-4 서빙엔 vLLM >= 0.19 필요. AWS 독립 vLLM DLC(vllm:0.25.1-...-sagemaker) 또는
#     최신 DJL LMI(27.0.0=vLLM 0.23.1) 사용. 구 LMI 0.36.0은 불가. (common/dlc.py)
#   - transformers: E계열/26B/31B >= 5.5.0, 12B(unified) >= 5.10.1. (scripts/requirements.txt로 설치)
#     실측(2026-08-01): models/gemma4/ 는 v5.4.0에 없고 v5.5.0에 처음 등장(404 → 200).
#     gemma4_unified(12B)는 v5.9.0까지 없고 v5.10.1에서 등장 — 그래서 12B만 floor가 다릅니다.
#   재배포/서빙 전 live 모델 페이지 + available_images 재확인.
# ⚠️ 파라미터 수를 HF API로 읽을 때: `safetensors.total` 은 tie_word_embeddings=True인 모델에서
#    embedding을 이중 계산합니다(31B 실측: total 32,682,372,656 vs 실제 31,273,088,876 — 1.4B 과대).
#    `parameters.BF16` 또는 model.safetensors.index.json 의 metadata.total_parameters 를 쓰세요.
# 🔴 인스턴스 선택 시 GPU뿐 아니라 '호스트 RAM'도 본다: QLoRA 학습은 GPU에 들어가지만, 학습 후 merge/re-export가
#    base 모델을 bf16 full로 CPU에 로드하므로 RAM이 병목이다(초기 버전은 여기서 OOM으로 죽었음). train.py는 merge 전
#    학습 모델을 해제하고 base를 CPU low_cpu_mem_usage로 로드해 사본을 최소화 → E4B 실측 peak RAM ≈ 17.5GB(2026-07).
#    g6.2xlarge = L4 24GB GPU + 32GB RAM. 12B/26B는 merge 시 RAM이 더 커 g6.12xlarge(RAM 큼)로 둔다.
# 🔴 servable_engine — 전 사이즈 'vllm'이 기본입니다(실측 2026-07-30).
#   E2B/E4B는 num_kv_shared_layers>0(KV-sharing)인데, 여기에 오래 알려진 함정이 있습니다:
#     · transformers는 KV-shared 레이어에 k_norm/k_proj/v_proj 모듈을 **아예 만들지 않으므로**
#       (modeling_gemma4.py "Layers sharing kv states don't need any weight matrices"),
#       파인튜닝 후 save_pretrained로 저장하면 원본에 있던 그 텐서가 소실됩니다(E4B 실측 54개).
#     · vLLM Gemma4Attention은 k_norm을 전 레이어에 등록하므로 → "weights not initialized ...k_norm"
#       ValueError로 엔진 초기화 실패(vLLM 이슈 #44788).
#   ⚠️ 따라서 #44788은 "E계열은 vLLM 불가"가 아니라 "transformers가 저장한 체크포인트가 vLLM 불가"입니다.
#      원본 google/gemma-4-E4B-it 및 FP8 변형은 그 54개를 모두 갖고 있어 vLLM 0.25.1에서 정상 로드됩니다
#      (safetensors 헤더 직접 확인 + L40S 실측 로드/생성).
#   → 이 kit의 train.py/train_grpo.py는 저장 직전에 그 텐서를 base에서 복원합니다
#     (_revive_kv_shared_from_base). 연산에 쓰이지 않는 dead weight라 정확도에 무해합니다.
#     실측: 복원 전 665키 → vLLM 실패 / 복원 후 719키(원본과 동일) → vLLM 로드 OK,
#     생성 결과가 transformers와 완전 일치('Paris' == 'Paris').
#   → 서빙 엔진 선택지는 vllm(기본) / sglang / lmi 셋뿐입니다. 셋 다 연속 배칭 + OpenAI 호환이라
#     호출 코드가 같습니다. 아래 SERVING_ENGINE 참조.
GEMMA4_PRESETS: dict[str, dict] = {
    "E2B": {  # effective 2.3B (on-disk 5.12B — PLE가 2.39B, 전체의 46.7%). 계열 최소. 스모크 테스트에 적합.
        "model_id": "google/gemma-4-E2B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        "train_instance": "ml.g5.2xlarge", "infer_instance": "ml.g5.2xlarge",
        # 2.3B effective라 4bit까지 갈 필요가 없다. bf16 LoRA가 더 빠르고 품질 저하도 없다.
        "use_qlora": False, "min_transformers": "5.5.0", "has_audio": True,
        # num_kv_shared_layers=20 / 35층 → 레이어 15~34가 shared. E4B와 같은 dead-weight 소실이 발생하므로
        # 저장 직전 복원이 필요하다(train.py의 _revive_kv_shared_from_base가 자동 처리).
        "kv_shared": True, "servable_engine": "vllm",
    },
    "E4B": {  # effective 4.5B (~8B w/ PLE embeddings). 단일 L4 24GB QLoRA 여유. merge peak RAM~17.5GB<32GB OK. 기본값.
        "model_id": "google/gemma-4-E4B-it",
        "arch": "Gemma4ForConditionalGeneration", "model_type": "gemma4",
        "train_instance": "ml.g5.2xlarge", "infer_instance": "ml.g5.2xlarge",
        "use_qlora": True, "min_transformers": "5.5.0", "has_audio": True,
        # kv_shared=True → 저장 시 dead weight 복원이 필요(train.py가 자동 처리). 복원하면 vLLM 정상.
        "kv_shared": True, "servable_engine": "vllm",
    },
    "12B": {  # 11.95B dense. unified arch, transformers >= 5.10 필수. LoRA 4bit 권장.
        "model_id": "google/gemma-4-12B-it",
        "arch": "Gemma4UnifiedForConditionalGeneration", "model_type": "gemma4_unified",
        "train_instance": "ml.g5.12xlarge", "infer_instance": "ml.g5.12xlarge",
        "use_qlora": True, "min_transformers": "5.10.1", "has_audio": True,
        "kv_shared": False, "servable_engine": "vllm",          # KV-sharing 없음 → vLLM DLC(연속배칭)
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
        # 🔴 g6e = L40S(nominal 48GB). 4bit로도 base resident가 24GB 카드를 넘길 위험이 있어 올린다.
        #    사이징은 AWS 문서의 usable 값(L40S 44GiB / L4·A10G 22GiB)으로 잡는다 — nvidia-smi 실측은
        #    L40S 46,068MiB(=45.0GiB)로, 벤더 nominal보다 작다.
        #    실측 내역: quantizable linear 29.29B → NF4 14.6GB + double-quant 상수 0.46GB,
        #    여기에 4bit로 내려가지 '않는' embed_tokens 1.41B(bf16 2.82GB)와 vision tower 0.58B(1.15GB)를
        #    더해 base만 ≈19.1GB. activation/optimizer까지 얹으면 22GiB로는 sharding이 강제된다.
        #    (params/2 = 15.5GB 어림은 embedding·vision tower가 양자화되지 않는 점을 놓친 값이다.)
        #    호스트 RAM은 병목이 아니다: merge peak ≈68GB(bf16 62.5GB × E4B 실측 오버헤드 1.094배)로
        #    g6e.12xlarge의 384GiB에 크게 여유가 있다.
        "train_instance": "ml.g6e.12xlarge", "infer_instance": "ml.g6e.12xlarge",
        "use_qlora": True, "min_transformers": "5.5.0", "has_audio": False,
        "kv_shared": False, "servable_engine": "vllm",
        # 🔴 31B만의 함정 — attention_k_eq_v=True: global attention 레이어(5,11,...,59 총 10개)는
        #    V를 K로 재사용해 v_proj 모듈이 아예 없습니다(transformers v5.5.0 configuration_gemma4.py:
        #    use_alternative_attention = attention_k_eq_v and not is_sliding → v_proj = None).
        #    실측: v_proj가 60층 중 50층에만 존재. PEFT는 없는 모듈을 조용히 건너뛰므로 v_proj를
        #    이름으로 나열하면 경고 없이 비대칭 adapter가 됩니다. 이 kit은 정규식 target을 쓰므로
        #    (train.py의 lora_targets) 실제 존재하는 모듈만 매칭돼 문제가 없습니다.
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

# 🔴 서빙 엔진 — 기본은 프리셋 값('vllm', 전 사이즈). SERVING_ENGINE env로 전환.
#   | SERVING_ENGINE  | 컨테이너            | 특징                                     | 이미지 env(완전 URI) |
#   |-----------------|--------------------|------------------------------------------|---------------------|
#   | vllm (기본)      | vLLM DLC           | 연속배칭·OpenAI 호환·스트리밍 O            | VLLM_IMAGE_URI      |
#   | sglang          | SGLang DLC         | vLLM 대안(RadixAttention). gemma4 지원    | SGLANG_IMAGE_URI    |
#   | lmi             | DJL LMI            | AWS 관리형(내부 백엔드 vLLM), OPTION_* env | LMI_IMAGE_URI       |
#   세 엔진 모두 연속 배칭 + OpenAI 호환(messages) 스키마이므로 호출 코드가 동일하다.
#   버전/태그만 바꾸려면 완전 URI 대신 VLLM_DLC_VERSION / SGLANG_DLC_VERSION / LMI_VERSION 등을 쓰세요(common/dlc.py).
_VALID_ENGINES = ("vllm", "sglang", "lmi")
SERVING_ENGINE = os.environ.get("SERVING_ENGINE", GEMMA4_PRESET["servable_engine"]).strip().lower()
if SERVING_ENGINE not in _VALID_ENGINES:
    raise ValueError(f"SERVING_ENGINE={SERVING_ENGINE!r} invalid. Choose one of {list(_VALID_ENGINES)}")

# gated 모델(gemma-3/2/3n)을 당길 때만 필요. gemma-4 계열이면 비워도 됨.
# 토큰 조회 순서: (1) env HF_TOKEN / HUGGING_FACE_HUB_TOKEN → (2) huggingface_hub 저장 토큰
#   (`hf auth login`이 $HF_HOME/token 또는 ~/.cache/huggingface/token 에 저장한 것).
# 이렇게 해야 `hf auth login`만 해도 kit이 토큰을 인식한다(env 재설정 불필요).
# ⚠️ hf login을 특정 HF_HOME(예: ~/hf-cache)로 했다면, 노트북 프로세스에도 같은 HF_HOME이 설정돼 있어야
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


# 🔴 모델이 gated인지(HF 약관 동의 + 토큰 필요) 여부. gemma-4 전 사이즈는 apache-2.0/ungated라 False.
#    gemma-3/2 등 gated 모델을 MODEL_ID로 지정해 쓸 때만 True로(env MODEL_IS_GATED=1).
MODEL_IS_GATED = os.environ.get("MODEL_IS_GATED", "0") not in ("0", "", "false", "False")


def get_serving_hf_token() -> str | None:
    """서빙 컨테이너 env에 넣을 HF 토큰(없으면 None).

    🔴 이 kit의 엔드포인트는 학습 산출 모델(S3 model_data)을 서빙하므로 HF에서 가중치를 당기지
    않는다 → 서빙 env에 토큰이 필요 없다. 게다가 gemma-4는 ungated라 더더욱 불필요하다.
    토큰을 서빙 env에 실으면 SageMaker 리소스 메타데이터(describe_endpoint 등)에 평문으로
    남아 유출 위험이 있다. 따라서 gated 모델(MODEL_IS_GATED=1)일 때만 토큰을 반환한다.
    (학습은 base를 HF에서 당기므로 get_hf_token()을 그대로 쓴다.)
    """
    return get_hf_token() if MODEL_IS_GATED else None


# ---------------------------------------------------------------------------
# AWS / SageMaker
# ---------------------------------------------------------------------------
# 🔴 GPU 인스턴스 용량은 리전마다 다르다. InsufficientInstanceCapacity로 막히면 AWS_REGION env만 바꿔 재시도.
#    기본값 us-west-2(오레곤, GPU 용량 여유 큰 편). env AWS_REGION로 언제든 오버라이드.
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# SageMaker 실행 role. env SAGEMAKER_ROLE_ARN 로 명시하거나(권장: 셸 export, .env에 박지 말 것),
# 비우면 resolve_sagemaker_role()이 자동 탐지한다.
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")


# 신규 계정 폴백: 실행 role이 하나도 없을 때 AmazonSageMaker-DefaultRole을 자동 생성할지.
#   🔴 기본 False — get_execution_role(use_default=True)는 AmazonSageMakerFullAccess(광범위)를 붙인 role을
#   생성하므로 opt-in으로만 허용(실수로 과한 권한 role이 만들어지지 않게). env SAGEMAKER_CREATE_DEFAULT_ROLE=1 로 켬.
SAGEMAKER_CREATE_DEFAULT_ROLE = os.environ.get("SAGEMAKER_CREATE_DEFAULT_ROLE", "0") not in ("0", "", "false", "False")


def resolve_sagemaker_role(sagemaker_session=None) -> str:
    """SageMaker 실행 role ARN 해석 — 하드코딩/시크릿 없이.

    우선순위:
      1) env SAGEMAKER_ROLE_ARN (있으면 그대로).
      2) sagemaker get_execution_role() — Studio/Notebook 인스턴스에서 연결된 role 자동.
      3) IAM에서 SageMaker 실행 role 자동 탐지 — IAM user로 로컬 실행할 때(get_execution_role 실패 시).
      4) (opt-in) SAGEMAKER_CREATE_DEFAULT_ROLE=1 이면 AmazonSageMaker-DefaultRole 자동 생성
         (신규 계정에 role이 하나도 없을 때). ⚠️ AmazonSageMakerFullAccess가 붙으므로 기본 비활성.
    전부 실패하면 명확한 에러로 안내.
    """
    if SAGEMAKER_ROLE_ARN:
        return SAGEMAKER_ROLE_ARN
    # 2) get_execution_role (Studio/NB 인스턴스 — 현재 신원이 role일 때)
    try:
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role(sagemaker_session=sagemaker_session) if sagemaker_session \
            else get_execution_role()
    except Exception:  # noqa: BLE001 (IAM user 등에서 실패 — 3)으로)
        pass
    # 3) IAM 자동 탐지 (기존 실행 role 재사용 — 권장)
    role = _autodiscover_sagemaker_role()
    if role:
        return role
    # 4) opt-in 최후 폴백: DefaultRole 자동 생성 (신규 계정)
    if SAGEMAKER_CREATE_DEFAULT_ROLE:
        try:
            from sagemaker.core.helper.session_helper import get_execution_role
            print("SAGEMAKER_CREATE_DEFAULT_ROLE=1 → AmazonSageMaker-DefaultRole 생성/조회 "
                  "(⚠️ AmazonSageMakerFullAccess 부착; iam:CreateRole/AttachRolePolicy 권한 필요)")
            return get_execution_role(sagemaker_session=sagemaker_session, use_default=True) if sagemaker_session \
                else get_execution_role(use_default=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"DefaultRole 생성 실패: {e}. SAGEMAKER_ROLE_ARN env로 직접 지정하세요.") from e
    raise RuntimeError(
        "SageMaker 실행 role을 찾지 못했습니다. 다음 중 하나를 하세요:\n"
        "  1) 셸에서 `export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCT>:role/<name>` 로 기존 role 지정, 또는\n"
        "  2) 신규 계정이라 role이 없으면 `export SAGEMAKER_CREATE_DEFAULT_ROLE=1` 로 DefaultRole 자동 생성"
        "(⚠️ AmazonSageMakerFullAccess 부착, iam:CreateRole 권한 필요), 또는\n"
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
                    return r["Arn"]           # 가장 표준적인 이름 → 즉시 채택
                if trusts_sm and best is None and "Bedrock" not in name:
                    best = r["Arn"]           # 후보로 보관
        return best
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: IAM role 자동 탐지 실패({type(e).__name__}). SAGEMAKER_ROLE_ARN env로 지정하세요.")
        return None

# 학습 데이터/아티팩트 버킷. 비우면 노트북에서 sagemaker.Session().default_bucket() 사용.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
# S3 prefix는 리포 이름과 무관하게 유지한다 — 바꾸면 기존 학습 산출물 경로를 못 찾는다.
S3_PREFIX = os.environ.get("S3_PREFIX", "gemma-e2e-toolkit")

# HF DLC 버전 핀 — SageMaker 학습/추론 컨테이너 이미지 선택용 (로컬 env의 transformers와 별개!).
# ⚠️ 이 값은 AWS가 게시한 HF DLC 태그 조합이어야 한다(임의 최신 아님). 정찰 2026-07 최신 태그:
#    pytorch2.9.0-transformers5.3.0-gpu-py312-cu130. 코드 실행 직전 aws/deep-learning-containers
#    available_images.md 로 현행 태그 재확인. 값을 비우면 estimator가 조합으로 이미지를 resolve하며,
#    컨테이너 안에서 최신 transformers가 필요하면 scripts/requirements.txt(>=5.14.1)가 업그레이드한다.
HF_TRANSFORMERS_VERSION = os.environ.get("HF_TRANSFORMERS_VERSION", "5.3.0")  # TODO verify (DLC 태그)
HF_PYTORCH_VERSION = os.environ.get("HF_PYTORCH_VERSION", "2.9.0")            # TODO verify (DLC 태그)
HF_PY_VERSION = os.environ.get("HF_PY_VERSION", "py312")                       # TODO verify (DLC 태그)

# 학습/추론 인스턴스 — env > 선택된 gemma-4 프리셋 > 폴백.
#   (프리셋마다 크기에 맞는 인스턴스가 다르므로 GEMMA4_PRESET 값을 기본으로 씀. env로 언제든 오버라이드.)
TRAIN_INSTANCE_TYPE = os.environ.get("TRAIN_INSTANCE_TYPE", GEMMA4_PRESET["train_instance"])
INFER_INSTANCE_TYPE = os.environ.get("INFER_INSTANCE_TYPE", GEMMA4_PRESET["infer_instance"])


# ---------------------------------------------------------------------------
# Bedrock (reasoning LLM — agentic loop & 합성 데이터)
# ---------------------------------------------------------------------------
# 🔴 모델 ID 하드코딩 금지 원칙 — 기본값은 env로 override 가능. inference-profile prefix(us./eu./apac./global.) 필수.
#    최신(5+) Claude는 dateless pinned-snapshot 형식(예: anthropic.claude-sonnet-5).
#    기본값 global.anthropic.claude-sonnet-5 는 이 계정에서 list_inference_profiles로 실측 확인(2026-07).
#    ⚠️ 모델 로스터는 자주 바뀌므로 다른 계정/리전에선 Bedrock 콘솔에서 현행 ID 재확인 후 env로 주입.
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
    """DRY_RUN=1 이면 로컬 소량/짧은 학습·합성으로 파이프라인만 검증."""
    return os.environ.get("DRY_RUN", "0") not in ("0", "", "false", "False")


@dataclass
class TrackConfig:
    """트랙별 설정. 각 트랙 노트북이 자신의 값으로 인스턴스화."""
    name: str
    seed_dataset: str            # HF dataset id (정찰 검증됨)
    seed_split: str = "train"
    seed_config: str | None = None
    hf_token_required: bool = False
    # 학습 하이퍼 (Gemma LoRA 관용구 — 정찰 검증)
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    multimodal: bool = False     # True면 이미지→텍스트 트랙(processor·vision, 합성 스킵)
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
        # 🔴 원본 PolyAI/banking77 은 스크립트 기반이라 datasets>=5.0.0에서 로드 불가(실측 2026-07-30)
        #    → parquet 미러 사용. 상세 근거는 tracks/02_classification/track_data.py 독스트링.
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
    # 🖼️ 멀티모달 트랙: 영수증 이미지 → 구조화 JSON (gemma-4 vision).
    "mm_extraction": TrackConfig(
        name="mm_extraction",
        seed_dataset="naver-clova-ix/cord-v2",  # cc-by-4.0, ungated — 영수증 이미지+구조화 JSON
        max_seq_length=2048,
        num_train_epochs=2,
        multimodal=True,
    ),
}


# ---------------------------------------------------------------------------
# 서드파티 로그 소음 억제 (import 시 자동)
# ---------------------------------------------------------------------------
# 🔴 왜 여기서 하나 (실측 2026-07-31): setup_logging()은 00_setup 노트북에서만 호출되므로,
#    02a/03/04 처럼 별도로 실행하는 노트북에서는 소음 억제가 적용되지 않았다.
#    그 결과 huggingface_hub/httpx 의 HTTP INFO 로그가 셀 출력을 덮어 진행 상황이 안 보였고,
#    정상 동작인 404 탐색("로딩 스크립트가 있나?" 확인)이 에러처럼 보였다.
#    config 는 모든 노트북이 반드시 import 하므로 여기서 한 번 낮춰 둔다.
#    (핸들러는 건드리지 않는다 — 라이브러리 위생. setup_logging()이 여전히 구성 주체다.)
def _quiet_noisy_loggers() -> None:
    import logging
    for _name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                  "botocore", "boto3", "s3transfer", "filelock", "fsspec"):
        logging.getLogger(_name).setLevel(logging.WARNING)


_quiet_noisy_loggers()
