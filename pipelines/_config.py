"""config.yaml을 읽어 파이프라인 설정과 환경변수로 변환합니다.

우선순위는 셸과 .env의 환경변수, config.yaml, common/config.py 기본값 순서입니다.
시크릿과 리전처럼 환경에 종속된 값은 config.yaml에서 무시합니다.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# 저장소 루트
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


def _parse_percentiles(raw: Any) -> tuple[float, ...]:
    """쉼표로 구분한 백분위를 정렬된 숫자 튜플로 변환합니다."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = str(raw)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('쉼표로 구분한 백분위 문자열이어야 합니다(예: "50,95,99").')
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        try:
            v = float(part)
        except ValueError:
            raise ValueError(f"{part!r} 는 숫자가 아닙니다.") from None
        if not 0 < v < 100:
            raise ValueError(f"{v} 는 0 초과 100 미만이어야 합니다.")
        out.append(v)
    return tuple(sorted(set(out)))


class ConfigError(ValueError):
    """config.yaml 검증 실패. 문제 키 이름과 허용값을 함께 담는다(KeyError 세 프레임 아래 금지)."""


# ---------------------------------------------------------------------------
# config.yaml에서 허용하지 않는 키
# ---------------------------------------------------------------------------
# 역할 ARN, 토큰, 계정 ID, 리전은 환경변수에서만 읽습니다.
_ENV_ONLY_KEYS = (
    "hf_token", "hugging_face_hub_token", "sagemaker_role_arn", "role_arn", "role",
    "aws_region", "region", "aws_default_region",
    "aws_access_key_id", "aws_secret_access_key", "aws_session_token",
)
_SECRET_PATTERNS = re.compile(r"(token|secret|password|passwd|credential|access_key|_arn|arn$)", re.I)


def _looks_secret(leaf_key: str) -> bool:
    k = leaf_key.strip().lower()
    return k in _ENV_ONLY_KEYS or bool(_SECRET_PATTERNS.search(k))


# ---------------------------------------------------------------------------
# config.yaml이 없을 때 사용할 기본값
# ---------------------------------------------------------------------------
# 값의 변경 이유는 config.yaml에 기록합니다.
DEFAULTS: dict[str, Any] = {
    "model": {
        "size": "E4B",
        "id": "",
        "is_gated": False,
    },
    "training": {
        "instance_type": "ml.g6.2xlarge",
        "max_runtime_hours": 4,
        "epochs": 2,
        "max_train_samples": 200,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2.0e-4,
        "use_qlora": True,
        "merge_adapter": True,
        "lora": {"r": 16, "alpha": 16, "dropout": 0.05},
        "seconds_per_step": 17,
        "dlc": {"repository": "", "tag": ""},
    },
    "grpo": {
        "max_runtime_hours": 6,
        "prompt_source": "synth",
        "num_prompts": 100,
        "epochs": 1,
        "learning_rate": 1.0e-5,
        "num_generations": 8,
        "max_completion_length": 256,
    },
    "serving": {
        "engine": "vllm",
        "instance_type": "ml.g6.2xlarge",
        "max_num_seqs": 32,
        "gpu_memory_utilization": "0.90",
        "tensor_parallel": "1",
        "images": {
            "vllm": {"version": "", "tag_suffix": ""},
            "sglang": {"version": "", "tag_suffix": ""},
            "lmi": {"version": ""},
        },
    },
    "data": {
        "num_seed_samples": 300,
        "num_synthetic": 100,
        "synth_max_workers": 10,
        "s3_prefix": "gemma-e2e-toolkit",
        "dry_run_seed_samples": 8,
        "dry_run_synthetic": 6,
    },
    "evaluation": {
        "num_examples": 50,
        "dry_run_num_examples": 20,
        "workers": 8,
        "judge_max_examples": 20,
    },
    # 부하 벤치마크. 평가(evaluation)와 재는 것이 다릅니다: 평가는 답이 맞는지, 벤치마크는
    # 얼마나 빨리 오는지입니다. 기본값을 작게 둔 이유는 endpoint 가 시간당 과금이라 확인용
    # 실행이 몇 초에 끝나야 하기 때문입니다. 실제 용량 산정을 하려면 값을 올리세요.
    "benchmark": {
        "num_prompts": 20,
        "max_concurrency": 4,
        "request_rate": "inf",
        "input_len": 256,
        "output_len": 128,
        "num_warmups": 2,
        "dry_run_num_prompts": 4,
        # vllm bench serve 의 --metric-percentiles 와 같은 값. vLLM 기본은 "99" 하나지만
        # 평균으로 놓치기 쉬운 꼬리 지연을 확인합니다.
        "percentiles": "50,95,99",
        "save_results": True,
    },
    "aws": {
        "s3_bucket": "",
        "bedrock_model_id": "global.anthropic.claude-sonnet-5",
        "create_default_role": False,
    },
    "runtime": {
        "dry_run": False,
        "log_level": "INFO",
        "poll_seconds": 30,
    },
}

_SECTIONS = tuple(DEFAULTS)


# ---------------------------------------------------------------------------
# 스테이지에서 사용하는 설정 객체
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCfg:
    size: str
    id: str          # 빈 문자열 = 프리셋 model_id 사용
    is_gated: bool


@dataclass(frozen=True)
class LoraCfg:
    r: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class TrainingCfg:
    instance_type: str
    max_runtime_hours: float
    epochs: float
    max_train_samples: int | None      # None = 전량
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    use_qlora: bool
    merge_adapter: bool
    lora: LoraCfg
    seconds_per_step: float            # ETA 추정용(실측값). config.yaml 주석 참고
    dlc_repository: str
    dlc_tag: str


@dataclass(frozen=True)
class GrpoCfg:
    max_runtime_hours: float
    prompt_source: str                 # holdout | synth | failures
    num_prompts: int
    epochs: float
    learning_rate: float
    num_generations: int
    max_completion_length: int


@dataclass(frozen=True)
class ServingCfg:
    engine: str
    instance_type: str
    max_num_seqs: int
    # config.yaml에 적은 문자열 표현을 컨테이너 환경변수까지 유지합니다.
    gpu_memory_utilization: str
    tensor_parallel: str
    images: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DataCfg:
    num_seed_samples: int
    num_synthetic: int
    synth_max_workers: int
    s3_prefix: str
    dry_run_seed_samples: int
    dry_run_synthetic: int


@dataclass(frozen=True)
class EvalCfg:
    num_examples: int
    dry_run_num_examples: int
    workers: int
    judge_max_examples: int


@dataclass(frozen=True)
class BenchCfg:
    num_prompts: int
    dry_run_num_prompts: int
    max_concurrency: int
    request_rate: str                  # "inf" 또는 CLI에 전달할 숫자 문자열
    input_len: int
    output_len: int
    num_warmups: int
    percentiles: tuple[float, ...]     # 예: (50.0, 95.0, 99.0)
    save_results: bool


@dataclass(frozen=True)
class MlflowCfg:
    """환경변수에서 계산한 실험 추적 설정입니다."""
    tracking_uri: str          # "" | SQLite URI | SageMaker MLflow ARN
    app_name: str              # 찾을 MLflow App 이름
    local_uri: str             # 로컬 SQLite URI


@dataclass(frozen=True)
class AwsCfg:
    s3_bucket: str                     # 빈 문자열 = sagemaker default_bucket()
    bedrock_model_id: str
    create_default_role: bool


@dataclass(frozen=True)
class RuntimeCfg:
    dry_run: bool
    log_level: str
    poll_seconds: int


@dataclass(frozen=True)
class PipelineConfig:
    """config.yaml + env 를 합친 실행 설정.

    region, role, hf_token은 환경 종속 값이므로 common.config에서 조회합니다.
    (config.AWS_REGION / config.resolve_sagemaker_role() / config.get_hf_token()).
    """
    source_path: str | None            # 실제로 읽은 config.yaml 경로(없으면 None = 기본값만)
    model: ModelCfg
    training: TrainingCfg
    grpo: GrpoCfg
    serving: ServingCfg
    data: DataCfg
    evaluation: EvalCfg
    benchmark: BenchCfg
    mlflow: MlflowCfg
    aws: AwsCfg
    runtime: RuntimeCfg

    def summary(self) -> str:
        """CLI 상단에 한눈에 찍을 요약(사람이 읽는 용도)."""
        src = self.source_path or "(config.yaml 없음: 내장 기본값)"
        return (
            f"config    : {src}\n"
            f"model     : size={self.model.size} id={self.model.id or '(preset)'}\n"
            f"training  : {self.training.instance_type} epochs={self.training.epochs} "
            f"samples={self.training.max_train_samples or 'all'} "
            f"runtime_limit={self.training.max_runtime_hours}h\n"
            f"serving   : engine={self.serving.engine} {self.serving.instance_type} "
            f"max_num_seqs={self.serving.max_num_seqs} mem_util={self.serving.gpu_memory_utilization}\n"
            f"data      : seed={self.data.num_seed_samples} synth={self.data.num_synthetic}\n"
            f"dry_run   : {self.runtime.dry_run}"
        )


# ---------------------------------------------------------------------------
# YAML 읽기 + 병합
# ---------------------------------------------------------------------------
def _read_yaml(path: str) -> dict[str, Any]:
    """config.yaml 파싱. 파일이 없으면 {} (치명적 오류 아님)."""
    if not os.path.isfile(path):
        return {}
    try:
        import yaml   # PyYAML은 프로젝트 의존성에 포함됩니다.
    except ImportError as e:  # noqa: BLE001
        raise ConfigError(
            f"config.yaml 을 읽으려면 PyYAML 이 필요합니다({e}). `uv sync` 또는 "
            "`uv pip install pyyaml` 후 다시 실행하세요. (파일을 지우면 내장 기본값으로 동작합니다.)"
        ) from e
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} 최상위는 매핑(key: value)이어야 합니다. 현재 형식: {type(loaded).__name__}")
    return loaded


def _strip_secrets(raw: dict[str, Any], path: str) -> dict[str, Any]:
    """시크릿처럼 보이는 키를 재귀적으로 제거하고 크게 경고한다."""
    def walk(node: Any, trail: str) -> Any:
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for k, v in node.items():
            dotted = f"{trail}.{k}" if trail else str(k)
            if _looks_secret(str(k)):
                print(
                    "주의: config.yaml은 커밋되는 파일이므로 시크릿과 환경 전용 키를 무시합니다: "
                    f"`{dotted}` ({path})\n"
                    "   HF 토큰은 `hf auth login`, role ARN 은 `export SAGEMAKER_ROLE_ARN=...`, "
                    "리전은 `export AWS_REGION=...` 으로 주세요."
                )
                continue
            out[k] = walk(v, dotted)
        return out

    return walk(raw, "")


def _merge(base: Any, override: Any) -> Any:
    """dict 는 재귀 병합, 그 외는 override 우선(리스트도 통째 교체)."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _merge(base.get(k), v) if k in base else v
        return merged
    return base if override is None else override


# ---------------------------------------------------------------------------
# 설정 검증
# ---------------------------------------------------------------------------
def _render_errors(errors: list[str], path: str) -> str:
    return (f"config.yaml 검증 실패 ({path}), {len(errors)}건:\n  - "
            + "\n  - ".join(errors))


def _validate(cfg: dict[str, Any], *, allowed_sizes: tuple[str, ...],
              allowed_engines: tuple[str, ...], path: str) -> None:
    errors: list[str] = []

    def num(section: str, key: str, *, minimum: float | None = None,
            maximum: float | None = None, allow_none: bool = False) -> None:
        v = cfg[section].get(key)
        if v is None and allow_none:
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{section}.{key} = {v!r}: 숫자여야 합니다.")
            return
        if minimum is not None and v < minimum:
            errors.append(f"{section}.{key} = {v!r}: {minimum} 이상이어야 합니다.")
        if maximum is not None and v > maximum:
            errors.append(f"{section}.{key} = {v!r}: {maximum} 이하여야 합니다.")

    # 알 수 없는 섹션은 오타일 가능성이 높다(조용히 무시하면 '설정했는데 안 먹는' 상황이 된다).
    for section in cfg:
        if section not in _SECTIONS:
            errors.append(f"알 수 없는 섹션 `{section}`. 허용: {list(_SECTIONS)}")

    # 하위 검사를 시작하기 전에 각 섹션이 매핑인지 확인합니다.
    shape_errors = [
        f"{section} 섹션은 매핑(key: value)이어야 합니다. 현재 형식: "
        f"{type(cfg[section]).__name__} ({cfg[section]!r})."
        for section in _SECTIONS if section in cfg and not isinstance(cfg[section], dict)
    ]
    if shape_errors:
        raise ConfigError(_render_errors(errors + shape_errors, path))

    size = cfg["model"].get("size")
    if size not in allowed_sizes:
        errors.append(f"model.size = {size!r}. 허용: {list(allowed_sizes)}")

    engine = str(cfg["serving"].get("engine", "")).strip().lower()
    if engine not in allowed_engines:
        errors.append(f"serving.engine = {cfg['serving'].get('engine')!r}. 허용: {list(allowed_engines)}")

    src = str(cfg["grpo"].get("prompt_source", "")).strip().lower()
    if src not in ("holdout", "synth", "failures"):
        errors.append(f"grpo.prompt_source = {cfg['grpo'].get('prompt_source')!r}. "
                      "허용: ['holdout', 'synth', 'failures']")

    num("training", "epochs", minimum=0.01)
    num("training", "max_runtime_hours", minimum=0.25)   # 머지/업로드 여유 최소 15분(노트북 실측)
    num("training", "max_train_samples", minimum=1, allow_none=True)
    num("training", "per_device_train_batch_size", minimum=1)
    num("training", "gradient_accumulation_steps", minimum=1)
    num("training", "learning_rate", minimum=1e-8)
    num("training", "seconds_per_step", minimum=0.1)
    lora = cfg["training"].get("lora")
    if isinstance(lora, dict):
        for k, lo, hi in (("r", 1, None), ("alpha", 1, None), ("dropout", 0.0, 1.0)):
            v = lora.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"training.lora.{k} = {v!r}: 숫자여야 합니다.")
            elif lo is not None and v < lo:
                errors.append(f"training.lora.{k} = {v!r}: {lo} 이상이어야 합니다.")
            elif hi is not None and v > hi:
                errors.append(f"training.lora.{k} = {v!r}: {hi} 이하여야 합니다.")
    else:
        errors.append(f"training.lora = {lora!r}: 매핑(r/alpha/dropout)이어야 합니다.")

    num("grpo", "max_runtime_hours", minimum=0.25)
    num("grpo", "num_prompts", minimum=1)
    num("grpo", "epochs", minimum=0.01)
    num("grpo", "learning_rate", minimum=1e-8)
    num("grpo", "num_generations", minimum=2)     # 그룹 내 상대 비교라 1은 의미가 없다
    num("grpo", "max_completion_length", minimum=1)

    num("serving", "max_num_seqs", minimum=1)
    mem = cfg["serving"].get("gpu_memory_utilization")
    try:
        memf = float(mem)
        if not 0.0 < memf <= 1.0:
            errors.append(f"serving.gpu_memory_utilization = {mem!r}: (0, 1] 범위여야 합니다.")
    except (TypeError, ValueError):
        errors.append(f"serving.gpu_memory_utilization = {mem!r}: 0~1 사이 숫자여야 합니다. 문자열 '0.90'을 권장합니다.")

    num("data", "num_seed_samples", minimum=1)
    num("data", "synth_max_workers", minimum=1)
    num("data", "dry_run_seed_samples", minimum=1)
    # 합성 없이 시드만 학습할 수 있도록 합성 건수는 0을 허용합니다.
    num("data", "num_synthetic", minimum=0)
    num("data", "dry_run_synthetic", minimum=0)
    if not str(cfg["data"].get("s3_prefix", "")).strip():
        errors.append("data.s3_prefix = '': 비울 수 없습니다.")

    num("evaluation", "num_examples", minimum=1)
    num("evaluation", "dry_run_num_examples", minimum=1)
    num("evaluation", "workers", minimum=1)
    num("evaluation", "judge_max_examples", minimum=0)   # 0 = LLM-judge 생략(Bedrock 비용 0)

    num("benchmark", "num_prompts", minimum=1)
    num("benchmark", "dry_run_num_prompts", minimum=1)
    num("benchmark", "max_concurrency", minimum=1)
    num("benchmark", "input_len", minimum=1)
    num("benchmark", "output_len", minimum=1)
    num("benchmark", "num_warmups", minimum=0)           # 0 = 워밍업 없음
    # request_rate 만 숫자가 아닐 수 있다("inf"). num() 을 쓰면 문자열에서 걸린다.
    rate = cfg["benchmark"].get("request_rate")
    if not (rate in ("inf", float("inf"))
            or (isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0)):
        errors.append(f"benchmark.request_rate = {rate!r}: 양수 또는 \"inf\"여야 합니다.")
    try:
        _parse_percentiles(cfg["benchmark"].get("percentiles"))
    except ValueError as e:
        errors.append(f"benchmark.percentiles = {cfg['benchmark'].get('percentiles')!r}: {e}")
    if not isinstance(cfg["benchmark"].get("save_results"), bool):
        errors.append(f"benchmark.save_results = {cfg['benchmark'].get('save_results')!r} "
                      ": true/false여야 합니다.")


    num("runtime", "poll_seconds", minimum=1)

    if errors:
        raise ConfigError(_render_errors(errors, path))


# ---------------------------------------------------------------------------
# 환경변수 변환
# ---------------------------------------------------------------------------
def _as_env(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _env_map(cfg: dict[str, Any]) -> dict[str, str]:
    """config.yaml을 common 설정 모듈이 읽는 환경변수로 변환합니다.

    이미지 설정은 리전이 포함된 완전 URI 대신 repository와 tag 조합으로 보관합니다. 완전 URI가
    필요하면 .env의 ``*_IMAGE_URI``를 사용합니다.
    """
    m: dict[str, str] = {}
    model, train, serve = cfg["model"], cfg["training"], cfg["serving"]
    data, ev, aws, rt = cfg["data"], cfg["evaluation"], cfg["aws"], cfg["runtime"]

    m["MODEL_SIZE"] = _as_env(model["size"])
    if str(model.get("id", "")).strip():
        m["MODEL_ID"] = _as_env(model["id"])
    m["MODEL_IS_GATED"] = _as_env(model["is_gated"])

    m["TRAIN_INSTANCE_TYPE"] = _as_env(train["instance_type"])
    m["INFER_INSTANCE_TYPE"] = _as_env(serve["instance_type"])
    dlc = train.get("dlc") or {}
    if str(dlc.get("repository", "")).strip() and str(dlc.get("tag", "")).strip():
        m["DLC_REPOSITORY"] = _as_env(dlc["repository"])
        m["DLC_TAG"] = _as_env(dlc["tag"])

    m["SERVING_ENGINE"] = _as_env(serve["engine"]).strip().lower()
    images = serve.get("images") or {}
    for engine, (ver_key, suffix_key) in {
        "vllm": ("VLLM_DLC_VERSION", "VLLM_DLC_TAG_SUFFIX"),
        "sglang": ("SGLANG_DLC_VERSION", "SGLANG_DLC_TAG_SUFFIX"),
    }.items():
        spec = images.get(engine) or {}
        if str(spec.get("version", "")).strip():
            m[ver_key] = _as_env(spec["version"])
        if str(spec.get("tag_suffix", "")).strip():
            m[suffix_key] = _as_env(spec["tag_suffix"])
    if str((images.get("lmi") or {}).get("version", "")).strip():
        m["LMI_VERSION"] = _as_env(images["lmi"]["version"])

    m["NUM_SEED_SAMPLES"] = _as_env(data["num_seed_samples"])
    m["NUM_SYNTHETIC"] = _as_env(data["num_synthetic"])
    m["SYNTH_MAX_WORKERS"] = _as_env(data["synth_max_workers"])
    m["S3_PREFIX"] = _as_env(data["s3_prefix"])

    m["N_EVAL"] = _as_env(ev["num_examples"])
    m["EVAL_WORKERS"] = _as_env(ev["workers"])

    if str(aws.get("s3_bucket", "")).strip():
        m["S3_BUCKET"] = _as_env(aws["s3_bucket"])
    m["BEDROCK_CLAUDE_MODEL_ID"] = _as_env(aws["bedrock_model_id"])
    m["SAGEMAKER_CREATE_DEFAULT_ROLE"] = _as_env(aws["create_default_role"])

    m["DRY_RUN"] = _as_env(rt["dry_run"])
    m["LOG_LEVEL"] = _as_env(rt["log_level"])
    return m


def prime_environment(cfg: dict[str, Any], *, verbose: bool = True) -> dict[str, str]:
    """config.yaml 값을 환경변수에 주입하고 실제 적용된 값을 반환합니다."""
    effective: dict[str, str] = {}
    overridden: list[str] = []
    for key, value in _env_map(cfg).items():
        existing = os.environ.get(key)
        if existing is None:
            os.environ[key] = value
            effective[key] = value
        else:
            effective[key] = existing
            if existing != value:
                overridden.append(f"{key}={existing} (config.yaml: {value})")

    region = os.environ.get("AWS_REGION")
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region

    if verbose and overridden:
        print("note: 기존 환경변수가 config.yaml보다 우선합니다: " + ", ".join(overridden))
    return effective


# 환경변수 값을 config 구조에 반영할 때 사용하는 경로입니다.
_ENV_TO_PATH: dict[str, tuple[str, ...]] = {
    "MODEL_SIZE": ("model", "size"),
    "TRAIN_INSTANCE_TYPE": ("training", "instance_type"),
    "INFER_INSTANCE_TYPE": ("serving", "instance_type"),
    "SERVING_ENGINE": ("serving", "engine"),
    "NUM_SEED_SAMPLES": ("data", "num_seed_samples"),
    "NUM_SYNTHETIC": ("data", "num_synthetic"),
    "SYNTH_MAX_WORKERS": ("data", "synth_max_workers"),
    "S3_PREFIX": ("data", "s3_prefix"),
    "N_EVAL": ("evaluation", "num_examples"),
    "EVAL_WORKERS": ("evaluation", "workers"),
    "BEDROCK_CLAUDE_MODEL_ID": ("aws", "bedrock_model_id"),
    "LOG_LEVEL": ("runtime", "log_level"),
}


def _reconcile(cfg: dict[str, Any], effective: dict[str, str]) -> dict[str, Any]:
    """환경변수에서 확정된 값을 PipelineConfig에도 반영합니다."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    for env_key, (section, key) in _ENV_TO_PATH.items():
        if env_key not in effective:
            continue
        current = out.get(section, {}).get(key)
        raw = effective[env_key]
        try:
            if isinstance(current, bool):
                value: Any = raw not in ("0", "", "false", "False")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
        except (TypeError, ValueError):
            print(f"WARNING: {env_key}={raw!r}을 {type(current).__name__}로 해석하지 못해 "
                  f"config.yaml 값 {current!r}을 유지합니다.")
            continue
        out[section][key] = value
    return out


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def _mlflow_cfg() -> MlflowCfg:
    """환경변수에서 실험 추적 설정을 계산합니다."""
    from common.mlflow_utils import DEFAULT_APP_NAME, DEFAULT_LOCAL_URI, resolve_tracking_uri

    app_name = (os.environ.get("MLFLOW_APP_NAME") or "").strip() or DEFAULT_APP_NAME
    uri, reason = resolve_tracking_uri("", app_name=app_name)
    if reason:
        print(f"note: {reason}")
    return MlflowCfg(tracking_uri=uri, app_name=app_name, local_uri=DEFAULT_LOCAL_URI)


def load_config(path: str | None = None, *, dry_run: bool | None = None,
                verbose: bool = True) -> PipelineConfig:
    """config.yaml을 읽고 검증해 PipelineConfig를 반환합니다."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    if path and not os.path.isfile(cfg_path):
        raise ConfigError(
            f"--config로 지정한 경로가 없습니다: {cfg_path}\n"
            f"경로를 확인하거나 --config를 빼서 {DEFAULT_CONFIG_PATH} 또는 내장 기본값을 사용하세요.")

    # 1) 허용값은 common 쪽에서 가져온다(값 목록을 여기 복제하면 언젠가 어긋난다).
    #    이 import 는 아직 env 를 채우기 전이라 기존 env 기준으로 해석되지만, 우리가 쓰는 건
    #    상수 목록(GEMMA4_PRESETS 키 / SERVING_ENGINES)뿐이다.
    from common import config as common_config
    from common import dlc as common_dlc
    allowed_sizes = tuple(common_config.GEMMA4_PRESETS)
    allowed_engines = tuple(common_dlc.SERVING_ENGINES)

    # 2) 파일 읽기, 시크릿 제거, 기본값 병합
    raw = _read_yaml(cfg_path)
    raw = _strip_secrets(raw, cfg_path)
    merged = _merge(DEFAULTS, raw)

    # 3) 검증 (문제 키 + 허용값을 메시지에 담아 한 번에 전부)
    _validate(merged, allowed_sizes=allowed_sizes, allowed_engines=allowed_engines, path=cfg_path)

    # 4) dry-run 오버라이드는 검증 뒤에 반영 (CLI 플래그 > config.yaml)
    if dry_run is not None:
        merged["runtime"]["dry_run"] = bool(dry_run)
        os.environ["DRY_RUN"] = "1" if dry_run else "0"

    # 5) 기존 환경변수를 우선해 값을 주입합니다.
    effective = prime_environment(merged, verbose=verbose)

    # 6) 유효 환경변수를 config dict로 되돌려 PipelineConfig와 common.config가
    #    같은 값을 보게 한다(설정 경로가 둘로 갈라지지 않게. _reconcile 독스트링 참고).
    merged = _reconcile(merged, effective)

    # 7) 환경변수로 잘못된 값이 들어온 경우도 검증합니다.
    #    (SERVING_ENGINE=tensorrt 같은 셸 오타가 세 프레임 아래 KeyError 로 터지지 않게).
    _validate(merged, allowed_sizes=allowed_sizes, allowed_engines=allowed_engines,
              path=f"{cfg_path} + env")

    # 8) 이미 import된 common.config를 새 환경변수로 다시 해석합니다.
    #    (reload 하지 않으면 MODEL_SIZE/TRAIN_INSTANCE_TYPE 같은 모듈 상수가 옛 env 값으로 남는다.
    #     config.py 자체가 잘못된 MODEL_SIZE/SERVING_ENGINE 을 명확한 ValueError 로 잡아 주기도 한다.)
    import importlib
    importlib.reload(common_config)

    return _build(merged, cfg_path if os.path.isfile(cfg_path) else None)


def _build(cfg: dict[str, Any], source_path: str | None) -> PipelineConfig:
    train = cfg["training"]
    lora = train["lora"]
    serve = cfg["serving"]
    return PipelineConfig(
        source_path=source_path,
        model=ModelCfg(size=str(cfg["model"]["size"]), id=str(cfg["model"].get("id", "") or ""),
                       is_gated=bool(cfg["model"]["is_gated"])),
        training=TrainingCfg(
            instance_type=str(train["instance_type"]),
            max_runtime_hours=float(train["max_runtime_hours"]),
            epochs=float(train["epochs"]),
            max_train_samples=(int(train["max_train_samples"])
                               if train.get("max_train_samples") is not None else None),
            per_device_train_batch_size=int(train["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
            learning_rate=float(train["learning_rate"]),
            use_qlora=bool(train["use_qlora"]),
            merge_adapter=bool(train["merge_adapter"]),
            lora=LoraCfg(r=int(lora["r"]), alpha=int(lora["alpha"]), dropout=float(lora["dropout"])),
            seconds_per_step=float(train["seconds_per_step"]),
            dlc_repository=str((train.get("dlc") or {}).get("repository", "") or ""),
            dlc_tag=str((train.get("dlc") or {}).get("tag", "") or ""),
        ),
        grpo=GrpoCfg(
            max_runtime_hours=float(cfg["grpo"]["max_runtime_hours"]),
            prompt_source=str(cfg["grpo"]["prompt_source"]).strip().lower(),
            num_prompts=int(cfg["grpo"]["num_prompts"]),
            epochs=float(cfg["grpo"]["epochs"]),
            learning_rate=float(cfg["grpo"]["learning_rate"]),
            num_generations=int(cfg["grpo"]["num_generations"]),
            max_completion_length=int(cfg["grpo"]["max_completion_length"]),
        ),
        serving=ServingCfg(
            engine=str(serve["engine"]).strip().lower(),
            instance_type=str(serve["instance_type"]),
            max_num_seqs=int(serve["max_num_seqs"]),
            gpu_memory_utilization=str(serve["gpu_memory_utilization"]),
            tensor_parallel=str(serve["tensor_parallel"]),
            images={k: {kk: str(vv) for kk, vv in (v or {}).items()}
                    for k, v in (serve.get("images") or {}).items()},
        ),
        data=DataCfg(
            num_seed_samples=int(cfg["data"]["num_seed_samples"]),
            num_synthetic=int(cfg["data"]["num_synthetic"]),
            synth_max_workers=int(cfg["data"]["synth_max_workers"]),
            s3_prefix=str(cfg["data"]["s3_prefix"]),
            dry_run_seed_samples=int(cfg["data"]["dry_run_seed_samples"]),
            dry_run_synthetic=int(cfg["data"]["dry_run_synthetic"]),
        ),
        evaluation=EvalCfg(
            num_examples=int(cfg["evaluation"]["num_examples"]),
            dry_run_num_examples=int(cfg["evaluation"]["dry_run_num_examples"]),
            workers=int(cfg["evaluation"]["workers"]),
            judge_max_examples=int(cfg["evaluation"]["judge_max_examples"]),
        ),
        benchmark=BenchCfg(
            num_prompts=int(cfg["benchmark"]["num_prompts"]),
            dry_run_num_prompts=int(cfg["benchmark"]["dry_run_num_prompts"]),
            max_concurrency=int(cfg["benchmark"]["max_concurrency"]),
            request_rate=str(cfg["benchmark"]["request_rate"]),
            input_len=int(cfg["benchmark"]["input_len"]),
            output_len=int(cfg["benchmark"]["output_len"]),
            num_warmups=int(cfg["benchmark"]["num_warmups"]),
            percentiles=_parse_percentiles(cfg["benchmark"]["percentiles"]),
            save_results=bool(cfg["benchmark"]["save_results"]),
        ),
        mlflow=_mlflow_cfg(),
        aws=AwsCfg(
            s3_bucket=str(cfg["aws"].get("s3_bucket", "") or ""),
            bedrock_model_id=str(cfg["aws"]["bedrock_model_id"]),
            create_default_role=bool(cfg["aws"]["create_default_role"]),
        ),
        runtime=RuntimeCfg(
            dry_run=bool(cfg["runtime"]["dry_run"]),
            log_level=str(cfg["runtime"]["log_level"]),
            poll_seconds=int(cfg["runtime"]["poll_seconds"]),
        ),
    )


if __name__ == "__main__":   # `python pipelines/_config.py` 로 현재 해석 결과 확인
    print(load_config().summary())
