"""로컬 및 SageMaker Managed MLflow 연결을 관리합니다.

이 모듈은 MLflow API를 다시 감싸지 않습니다. 연결 대상을 해석하고, 추적 실패가 파이프라인
실패로 번지지 않도록 기록 호출을 보호합니다.

연결 대상은 비활성, 로컬 SQLite, SageMaker MLflow ARN 세 가지입니다. 설정은
`USE_MLFLOW`, `MLFLOW_APP_NAME`, `MLFLOW_TRACKING_URI` 환경변수로 관리합니다.
"""
from __future__ import annotations

import os
import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from common.logging_utils import get_logger

log = get_logger("mlflow")

# App과 Tracking Server의 ARN 형식이 달라 정규식으로 구분합니다.
_ARN_APP = re.compile(r"^arn:aws[\w-]*:sagemaker:[\w-]+:\d{12}:mlflow-app/(?P<name>[^/]+)$")
_ARN_TS = re.compile(r"^arn:aws[\w-]*:sagemaker:[\w-]+:\d{12}:mlflow-tracking-server/(?P<name>[^/]+)$")

# 로컬 DB에는 엔드포인트 이름과 S3 경로가 기록될 수 있으므로 `.gitignore`에 포함합니다.
DEFAULT_LOCAL_DIR = ".mlflow"
DEFAULT_LOCAL_URI = f"sqlite:///{DEFAULT_LOCAL_DIR}/mlflow.db"


class MlflowTarget:
    """어디에 기록하는지와, 그 사실을 사람에게 어떻게 설명할지."""

    def __init__(self, uri: str | None, kind: str, name: str = "") -> None:
        self.uri = uri              # None 이면 비활성
        self.kind = kind            # disabled | local | app | tracking-server | unknown-arn
        self.name = name

    @property
    def enabled(self) -> bool:
        return bool(self.uri)

    @property
    def is_managed(self) -> bool:
        return self.kind in ("app", "tracking-server", "unknown-arn")

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.kind == "local":
            return f"local sqlite ({self.uri})"
        if self.kind == "app":
            # name은 사람이 지정한 App 이름이 아니라 ARN 끝의 AWS 생성 ID입니다.
            return f"Managed MLflow App (ARN id: {self.name})"
        if self.kind == "tracking-server":
            return f"Managed MLflow Tracking Server '{self.name}' (billed per running hour)"
        return f"MLflow ARN ({self.uri})"

    def __repr__(self) -> str:  # 로그에 그대로 찍히므로 사람이 읽는 형태로
        return f"<MlflowTarget {self.describe()}>"


def resolve_target(tracking_uri: str | None, *, local_uri: str = DEFAULT_LOCAL_URI) -> MlflowTarget:
    """tracking URI를 `MlflowTarget`으로 변환합니다. 빈 값은 비활성입니다."""
    raw = (tracking_uri or "").strip()
    if not raw:
        return MlflowTarget(None, "disabled")
    if raw.lower() in ("local", "sqlite", "offline"):
        _ensure_local_dir(local_uri)
        return MlflowTarget(local_uri, "local")
    if raw.lower() in AUTO_ALIASES:
        # resolve_tracking_uri()를 건너뛴 호출입니다. 유효한 별칭이므로 로컬로 대체합니다.
        log.warning("[mlflow] tracking_uri=%r arrived unresolved. "
                    "This path skipped resolve_tracking_uri(); falling back to local.", raw)
        _ensure_local_dir(local_uri)
        return MlflowTarget(local_uri, "local")
    if m := _ARN_APP.match(raw):
        return MlflowTarget(raw, "app", m.group("name"))
    if m := _ARN_TS.match(raw):
        return MlflowTarget(raw, "tracking-server", m.group("name"))
    if raw.startswith("arn:"):
        # 알 수 없는 ARN은 plugin에 그대로 전달해 새 리소스 형식을 미리 차단하지 않습니다.
        log.warning("[mlflow] unrecognized MLflow ARN shape (using as-is): %s", raw)
        return MlflowTarget(raw, "unknown-arn")
    # sqlite:///나 http://를 직접 지정한 경우
    if raw.startswith(("sqlite:", "postgresql:", "mysql:", "http://", "https://", "file:")):
        return MlflowTarget(raw, "local")
    raise ValueError(
        f"cannot resolve mlflow tracking_uri: {raw!r}\n"
        '  allowed: "" (disabled) | "auto" (find app by name) | "local" | '
        'arn:aws:sagemaker:...:mlflow-app/app-XXXX | '
        "or a full URI such as sqlite:///path.db")


# ---------------------------------------------------------------------------
# 환경변수 스위치
# ---------------------------------------------------------------------------
_TRUE = ("1", "true", "yes", "y", "on")
_FALSE = ("0", "false", "no", "n", "off")


def _env_flag(name: str) -> bool | None:
    """미설정, 켜짐, 꺼짐을 구분하는 환경변수 플래그입니다."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return None
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    log.warning("[mlflow] cannot parse %s=%r; ignoring (allowed: %s / %s).",
                name, raw, "/".join(_TRUE), "/".join(_FALSE))
    return None


def env_switch(config_uri: str, *, local_alias: str = "local") -> tuple[str, str]:
    """환경변수를 반영한 추적 URI와 변경 이유를 반환합니다.

    우선순위:
      1) `MLFLOW_TRACKING_URI`
      2) `USE_MLFLOW=0`
      3) `USE_MLFLOW=1`
      4) 기존 URI
    """
    uri = (config_uri or "").strip()
    reasons: list[str] = []

    env_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if env_uri and env_uri != uri:
        reasons.append(f"MLFLOW_TRACKING_URI env overrode config.yaml ({env_uri})")
        uri = env_uri

    flag = _env_flag("USE_MLFLOW")
    if flag is False:
        if uri:
            reasons.append(f"USE_MLFLOW=0 disables tracking (ignoring the configured {uri})")
        return "", "; ".join(reasons)
    if flag is True and not uri:
        uri = local_alias

    return uri, "; ".join(reasons)


# App을 이름으로 찾는 별칭입니다.
AUTO_ALIASES = ("auto", "app", "managed")


def discover_app_uri(app_name: str = "", *, region: str | None = None) -> str:
    """사용 가능한 App을 이름으로 찾아 ARN을 반환합니다.

    App을 다시 만들면 ARN의 AWS 생성 ID가 바뀔 수 있습니다. 조회 실패는 빈 문자열로 처리해
    호출부가 로컬 SQLite로 대체할 수 있게 합니다.
    """
    name = app_name or DEFAULT_APP_NAME
    try:
        found = find_app(name, region=region)
    except Exception as e:  # noqa: BLE001
        log.warning("[mlflow] could not list apps (falling back to local): %s", e)
        return ""
    return str(found["arn"]) if found else ""


def resolve_tracking_uri(config_uri: str = "", *, app_name: str = "",
                         region: str | None = None, discover: bool = True) -> tuple[str, str]:
    """환경변수와 App 자동 탐색을 반영한 tracking URI를 반환합니다."""
    uri, reason = env_switch(config_uri, local_alias=AUTO_ALIASES[0])
    reasons = [reason] if reason else []

    if uri.lower() in AUTO_ALIASES:
        # auto는 대상 선택 방식이며 추적을 켜는 스위치가 아닙니다.
        if _env_flag("USE_MLFLOW") is not True:
            return "", "; ".join(r for r in reasons if r)
        name = app_name or DEFAULT_APP_NAME
        found = discover_app_uri(name, region=region) if discover else ""
        if found:
            reasons.append(f"found app '{name}'; logging to Managed MLflow")
            uri = found
        else:
            reasons.append(f"no app named '{name}'; logging to local sqlite"
                           " (create one with mlflow_setup.ipynb)")
            uri = "local"

    return uri, "; ".join(r for r in reasons if r)


def target_from_env(config_uri: str = "", *, local_uri: str = DEFAULT_LOCAL_URI,
                    app_name: str = "", region: str | None = None,
                    verbose: bool = True) -> MlflowTarget:
    """노트북에서 환경변수를 반영한 `MlflowTarget`을 만듭니다."""
    uri, reason = resolve_tracking_uri(config_uri, app_name=app_name, region=region)
    if verbose and reason:
        log.info("[mlflow] %s", reason)
    return resolve_target(uri, local_uri=local_uri)


def _ensure_local_dir(local_uri: str) -> None:
    """SQLite 파일의 상위 디렉터리를 만듭니다."""
    if not local_uri.startswith("sqlite:///"):
        return
    path = local_uri[len("sqlite:///"):]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def connect(target: MlflowTarget, experiment: str) -> Any | None:
    """tracking URI와 experiment를 설정합니다. 실패하면 `None`을 반환합니다."""
    if not target.enabled:
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("[mlflow] cannot import mlflow; skipping tracking. Run `uv sync`.")
        return None
    try:
        mlflow.set_tracking_uri(target.uri)
        mlflow.set_experiment(experiment)
    except Exception as e:  # noqa: BLE001. 자격증명, 권한, 네트워크 어느 쪽이든 진행은 계속한다
        log.warning("[mlflow] connection failed (%s). Continuing without tracking: %s", target.describe(), e)
        # 삭제된 experiment 이름은 재사용할 수 없으므로 복원 방법을 함께 안내합니다.
        if "deleted experiment" in str(e).lower():
            log.warning("[mlflow] experiment '%s' is deleted. To restore it:\n"
                        "    from common import mlflow_utils\n"
                        "    mlflow_utils.restore_deleted_experiments(%r)",
                        experiment, target.uri)
        return None
    log.info("[mlflow] %s | experiment=%s", target.describe(), experiment)
    return mlflow


def _str_tags(tags: dict[str, Any] | None) -> dict[str, str] | None:
    """tag 값을 문자열로 바꾸고 `None`은 제거합니다."""
    if not tags:
        return None
    clean = {k: str(v) for k, v in tags.items() if v is not None}
    return clean or None


@contextmanager
def run(target: MlflowTarget, experiment: str, *, run_name: str = "",
        tags: dict[str, Any] | None = None) -> Generator[RunHandle, None, None]:
    """MLflow run 컨텍스트. 비활성이거나 연결 실패면 아무것도 하지 않는 핸들을 준다.

    호출부에 `if enabled:` 분기를 두지 않으려는 구조입니다.
    """
    mlflow = connect(target, experiment)
    if mlflow is None:
        yield RunHandle(None, None)
        return
    active = None
    try:
        # Managed backend가 boolean tag를 거부하므로 모든 값을 문자열로 변환합니다.
        active = mlflow.start_run(run_name=run_name or None, tags=_str_tags(tags))
    except Exception as e:  # noqa: BLE001
        log.warning("[mlflow] could not start run. Continuing without tracking: %s", e)
        yield RunHandle(None, None)
        return
    handle = RunHandle(mlflow, active)
    try:
        yield handle
    except Exception:
        handle.set_tags({"kit.status": "failed"})
        handle._end("FAILED")
        raise
    else:
        handle.set_tags({"kit.status": "finished"})
        handle._end("FINISHED")


class RunHandle:
    """활성 run 기록을 보호합니다. 추적이 꺼져 있으면 모든 메서드가 no-op입니다."""

    def __init__(self, mlflow: Any | None, active: Any | None) -> None:
        self._mlflow = mlflow
        self._active = active

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None and self._active is not None

    @property
    def run_id(self) -> str | None:
        return self._active.info.run_id if self.enabled else None

    @property
    def experiment_id(self) -> str | None:
        """UI 딥링크에 사용하는 숫자 experiment ID입니다."""
        return self._active.info.experiment_id if self.enabled else None

    def _safe(self, what: str, fn: Any) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.warning("[mlflow] failed to log %s (ignored): %s", what, e)

    def log_params(self, params: dict[str, Any]) -> None:
        """parameter를 문자열로 바꾸고 긴 값은 자릅니다."""
        flat = {k: _param_value(v) for k, v in (params or {}).items() if v is not None}
        if flat:
            self._safe("params", lambda: self._mlflow.log_params(flat))

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """숫자만 metric으로 기록합니다. boolean은 parameter이므로 제외합니다."""
        nums = {k: float(v) for k, v in (metrics or {}).items() if _is_number(v)}
        if nums:
            self._safe("metrics", lambda: self._mlflow.log_metrics(nums, step=step))

    def set_tags(self, tags: dict[str, Any]) -> None:
        # start_run과 같은 변환 규칙을 사용합니다.
        if clean := _str_tags(tags):
            self._safe("tags", lambda: self._mlflow.set_tags(clean))

    def log_json(self, obj: Any, filename: str) -> None:
        """작은 JSON을 artifact로 기록합니다. 모델 파일은 S3 URI만 남깁니다."""
        self._safe(f"artifact({filename})",
                   lambda: self._mlflow.log_dict(obj, filename))

    def _end(self, status: str) -> None:
        """현재 run을 닫습니다."""
        self._safe("run 종료", lambda: self._mlflow.end_run(status=status))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _param_value(v: Any, limit: int = 480) -> str:
    s = str(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def training_env(target: MlflowTarget, *, experiment: str,
                 parent_run_id: str | None = None) -> dict[str, str]:
    """학습 컨테이너에 전달할 `MLFLOW_*` 환경변수를 만듭니다.

    부모 run ID는 `MLFLOW_PARENT_RUN_ID`로 전달합니다. 컨테이너가 자식 run을 만든 뒤
    `MLFLOW_RUN_ID`를 설정해야 하므로 부모 ID를 그 변수에 직접 넣지 않습니다.
    """
    if not target.enabled or not target.is_managed:
        return {}
    env = {
        "MLFLOW_TRACKING_URI": str(target.uri),
        "MLFLOW_EXPERIMENT_NAME": experiment,
        # 중첩 하이퍼파라미터를 평평하게 기록합니다.
        "MLFLOW_FLATTEN_PARAMS": "true",
        "HF_MLFLOW_LOG_ARTIFACTS": "false",
        # 시스템 지표는 opt-in입니다. 필수 패키지는 scripts/requirements.txt에 명시합니다.
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true",
        # 짧은 실습에서도 추세가 보이도록 5초 간격을 사용합니다.
        "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": "5",
    }
    if parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    return env


def describe_managed(target: MlflowTarget, *, region: str | None = None) -> dict[str, Any]:
    """Managed MLflow 리소스의 상태와 version을 조회합니다."""
    if not target.is_managed:
        return {}
    try:
        import boto3

        from common import config as _cfg

        sm = boto3.client("sagemaker", region_name=region or _cfg.AWS_REGION)
        if target.kind == "app":
            # App은 ARN, Tracking Server는 이름으로 조회합니다.
            d = sm.describe_mlflow_app(Arn=str(target.uri))
            return {"kind": "app", "name": target.name, "status": d.get("Status"),
                    "mlflow_version": d.get("MlflowVersion"),
                    "artifact_store_uri": d.get("ArtifactStoreUri")}
        if target.kind == "tracking-server":
            d = sm.describe_mlflow_tracking_server(TrackingServerName=target.name)
            return {"kind": "tracking-server", "name": target.name,
                    "status": d.get("TrackingServerStatus"),
                    "mlflow_version": d.get("MlflowVersion"),
                    "size": d.get("TrackingServerSize"),
                    "is_active": d.get("IsActive"),
                    "artifact_store_uri": d.get("ArtifactStoreUri")}
    except Exception as e:  # noqa: BLE001
        log.warning("[mlflow] could not describe resource (still logging): %s", e)
    return {}


def client_version() -> str:
    try:
        import mlflow

        return str(mlflow.__version__)
    except Exception:  # noqa: BLE001
        return "(not installed)"


def warn_version_gap(info: dict[str, Any]) -> None:
    """클라이언트와 관리형 서버의 주 버전 또는 부 버전이 다르면 경고합니다."""
    server = str(info.get("mlflow_version") or "")
    if not server:
        return
    client = client_version()
    if client.startswith("(") or not server:
        return

    def mm(v: str) -> tuple[str, ...]:
        return tuple(v.split(".")[:2])

    if mm(client) != mm(server):
        log.warning(
            "[mlflow] client %s and server %s differ in minor version. AWS docs recommend "
            "matching them; check this first if something behaves oddly.", client, server)


# ===========================================================================
# Managed MLflow App 생성, 조회, 삭제
# ===========================================================================
# 생성과 삭제는 SDK v3를, 조회는 boto3를 사용합니다.

DEFAULT_APP_NAME = "gemma-e2e"


def _region_or_default(region: str | None) -> str:
    if region:
        return region
    from common import config as _cfg

    return _cfg.AWS_REGION


def default_artifact_store_uri(*, region: str | None = None, bucket: str = "",
                              prefix: str = "mlflow") -> str:
    """아티팩트용 S3 URI를 반환합니다. 버킷을 비우면 SageMaker 기본 버킷을 사용합니다."""
    region = _region_or_default(region)
    if bucket:
        return f"s3://{bucket.rstrip('/')}/{prefix}"
    import boto3
    from sagemaker.core.helper.session_helper import Session

    resolved = Session(boto_session=boto3.Session(region_name=region)).default_bucket()
    return f"s3://{resolved}/{prefix}"


def _app_to_dict(app: Any) -> dict[str, Any]:
    """SDK v3 리소스를 노트북에서 읽기 쉬운 dict로 변환합니다."""
    return {
        "arn": str(getattr(app, "arn", "") or ""),
        "name": str(getattr(app, "name", "") or ""),
        "status": str(getattr(app, "status", "") or ""),
        "mlflow_version": str(getattr(app, "mlflow_version", "") or ""),
        "artifact_store_uri": str(getattr(app, "artifact_store_uri", "") or ""),
        "role_arn": str(getattr(app, "role_arn", "") or ""),
        "model_registration_mode": str(getattr(app, "model_registration_mode", "") or ""),
    }


# 삭제된 App도 목록에 남을 수 있으므로 재사용 대상에서 제외합니다.
_APP_DEAD = ("Deleted", "Deleting", "DeleteFailed", "CreateFailed")


def _reset_sdk_client_cache() -> None:
    """SDK v3의 싱글턴 SageMaker 클라이언트 캐시를 초기화합니다."""
    try:
        from sagemaker.core.utils.utils import SageMakerClient

        SageMakerClient.reset()
    except Exception as e:  # noqa: BLE001. SDK 내부 구조가 바뀌어도 실행을 중단하지 않는다
        log.debug("[mlflow] could not reset SDK client cache: %s", e)


def list_apps(*, region: str | None = None) -> list[dict[str, Any]]:
    """지정한 리전의 MLflow App 목록을 반환합니다.

    SDK v3의 싱글턴 클라이언트가 리전을 고정하는 문제를 피하려고 이 조회는 boto3를 사용합니다.
    """
    import boto3

    region = _region_or_default(region)
    sm = boto3.client("sagemaker", region_name=region)
    out: list[dict[str, Any]] = []
    for page in sm.get_paginator("list_mlflow_apps").paginate():
        for s in page.get("Summaries") or []:
            out.append({
                "arn": s.get("Arn", ""), "name": s.get("Name", ""),
                "status": s.get("Status", ""), "mlflow_version": s.get("MlflowVersion", ""),
                # 목록 API가 제공하지 않는 값은 빈 문자열로 둡니다.
                "artifact_store_uri": "", "role_arn": "", "model_registration_mode": "",
            })
    return out


def find_app(name: str = DEFAULT_APP_NAME, *, region: str | None = None,
             include_dead: bool = False) -> dict[str, Any] | None:
    """이름으로 사용 가능한 App을 찾습니다. 없으면 `None`을 반환합니다."""
    dead: list[str] = []
    for app in list_apps(region=region):
        if app["name"] != name:
            continue
        if not include_dead and app["status"] in _APP_DEAD:
            dead.append(app["status"])
            continue
        return app
    if dead:
        log.info("[mlflow] an app named '%s' exists but is not usable (%s). "
                 "A new one must be created.", name, ", ".join(sorted(set(dead))))
    return None


def create_app(name: str = DEFAULT_APP_NAME, *, artifact_store_uri: str = "",
               role_arn: str = "", region: str | None = None,
               model_registration_mode: str | None = None,
               tags: dict[str, str] | None = None,
               wait: bool = True, timeout: int = 900) -> dict[str, Any]:
    """Managed MLflow App을 만듭니다.

    artifact_store_uri: 비우면 SageMaker 기본 버킷의 s3://<bucket>/mlflow
    role_arn:           비우면 common.config.resolve_sagemaker_role()
    model_registration_mode: "AutoModelRegistrationEnabled" | "AutoModelRegistrationDisabled"
                        (비우면 AWS 기본값)

    S3 아티팩트 저장 비용이 발생할 수 있습니다. App 비용은 요금 페이지에서 확인하세요.
    """
    from sagemaker.core.resources import MlflowApp

    region = _region_or_default(region)
    # 싱글턴 클라이언트에 남은 리전 설정을 초기화합니다.
    _reset_sdk_client_cache()
    if not artifact_store_uri:
        artifact_store_uri = default_artifact_store_uri(region=region)
    if not role_arn:
        from common import config as _cfg

        role_arn = _cfg.resolve_sagemaker_role()

    log.info("[mlflow] creating app: name=%s region=%s", name, region)
    log.info("[mlflow]   artifact store : %s", artifact_store_uri)
    log.info("[mlflow]   execution role : %s", role_arn.split("/")[-1])
    log.info("[mlflow]   S3 artifact storage may be billed; empty the prefix when done.")

    kwargs: dict[str, Any] = {
        "name": name,
        "artifact_store_uri": artifact_store_uri,
        "role_arn": role_arn,
        "region": region,
    }
    if model_registration_mode:
        kwargs["model_registration_mode"] = model_registration_mode
    if tags:
        from sagemaker.core.shapes import Tag

        kwargs["tags"] = [Tag(key=k, value=v) for k, v in tags.items()]

    app = MlflowApp.create(**kwargs)
    if app is None:
        raise RuntimeError(
            f"MlflowApp.create returned None (name={name}). Check the status in the console.")

    info = _app_to_dict(app)
    log.info("[mlflow] create requested: status=%s", info["status"] or "?")
    log.info("[mlflow] tracking_uri (no need to write this down; it is found by name):\n    %s", info["arn"])

    if not wait:
        # 생성 대기는 별도 셀이나 호출에서 이어갈 수 있습니다.
        log.info("[mlflow] returning without waiting. Check readiness with:\n"
                 "    mlflow_utils.wait_for_app(%r, region=%r)", name, region)
        return info

    return wait_for_app(info["arn"], region=region, timeout=timeout, name_hint=name)


def wait_for_app(name_or_arn: str = DEFAULT_APP_NAME, *, region: str | None = None,
                 timeout: int = 900, poll_seconds: int = 30,
                 name_hint: str = "") -> dict[str, Any]:
    """App이 준비될 때까지 폴링하고 마지막 상태를 반환합니다.

    제한 시간을 넘겨도 생성 요청은 AWS에서 계속 진행되므로 예외를 올리지 않습니다.
    """
    import time

    region = _region_or_default(region)
    arn = name_or_arn
    if not arn.startswith("arn:"):
        found = find_app(name_or_arn, region=region)
        if not found:
            raise ValueError(f"app '{name_or_arn}' not found in {region}.")
        arn = found["arn"]

    label = name_hint or arn.rsplit("/", 1)[-1]
    log.info("[mlflow] waiting for '%s' to reach Created (measured ~5 min, cap %ds).",
             label, timeout)

    started = time.time()
    last_status = ""
    last_tick = -1
    info: dict[str, Any] = {}
    while True:
        info = describe_app(arn, region=region)
        status = str(info.get("status") or "")
        waited = int(time.time() - started)

        if status != last_status:
            log.info("  mlflow app: %s (%dm %ds elapsed)", status or "?", waited // 60, waited % 60)
            last_status = status
        elif waited // 30 != last_tick:
            # 상태가 같아도 경과 시간을 주기적으로 출력합니다.
            last_tick = waited // 30
            log.info("  waiting... %dm %ds elapsed (status=%s)", waited // 60, waited % 60, status)

        if status in ("Created", "Updated"):
            log.info("[mlflow] ready: status=%s mlflow=%s (took %dm %ds)",
                     status, info.get("mlflow_version") or "?", waited // 60, waited % 60)
            return info
        if status in ("CreateFailed", "UpdateFailed", "DeleteFailed", "Deleted"):
            log.warning("[mlflow] creation did not finish: status=%s. Check the console.", status)
            return info
        if waited >= timeout:
            log.warning("[mlflow] exceeded the %ds cap (status=%s). Creation continues in AWS; "
                        "call wait_for_app() again shortly.", timeout, status)
            return info
        time.sleep(poll_seconds)


def describe_app(name_or_arn: str = DEFAULT_APP_NAME, *,
                 region: str | None = None) -> dict[str, Any]:
    """App 상태를 조회합니다. 이름을 받으면 목록에서 ARN을 찾습니다."""
    import boto3

    region = _region_or_default(region)
    if not name_or_arn.startswith("arn:"):
        found = find_app(name_or_arn, region=region)
        if not found:
            raise ValueError(f"app '{name_or_arn}' not found in {region}.")
        return found
    d = boto3.client("sagemaker", region_name=region).describe_mlflow_app(Arn=name_or_arn)
    return {"arn": d.get("Arn", name_or_arn), "name": d.get("Name", ""),
            "status": d.get("Status", ""), "mlflow_version": d.get("MlflowVersion", ""),
            "artifact_store_uri": d.get("ArtifactStoreUri", ""),
            "role_arn": d.get("RoleArn", ""),
            "model_registration_mode": d.get("ModelRegistrationMode", "")}


def ensure_app(name: str = DEFAULT_APP_NAME, **kwargs: Any) -> dict[str, Any]:
    """같은 이름의 App을 재사용하고, 없으면 새로 만듭니다."""
    region = _region_or_default(kwargs.get("region"))
    existing = find_app(name, region=region)
    if existing:
        log.info("[mlflow] reusing existing app: %s (status=%s, mlflow=%s)",
                 name, existing["status"] or "?", existing["mlflow_version"] or "?")
        log.info("[mlflow] tracking_uri:\n    %s", existing["arn"])
        return existing
    kwargs["region"] = region
    return create_app(name, **kwargs)


def app_ui_url(name_or_arn: str = DEFAULT_APP_NAME, *, region: str | None = None,
               expires_in_seconds: int = 300, session_seconds: int = 0) -> str:
    """MLflow UI 브라우저 세션을 만드는 presigned URL을 반환합니다.

    URL에는 인증 정보가 포함됩니다. 공유하거나 노트북 출력에 남기지 마세요.
    """
    import boto3

    region = _region_or_default(region)
    arn = name_or_arn
    if not arn.startswith("arn:"):
        found = find_app(name_or_arn, region=region)
        if not found:
            raise ValueError(f"app '{name_or_arn}' not found in {region}. "
                             "Create it first with ensure_app().")
        arn = found["arn"]
    kwargs: dict[str, Any] = {"Arn": arn, "ExpiresInSeconds": expires_in_seconds}
    if session_seconds:
        kwargs["SessionExpirationDurationInSeconds"] = session_seconds
    sm = boto3.client("sagemaker", region_name=region)
    return str(sm.create_presigned_mlflow_app_url(**kwargs)["AuthorizedUrl"])


def delete_app(name_or_arn: str = DEFAULT_APP_NAME, *, region: str | None = None,
               wait: bool = True, timeout: int = 900) -> None:
    """App을 삭제합니다. S3 아티팩트는 별도로 정리해야 합니다."""
    from sagemaker.core.resources import MlflowApp

    region = _region_or_default(region)
    arn = name_or_arn
    if not arn.startswith("arn:"):
        found = find_app(name_or_arn, region=region)
        if not found:
            log.info("[mlflow] no app to delete: %s (%s)", name_or_arn, region)
            return
        arn = found["arn"]

    _reset_sdk_client_cache()
    app = MlflowApp.get(arn=arn, region=region)
    if app is None:
        log.info("[mlflow] no app to delete: %s", arn)
        return
    app.delete()
    log.info("[mlflow] delete requested: %s", arn)
    if wait:
        try:
            app.wait_for_delete(timeout=timeout)
            log.info("[mlflow] deleted. The S3 artifact prefix remains; empty it if no longer needed.")
        except Exception as e:  # noqa: BLE001
            log.warning("[mlflow] interrupted while waiting for delete (deletion continues): %s", e)


# ===========================================================================
# IAM 권한 확인과 부여
# ===========================================================================
# `sagemaker:*`는 별도 서비스 prefix인 `sagemaker-mlflow:*`를 포함하지 않습니다.
# 학습 컨테이너에 필요한 MLflowCallback 액션만 정책에 포함합니다.
TRAINING_POLICY_ACTIONS = (
    "sagemaker-mlflow:GetExperimentByName",
    "sagemaker-mlflow:CreateExperiment",
    "sagemaker-mlflow:CreateRun",
    "sagemaker-mlflow:GetRun",
    "sagemaker-mlflow:UpdateRun",
    "sagemaker-mlflow:LogBatch",
    "sagemaker-mlflow:LogMetric",
    "sagemaker-mlflow:LogParam",
    "sagemaker-mlflow:SetTag",
    # SystemMetricsMonitor가 이어 쓸 스텝을 조회할 때 필요합니다.
    "sagemaker-mlflow:GetMetricHistory",
)

DEFAULT_POLICY_NAME = "SageMakerMlflowTracking"

# 권한 확인에 사용할 대표 액션입니다.
_PROBE_ACTIONS = ("sagemaker-mlflow:CreateRun", "sagemaker-mlflow:LogMetric",
                  "sagemaker-mlflow:LogBatch", "sagemaker-mlflow:AccessUI")


def training_role_policy_document() -> dict[str, Any]:
    """학습 실행 역할에 연결할 최소 권한 정책 문서를 반환합니다.

    App ARN으로 리소스 범위를 제한하는 방식은 확인되지 않아 `Resource`는 `"*"`를 사용합니다.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "MlflowTrackingFromTrainingContainer",
            "Effect": "Allow",
            "Action": list(TRAINING_POLICY_ACTIONS),
            "Resource": "*",
        }],
    }


def _role_arn(role: str) -> str:
    """역할 이름 또는 ARN을 완전한 ARN으로 변환합니다."""
    if role.startswith("arn:"):
        return role
    import boto3

    try:
        return str(boto3.client("iam").get_role(RoleName=role)["Role"]["Arn"])
    except Exception:  # noqa: BLE001. 조회 권한이 없으면 path 없는 형태로라도 시도한다
        account = boto3.client("sts").get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/{role}"


def check_mlflow_permissions(*, role: str = "", include_caller: bool = True,
                             actions: tuple[str, ...] = _PROBE_ACTIONS) -> dict[str, Any]:
    """호출자와 학습 실행 역할의 `sagemaker-mlflow` 권한을 확인합니다.

    IAM simulate API는 액션명 자체의 유효성을 검증하지 않습니다.
    """
    import boto3

    iam = boto3.client("iam")
    principals: list[tuple[str, str]] = []
    if include_caller:
        principals.append(("caller", boto3.client("sts").get_caller_identity()["Arn"]))
    if not role:
        from common import config as _cfg

        role = _cfg.resolve_sagemaker_role() or ""
    if role:
        # PolicySourceArn은 완전한 ARN만 받습니다.
        principals.append(("training execution role", _role_arn(role)))

    out: dict[str, Any] = {}
    for label, arn in principals:
        try:
            res = iam.simulate_principal_policy(PolicySourceArn=arn, ActionNames=list(actions))
            denied = [e["EvalActionName"] for e in res["EvaluationResults"]
                      if e["EvalDecision"] != "allowed"]
        except Exception as e:  # noqa: BLE001. 권한 조회 자체가 막힐 수 있다
            log.warning("[mlflow] permission check failed for %s: %s", label, e)
            out[label] = {"arn": arn, "ok": None, "denied": [], "error": str(e)}
            continue
        out[label] = {"arn": arn, "ok": not denied, "denied": denied}
        if denied:
            log.warning("[mlflow] %s (%s) is denied: %s",
                        label, arn.split("/")[-1], ", ".join(denied))
        else:
            log.info("[mlflow] %s (%s) has permission", label, arn.split("/")[-1])
    return out


def grant_mlflow_to_role(role: str = "", *, policy_name: str = DEFAULT_POLICY_NAME,
                         document: dict[str, Any] | None = None,
                         attach: bool = True) -> dict[str, Any]:
    """학습 실행 역할에 `sagemaker-mlflow` 최소 권한 정책을 연결합니다.

    role: 실행 역할 이름 또는 ARN. 비우면 `config.resolve_sagemaker_role()`을 사용합니다.

    반복 실행 시 동작:
      - 없으면 만든다
      - 내용이 같으면 그대로 쓴다
      - 내용이 다르면 새 버전을 기본으로 지정한다
      - 이미 붙어 있으면 붙이지 않는다
    """
    import json as _json

    import boto3

    iam = boto3.client("iam")
    if not role:
        from common import config as _cfg

        role = _cfg.resolve_sagemaker_role() or ""
    if not role:
        raise ValueError("could not resolve the training execution role. Pass role= explicitly "
                         "or set the SAGEMAKER_ROLE_ARN env var.")
    role_name = role.rsplit("/", 1)[-1]
    doc = document or training_role_policy_document()
    account = boto3.client("sts").get_caller_identity()["Account"]
    policy_arn = f"arn:aws:iam::{account}:policy/{policy_name}"

    actions = [a for s in doc["Statement"] for a in
               (s["Action"] if isinstance(s["Action"], list) else [s["Action"]])]
    log.info("[mlflow] modifying IAM: role=%s policy=%s (%d actions)",
             role_name, policy_name, len(actions))

    # 정책 생성 또는 버전 갱신
    created = False
    try:
        iam.create_policy(PolicyName=policy_name, PolicyDocument=_json.dumps(doc),
                          Description="sagemaker-mlflow tracking from SageMaker training jobs")
        created = True
        log.info("[mlflow] policy created: %s", policy_arn)
    except iam.exceptions.EntityAlreadyExistsException:
        cur = iam.get_policy(PolicyArn=policy_arn)["Policy"]["DefaultVersionId"]
        existing = iam.get_policy_version(PolicyArn=policy_arn, VersionId=cur)[
            "PolicyVersion"]["Document"]
        if _same_policy(existing, doc):
            log.info("[mlflow] policy already exists with identical content (%s, %s)", policy_name, cur)
        else:
            iam.create_policy_version(PolicyArn=policy_arn,
                                      PolicyDocument=_json.dumps(doc), SetAsDefault=True)
            log.info("[mlflow] policy content differed; set a new default version (previous %s)", cur)

    # 역할에 연결합니다. 이미 연결돼 있으면 건너뜁니다.
    if not attach:
        return {"policy_arn": policy_arn, "role": role_name, "created": created, "attached": None}
    attached = [p["PolicyArn"] for p in
                iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]]
    if policy_arn in attached:
        log.info("[mlflow] already attached to %s", role_name)
    else:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        log.info("[mlflow] attached to %s", role_name)

    # SCP, permission boundary, IAM 전파 상태를 다시 확인합니다.
    check_mlflow_permissions(role=role, include_caller=False)
    return {"policy_arn": policy_arn, "role": role_name, "created": created, "attached": True}


def _same_policy(a: Any, b: Any) -> bool:
    """액션 순서와 list 또는 str 표기 차이를 무시하고 정책을 비교합니다."""
    def norm(doc: Any) -> Any:
        stmts = doc.get("Statement", []) if isinstance(doc, dict) else []
        if isinstance(stmts, dict):
            stmts = [stmts]
        out = []
        for s in stmts:
            def lst(v: Any) -> list[str]:
                if v is None:
                    return []
                return sorted(v) if isinstance(v, list) else [v]
            out.append((s.get("Effect"), tuple(lst(s.get("Action"))),
                        tuple(lst(s.get("Resource")))))
        return sorted(out)
    return norm(a) == norm(b)


# ===========================================================================
# 로컬 UI 포트포워딩 안내
# ===========================================================================
# 기본 바인딩을 유지하면서 접속할 수 있도록 터널 명령을 제공합니다.

def _imds(path: str, timeout: float = 1.0) -> str:
    """EC2 IMDSv2 값을 읽습니다. 사용할 수 없으면 빈 문자열을 반환합니다."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            token = r.read().decode()
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode().strip()
    except Exception:  # noqa: BLE001. EC2가 아니면 빈 문자열을 반환한다
        return ""


def port_forward_hint(*, port: int = 5000, local_port: int = 5555) -> str:
    """`mlflow ui`를 로컬 브라우저에서 열기 위한 안내를 반환합니다.

    `local_port` 기본값은 macOS AirPlay Receiver와의 5000번 포트 충돌을 피하려고 5555로 둡니다.
    """
    iid = _imds("instance-id")
    if not iid:
        return ("EC2 인스턴스 메타데이터를 찾지 못했습니다.\n"
                f"로컬에서 실행 중이면 http://localhost:{port}를 여세요.")

    region = _imds("placement/region")
    public_ip = _imds("public-ipv4")
    vscode = any(os.environ.get(k) for k in ("VSCODE_IPC_HOOK_CLI", "VSCODE_GIT_ASKPASS_NODE"))

    lines = [
        f"EC2 인스턴스: {iid} ({region or '리전 미확인'})",
        f"`mlflow ui`는 127.0.0.1:{port}에서 대기합니다. 로컬 PC로 포트를 전달하세요.",
        "",
    ]
    if vscode:
        lines += [
            "VS Code Remote: 통합 터미널에서 `mlflow ui`를 실행한 뒤 PORTS 패널에서",
            f"전달된 포트를 여세요. 로컬 포트는 {local_port}를 권장합니다.",
            "",
        ]
    lines += [f"로컬 PC에서 다음 명령 중 하나를 실행하세요. 로컬 포트: {local_port}", ""]
    if public_ip:
        lines += [
            "  # SSH",
            f"  ssh -i ~/.ssh/<key-pair>.pem -N -L {local_port}:localhost:{port} "
            f"ubuntu@{public_ip}",
            "",
        ]
    else:
        lines += ["  # 공개 IP가 없으므로 SSM을 사용합니다.", ""]
    lines += [
        "  # SSM (로컬 PC에 session-manager-plugin 필요)",
        f"  aws ssm start-session --target {iid} --region {region or '<region>'} \\",
        "    --document-name AWS-StartPortForwardingSession \\",
        f"    --parameters '{{\"portNumber\":[\"{port}\"],"
        f"\"localPortNumber\":[\"{local_port}\"]}}'",
        "",
        f"터널이 연결되면 http://localhost:{local_port}를 여세요.",
        f"macOS에서 {port}번 포트가 403을 반환하면 AirPlay Receiver 점유 여부를 확인하세요.",
        f"  lsof -nP -iTCP:{port} -sTCP:LISTEN",
        "",
        "기본 바인딩인 127.0.0.1을 유지하세요. MLflow UI에는 자체 인증이 없습니다.",
    ]
    return "\n".join(lines)


def restore_deleted_experiments(target: MlflowTarget | str, *, names: tuple[str, ...] = ()) -> list[str]:
    """삭제된 experiment를 복원하고 이름을 반환합니다."""
    import mlflow
    from mlflow.entities import ViewType

    uri = target if isinstance(target, str) else str(target.uri or "")
    if not uri:
        raise ValueError("tracking URI is empty.")
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()

    restored: list[str] = []
    for exp in client.search_experiments(view_type=ViewType.DELETED_ONLY):
        if names and exp.name not in names:
            continue
        try:
            client.restore_experiment(exp.experiment_id)
        except Exception as e:  # noqa: BLE001. 하나가 실패해도 나머지는 계속 시도한다
            log.warning("[mlflow] failed to restore '%s': %s", exp.name, e)
            continue
        restored.append(exp.name)
        log.info("[mlflow] restored experiment: %s (id=%s)", exp.name, exp.experiment_id)
    if not restored:
        log.info("[mlflow] no deleted experiments to restore.")
    return restored


# ===========================================================================
# UI 링크 (Studio 없이 보기)
# ===========================================================================
def ui_url(target: MlflowTarget, *, experiment_id: str = "", run_id: str = "") -> str:
    """MLflow App UI 딥링크를 반환합니다. App이 아니면 빈 문자열입니다."""
    if target.kind != "app" or not target.uri or not target.name:
        return ""
    try:
        region = str(target.uri).split(":")[3]
    except IndexError:
        return ""
    url = f"https://{target.name}.mlflow.sagemaker.{region}.app.aws"
    if experiment_id and run_id:
        return f"{url}/#/experiments/{experiment_id}/runs/{run_id}"
    if experiment_id:
        return f"{url}/#/experiments/{experiment_id}"
    return url


def presigned_url_command(target: MlflowTarget) -> str:
    """브라우저 세션을 만드는 CLI 명령을 반환합니다. 실행하지는 않습니다."""
    if target.kind != "app" or not target.uri:
        return ""
    try:
        region = str(target.uri).split(":")[3]
    except IndexError:
        return ""
    return (f"aws sagemaker create-presigned-mlflow-app-url --arn {target.uri} "
            f"--region {region} --query AuthorizedUrl --output text")
