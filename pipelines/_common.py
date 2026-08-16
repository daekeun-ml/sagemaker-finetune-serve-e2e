"""코스별 E2E 스테이지와 상태 저장소를 구현합니다.

이 모듈은 개발자 환경에서 SageMaker 작업을 제출하고 상태를 관리합니다. 컨테이너에서 실행되는
학습 코드는 ``tracks/*/scripts``에 있습니다.

스테이지 간 값은 ``.pipeline_state/<course>.json``에 저장합니다. dry-run은 별도 상태 파일과
데이터 디렉토리를 사용하므로 실제 실행의 재개 판단에 섞이지 않습니다.

  python pipelines/run_extraction.py --stages data,train   # model_data 를 기록
  python pipelines/run_extraction.py --stages deploy       # 그 값을 읽어 배포
  python pipelines/run_extraction.py --dry-run             # 과금 리소스 0으로 전 경로 검증

코스별 진입 스크립트(pipelines/run_<course>.py)는 아래 3줄이면 된다:

    from pipelines._common import main
    if __name__ == "__main__":
        raise SystemExit(main(default_course="extraction"))

코스별 상태 파일을 사용해 ``%store``의 전역 키 충돌을 방지합니다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(REPO_ROOT, ".pipeline_state")
TRACKS_ROOT = os.path.join(REPO_ROOT, "tracks")

# common/ 과 tracks/ 를 import 할 수 있게 (노트북의 SETUP_PATH 와 같은 역할)
for _p in (REPO_ROOT, TRACKS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import mlflow_utils  # noqa: E402
from common.logging_utils import get_logger, hyperlink  # noqa: E402  (경로 세팅 이후여야 한다)
from pipelines._config import PipelineConfig, load_config  # noqa: E402

log = get_logger("pipelines")

STAGE_ORDER = ("data", "train", "grpo", "deploy", "eval", "cleanup")


class StageFailed(RuntimeError):
    """사용자용 오류 메시지를 이미 출력한 스테이지 실패입니다."""

# 에이전트 단계는 대화형 탐색에 적합하므로 노트북에서만 제공합니다.


# ===========================================================================
# 1) 코스 스펙
# ===========================================================================
@dataclass(frozen=True)
class CourseSpec:
    """한 코스의 파이프라인 설정입니다.

    노트북과 CLI가 같은 값을 사용하도록 ``TrackSpec``을 그대로 참조합니다.
    """
    key: str                    # config.TRACKS 키 (extraction/classification/...)
    dir_name: str               # tracks/ 하위 디렉토리
    track: Any                  # tracks._shared_build.TrackSpec
    train_entry: str            # scripts/ 안의 학습 스크립트 (train.py | train_mm.py)
    multimodal: bool = False
    has_synth: bool = True      # 멀티모달 코스는 합성 단계가 없다(이미지 합성은 별개 문제)
    has_eval_stage: bool = True # 멀티모달 코스에는 평가 스테이지가 없음

    @property
    def dir(self) -> str:
        return os.path.join(TRACKS_ROOT, self.dir_name)

    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.dir, "scripts")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.dir, "data")

    @property
    def endpoint_prefix(self) -> str:
        return self.track.endpoint_prefix          # 예: gemma-extraction

    @property
    def train_job_prefix(self) -> str:
        return f"{self.track.endpoint_prefix}-train"

    @property
    def grpo_job_prefix(self) -> str:
        return f"{self.track.endpoint_prefix}-grpo"

    @property
    def serve_max_model_len(self) -> int:
        # 서빙 컨텍스트에는 입력과 생성 토큰이 모두 들어갑니다.
        return self.track.serve_max_model_len or self.track.max_seq_length * 2

    @property
    def gen_max_tokens(self) -> int:
        return self.track.gen_max_tokens

    @property
    def grpo_reward_kind(self) -> str:
        return self.track.grpo_reward_kind


def _mm_track_spec():
    """멀티모달 코스의 ``TrackSpec``을 반환합니다.

    이미지 입력의 텍스트 컨텍스트는 짧지만 정답 JSON을 위해 생성 길이는 768로 둡니다.
    """
    from _shared_build import TrackSpec
    return TrackSpec(
        key="mm_extraction", dir_name="05_multimodal_extraction",
        title="멀티모달 추출(이미지 JSON)", endpoint_prefix="gemma-mm-extraction",
        max_seq_length=2048, use_qlora=True, eval_kind="extraction",
        tool_name="extract_receipt_json", tool_doc="", agent_system="",
        smoke_user="", deploy_smoke_user="",
        serve_max_model_len=2048, gen_max_tokens=768,
        has_local_serve=False,
    )


def load_courses() -> dict[str, CourseSpec]:
    """코스 레지스트리. TrackSpec 은 노트북 빌더가 쓰는 것과 **같은 객체**를 재사용한다."""
    import importlib.util

    # 정보 추출 코스의 spec은 빌더 파일에서 직접 읽습니다.
    flagship_path = os.path.join(TRACKS_ROOT, "01_extraction_to_json", "_build_notebooks.py")
    spec = importlib.util.spec_from_file_location("_bn_extraction", flagship_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    extraction = mod._flagship_spec()

    import build_all_tracks

    courses = {
        "extraction": CourseSpec(key="extraction", dir_name="01_extraction_to_json",
                                 track=extraction, train_entry="train.py"),
    }
    for ts in build_all_tracks.SPECS:
        courses[ts.key] = CourseSpec(key=ts.key, dir_name=ts.dir_name, track=ts,
                                     train_entry="train.py")
    mm = _mm_track_spec()
    courses[mm.key] = CourseSpec(key=mm.key, dir_name=mm.dir_name, track=mm,
                                 train_entry="train_mm.py", multimodal=True,
                                 has_synth=False, has_eval_stage=False)
    return courses


def load_track_data(course: CourseSpec):
    """코스별 이름으로 ``track_data.py``를 로드해 모듈 충돌을 막습니다."""
    import importlib.util

    path = os.path.join(course.dir, "track_data.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} 가 없습니다. 코스 디렉토리 구성을 확인하세요.")
    name = f"track_data_{course.key}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 2) 상태 저장소
# ===========================================================================
# 노트북이 쓰던 키 이름을 그대로 유지한다(사람이 두 경로를 오갈 때 헷갈리지 않게).
STATE_KEYS = ("bucket", "role", "model_data", "grpo_model_data", "endpoint_name")
# 실행 추적용 부가 기록이며 스테이지 판단에는 사용하지 않습니다.
BOOKKEEPING_KEYS = ("train_s3", "training_job", "grpo_job", "region", "engine", "dry_run")


class StateStore:
    """.pipeline_state/<course>.json 읽기/쓰기.

    - 파일이 없거나 깨져 있으면 빈 상태로 취급하고 경고합니다.
    - 쓰기는 tmp+rename 원자적 쓰기다. 학습 대기 중 Ctrl-C 로 끊겨도 반쯤 쓰인 JSON 이 남아
      다음 실행이 "corrupt" 로 시작하는 일을 막는다.
    """

    def __init__(self, course_key: str, state_dir: str = STATE_DIR,
                 *, dry_run: bool = False, path: str | None = None) -> None:
        # dry-run 산출물이 실제 실행에 섞이지 않도록 상태 파일을 분리합니다.
        # --state로 경로를 지정해도 dry-run 접미사는 유지합니다.
        self.course_key = course_key
        suffix = ".dryrun" if dry_run else ""
        if path:
            stem, ext = os.path.splitext(path)
            self.path = f"{stem}{suffix}{ext or '.json'}"
        else:
            self.path = os.path.join(state_dir, f"{course_key}{suffix}.json")
        self._data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        if not os.path.isfile(self.path):
            return {"course": self.course_key, "stages": {}}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("최상위가 객체가 아님")
        except (json.JSONDecodeError, ValueError, OSError) as e:  # noqa: BLE001
            print(f"WARNING: 상태 파일을 읽지 못해 빈 상태로 시작합니다: {self.path} ({e})")
            return {"course": self.course_key, "stages": {}}
        data.setdefault("course", self.course_key)
        data.setdefault("stages", {})
        return data

    # --- 조회 ---------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return bool(self._data.get(key))

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    @property
    def stages(self) -> dict[str, str]:
        return dict(self._data.get("stages") or {})

    # --- 기록 ---------------------------------------------------------------
    def set(self, **kwargs: Any) -> None:
        """값 갱신 후 즉시 저장. None 은 무시한다(스테이지가 아무것도 만들지 못한 경우)."""
        changed = False
        for k, v in kwargs.items():
            if v is None:
                continue
            if self._data.get(k) != v:
                self._data[k] = v
                changed = True
        if changed:
            self._write()

    def clear(self, *keys: str) -> None:
        for k in keys:
            self._data.pop(k, None)
        self._write()

    def mark_stage(self, stage: str) -> None:
        """스테이지 완료 시각을 기록합니다."""
        self._data.setdefault("stages", {})[stage] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write()

    def _write(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)   # 원자적 교체로 불완전한 파일을 방지합니다.

    def summary(self, *, verbose: bool = False) -> str:
        """상태를 요약하고 필요하면 부가 기록도 포함합니다."""
        lines = [f"state     : {self.path}"]
        # 첫 실행에서는 상태가 비어 있으므로 한 줄로만 안내합니다.
        filled = {k: self._data.get(k) for k in STATE_KEYS if self._data.get(k)}
        if not filled and not verbose:
            lines.append("  (첫 실행: bucket, role, 산출물은 각 스테이지가 채웁니다)")
        else:
            for k in STATE_KEYS:
                v = self._data.get(k)
                lines.append(f"  {k:16s}: {v if v else '(아직 없음)'}")
        if verbose:
            for k in BOOKKEEPING_KEYS:
                v = self._data.get(k)
                if v is not None:
                    lines.append(f"  {k:16s}: {v}")
        st = self.stages
        if st:
            lines.append("  stages          : " + ", ".join(f"{k}@{v}" for k, v in st.items()))
        return "\n".join(lines)


# ===========================================================================
# 3) 공통 헬퍼
# ===========================================================================
DRY_PREFIX = "dryrun://"   # AWS 호출 전에 걸러낼 dry-run 산출물 표시


def _is_dry_value(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(DRY_PREFIX)


def _region() -> str:
    """``common.config``에서 현재 리전을 읽습니다."""
    from common import config
    return config.AWS_REGION


@dataclass
class AwsContext:
    session: Any
    role: str
    bucket: str
    region: str


def aws_context(cfg: PipelineConfig, state: StateStore) -> AwsContext:
    """SageMaker Session + 실행 role + 버킷 해석 (00_setup 노트북과 동일한 순서).

    dry-run에서는 버킷 생성과 IAM 조회를 포함한 AWS 호출을 하지 않습니다.
    """
    from common import config

    region = config.AWS_REGION
    if cfg.runtime.dry_run:
        ctx = AwsContext(session=None, role=f"{DRY_PREFIX}role", bucket=f"{DRY_PREFIX}bucket",
                         region=region)
        state.set(region=region, dry_run=True)
        return ctx

    import boto3
    from sagemaker.core.helper.session_helper import Session

    sess = Session(boto3.Session(region_name=region))
    role = config.resolve_sagemaker_role(sess)      # 환경변수, 세션, IAM 순서로 탐색합니다.
    bucket = cfg.aws.s3_bucket or config.S3_BUCKET or sess.default_bucket()
    state.set(role=role, bucket=bucket, region=region, dry_run=False)
    return AwsContext(session=sess, role=role, bucket=bucket, region=region)


def _fold_system_messages(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    """학습 형식에 맞게 system 지시문을 첫 user 턴에 병합합니다."""
    return [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]


def invoke_chat(endpoint_name: str, messages: list[dict[str, Any]], *, region: str,
                max_tokens: int, temperature: float = 0.2) -> tuple[str, str | None]:
    """엔드포인트 응답의 텍스트와 ``finish_reason``을 반환합니다.

    OpenAI 호환 스키마의 생성 한도 키는 ``max_tokens``입니다. ``finish_reason``은 응답 절단 여부를
    판별하는 데 사용합니다.
    """
    import boto3
    from common.aws_utils import _parse_endpoint_response

    client = boto3.client("sagemaker-runtime", region_name=region)
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    resp = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(resp["Body"].read().decode("utf-8"))
    text = _parse_endpoint_response(body)
    finish = None
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish = choices[0].get("finish_reason")
    return text, finish


def warn_if_truncated(finish_reason: str | None, max_tokens: int, *, label: str = "응답") -> bool:
    """finish_reason == 'length' 면 잘린 것이다. True 를 반환(호출부가 지표에 반영할 수 있게)."""
    if finish_reason == "length":
        print(f"WARNING: {label}이 max_tokens({max_tokens})에서 잘렸습니다 "
              "(finish_reason='length'). TrackSpec.gen_max_tokens 또는 서빙 컨텍스트를 늘리세요.")
        return True
    return False


# 컨테이너 로그에 이미 포함된 타임스탬프, 레벨, 모듈을 제거합니다.
_TS_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|\s*[\w.]+\s*\|\s*")
# 학습 진행과 무관해 화면만 채우는 줄.
_LOG_NOISE = re.compile(r"huggingface\.co/api/telemetry|resolve-cache/models|HTTP Request: HEAD|urllib3|filelock")


def _tail_training_logs(job_name: str, *, region: str, token: str | None) -> tuple[list[str], str | None]:
    """학습 Job 의 CloudWatch 로그를 이어서 읽는다. (새 줄들, 다음 token) 을 반환.

    상태만 찍으면(InProgress / Training) 안에서 무슨 일이 벌어지는지 알 수 없다. 학습이 도는지
    OOM 으로 재시도 중인지, loss 가 내려가는지가 전부 이 로그에만 있다.
    로그 그룹은 Job 이 Training 단계에 들어간 뒤에 생기므로, 그 전에는 조용히 넘어간다.
    """
    import boto3
    from botocore.exceptions import ClientError

    grp = "/aws/sagemaker/TrainingJobs"
    cw = boto3.client("logs", region_name=region)
    try:
        # 스트림 이름은 <job>/algo-1-<epoch> 형태다. 단일 인스턴스 학습이라 첫 스트림만 본다.
        st = cw.describe_log_streams(logGroupName=grp, logStreamNamePrefix=job_name,
                                     limit=1).get("logStreams") or []
        if not st:
            return [], token          # Training 단계 진입 전
        kw = dict(logGroupName=grp, logStreamName=st[0]["logStreamName"], limit=40)
        # 첫 호출은 최신 40줄을 읽고 이후에는 nextToken으로 이어 읽습니다.
        r = cw.get_log_events(**kw, nextToken=token) if token else \
            cw.get_log_events(**kw, startFromHead=False)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceNotFoundException", "InvalidParameterException"):
            return [], token
        raise
    # 컨테이너 로그를 그대로 흘리면 두 가지가 읽기를 방해한다.
    #  1) tqdm 진행바: 한 메시지에 수십 프레임이 #015(=\r) 로 이어져 온다. 마지막 프레임만 남긴다.
    #  2) 컨테이너 안 로거의 자체 타임스탬프: 우리 로거가 또 시간을 붙여 두 번 찍힌다. 떼어낸다.
    out = []
    for ev in r.get("events", []):
        m = ev["message"].rstrip().replace("#015", "\r").split("\r")[-1].strip()
        if not m:
            continue
        m = _TS_PREFIX.sub("", m)        # '2026-08-02 06:56:18 | INFO | httpx | ' 제거
        if _LOG_NOISE.search(m):         # HF 텔레메트리와 config.json HEAD 요청 등
            continue
        out.append(m)
    return out, r.get("nextForwardToken") or token


def _wait_training_job(job, *, poll_seconds: int) -> str:
    """학습 잡이 종료될 때까지 폴링. 마지막 상태 문자열을 반환."""
    from common import aws_utils

    last = ""
    token: str | None = None
    tailing = False
    upload_started: float | None = None
    last_tick = -1
    while True:
        job.refresh()
        st = job.training_job_status
        secondary = getattr(job, "secondary_status", None)
        line = f"{st} / {secondary}"
        if line != last:
            log.info(f"  training job: {line}")
            last = line
            if secondary == "Uploading":
                # 컨테이너 로그가 끝난 뒤에는 S3 업로드 상태를 별도로 안내합니다.
                log.info("  (모델 아티팩트를 S3에 업로드하고 있습니다. 크기에 따라 몇 분 걸릴 수 있습니다.)")
                upload_started = time.time()

        # Training 단계부터 CloudWatch 로그를 함께 출력합니다.
        if secondary in ("Training", "Uploading") or st in ("Failed", "Stopped"):
            if not tailing:
                log.info("  CloudWatch 로그")
                tailing = True
            try:
                msgs, token = _tail_training_logs(job.training_job_name, region=_region(),
                                                  token=token)
                for m in msgs:
                    log.info(f"  │ {m[:180]}")
            except Exception as e:      # 로그를 못 읽어도 학습 대기는 계속한다
                log.warning(f"  (로그 조회 실패: {type(e).__name__}. 상태 확인은 계속합니다.)")

        # 업로드는 로그가 없으므로 경과 시간만이라도 흘려 준다.
        if secondary == "Uploading" and upload_started:
            waited = int(time.time() - upload_started)
            if waited and waited // 30 != last_tick:
                last_tick = waited // 30
                log.info(f"  업로드 대기: {waited // 60}분 {waited % 60}초")
        if st in ("Completed", "Failed", "Stopped"):
            if st != "Completed":
                # FailureReason이 비어 있을 수 있어 상세 상태도 함께 출력합니다.
                aws_utils.training_job_status(job.training_job_name, _region())
            return st
        time.sleep(poll_seconds)


def _endpoint_status(endpoint_name: str) -> str | None:
    """endpoint 상태 문자열. 없으면 None (dry-run 산출물이면 조회하지 않는다)."""
    if not endpoint_name or endpoint_name.startswith(DRY_PREFIX):
        return None
    import boto3
    from botocore.exceptions import ClientError
    try:
        d = boto3.client("sagemaker", region_name=_region()).describe_endpoint(
            EndpointName=endpoint_name)
        return d["EndpointStatus"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return None                      # 이미 지워졌다
        raise


def _resume_training_job(job_name: str, *, region: str, label: str,
                         poll_seconds: int) -> tuple[str, str | None]:
    """이전 실행의 학습 잡을 이어받고 상태와 ``model_data``를 반환합니다.

    상태 파일에 잡 이름이 있으면 새 잡을 제출하기 전에 기존 상태를 확인합니다.

    반환하는 상태:
      "InProgress": 완료까지 대기합니다.
      "Completed": 기존 산출물을 회수합니다.
      "Failed"/"Stopped": 실패 원인을 표시하고 중단합니다.
      "NotFound": 새 잡을 제출할 수 있습니다.
    """
    from common import aws_utils
    from sagemaker.core.resources import TrainingJob

    try:
        job = TrainingJob.get(job_name)
    except Exception as e:                       # 이름이 없거나 권한/리전 불일치
        print(f"[{label}] 이전 Job '{job_name}'을 조회하지 못했습니다({type(e).__name__}). 새로 제출합니다.")
        return "NotFound", None

    st = job.training_job_status
    started = getattr(job, "training_start_time", None)
    elapsed = ""
    if started:
        try:
            mins = int((time.time() - started.timestamp()) // 60)
            elapsed = f", {mins}분 경과"
        except Exception:
            pass

    if st == "InProgress":
        print(f"[{label}] 이전 실행의 Job이 아직 진행 중입니다: {job_name} "
              f"({getattr(job, 'secondary_status', '?')}{elapsed})\n"
              f"        새로 제출하지 않고 이어서 대기합니다. "
              f"멈추려면 콘솔이나 `aws sagemaker stop-training-job` 을 쓰세요.")
        _print_console_links(region, training_job=job_name)
        st = _wait_training_job(job, poll_seconds=poll_seconds)

    if st == "Completed":
        md = job.model_artifacts.s3_model_artifacts
        print(f"[{label}] 이전 Job이 완료되었습니다: {job_name}\n"
              f"        학습을 다시 돌리지 않고 산출물을 씁니다: {md}")
        return "Completed", md

    # Failed / Stopped
    print(f"[{label}] 이전 Job 상태: {st} ({job_name})")
    aws_utils.training_job_status(job_name, region)
    return st, None


def _resume_or_submit_guard(state: StateStore, key: str, *, label: str, cfg: PipelineConfig,
                            force: bool) -> str | None:
    """제출 전에 이전 Job 을 확인한다. model_data 를 얻었으면 그 값을, 아니면 None.

    진행 중인 Job은 중복 과금을 막기 위해 force=True여도 다시 제출하지 않습니다.
    종료된 Job은 force=True일 때 새로 제출합니다.
    """
    prev = state.get(key)
    if not prev or str(prev).startswith(DRY_PREFIX):
        return None

    st, md = _resume_training_job(prev, region=_region(), label=label,
                                 poll_seconds=cfg.runtime.poll_seconds)
    if st == "Completed":
        # force=True이면 완료된 산출물을 재사용하지 않고 새 Job을 제출합니다.
        if force:
            log.warning(f"[{label}] --force가 지정되어 완료된 Job 대신 새 Job을 제출합니다: {prev}")
            return None
        return md
    if st == "NotFound":
        return None
    if st == "InProgress":     # _resume_training_job 이 끝까지 기다렸으므로 여기 오면 실패로 끝난 것
        raise RuntimeError(f"이어받은 Job 이 완료되지 못했습니다: {prev}")
    if force:
        log.warning(f"[{label}] --force가 지정되어 {st} 상태의 Job 대신 새 Job을 제출합니다: {prev}")
        return None
    raise RuntimeError(
        f"이전 Job 이 {st} 상태입니다({prev}). 위 로그에서 원인을 확인하세요.\n"
        f"  같은 설정으로 다시 제출하려면 --force를 사용하거나 상태 파일에서 {key}를 지우세요.")


def _mlflow_print_ui(target: Any, mlf: Any) -> None:
    """Managed App run의 UI 딥링크를 로그에 남깁니다."""
    if not mlf.enabled:
        return
    url = mlflow_utils.ui_url(target, experiment_id=mlf.experiment_id or "",
                              run_id=mlf.run_id or "")
    if not url:
        return
    # 터미널에서는 링크로, 리다이렉트된 출력에서는 평문 URL로 남습니다.
    log.info("[mlflow] UI: %s", hyperlink(url))
    if cmd := mlflow_utils.presigned_url_command(target):
        log.info("[mlflow] 먼저 브라우저 세션을 만드세요. 세션은 최대 12시간 유지됩니다:\n    %s", cmd)


def _mlflow_container_env(cfg: PipelineConfig, state: StateStore) -> dict[str, str]:
    """학습 컨테이너에 전달할 `MLFLOW_*` 환경변수를 만듭니다."""
    target = mlflow_utils.resolve_target(cfg.mlflow.tracking_uri, local_uri=cfg.mlflow.local_uri)
    experiment = state.get("mlflow_experiment")
    if not experiment:
        return {}
    return mlflow_utils.training_env(target, experiment=experiment,
                                     parent_run_id=state.get("mlflow_run_id"))


def _submit_training_job(*, cfg: PipelineConfig, ctx: AwsContext, base_job_name: str,
                         entry_script: str, source_dir: str, hyperparameters: dict[str, Any],
                         max_runtime_hours: float,
                         input_channels: list[tuple[str, str]],
                         extra_environment: dict[str, str] | None = None) -> Any:
    """ModelTrainer를 구성하고 비동기로 제출합니다.

    input_channels: [(channel_name, s3_uri)]. 빈 리스트면 채널 없이 제출한다
                    (멀티모달 코스는 train_mm.py 가 컨테이너 안에서 시드 데이터셋을 받는다).
    """
    from common import config, dlc
    from sagemaker.core.training.configs import Compute, InputData, SourceCode, StoppingCondition
    from sagemaker.train.model_trainer import ModelTrainer

    image_uri = dlc.resolve_training_image(ctx.region)
    if not image_uri:
        raise RuntimeError(
            "학습 이미지 해석 실패: config.yaml training.dlc(repository/tag) 또는 .env 의 "
            f"DLC_IMAGE_URI 를 확인하세요(리전 포함 완전 URI). 태그 목록: {dlc.AVAILABLE_IMAGES_URL}")
    print(f"  training image: {image_uri}")

    environment = {"HF_TOKEN": config.get_hf_token()} if config.get_hf_token() else {}
    if extra_environment:
        # train.py는 MLFLOW_TRACKING_URI가 있으면 Trainer의 MLflowCallback을 사용합니다.
        environment.update({k: str(v) for k, v in extra_environment.items() if v is not None})
        shown = [k for k in extra_environment if k != "HF_TOKEN"]
        print(f"  컨테이너 환경변수 추가: {', '.join(sorted(shown))}")

    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(source_dir=source_dir, entry_script=entry_script,
                               requirements="requirements.txt"),
        compute=Compute(instance_type=cfg.training.instance_type, instance_count=1),
        hyperparameters=hyperparameters,
        environment=environment,
        role=ctx.role,
        sagemaker_session=ctx.session,
        base_job_name=base_job_name,
        # 기본값 1시간 대신 대기, 학습, 병합, 업로드를 포함할 실행 한도를 명시합니다.
        stopping_condition=StoppingCondition(max_runtime_in_seconds=int(max_runtime_hours * 3600)),
    )
    kwargs: dict[str, Any] = {"wait": False, "logs": False}   # 세션이 끊겨도 잡은 계속 실행됩니다.
    if input_channels:
        kwargs["input_data_config"] = [InputData(channel_name=name, data_source=uri)
                                       for name, uri in input_channels]
    trainer.train(**kwargs)
    job = trainer._latest_training_job
    print(f"  submitted: {job.training_job_name}")
    _print_console_links(ctx.region, training_job=job.training_job_name)
    return job


# us-east-1 온디맨드 시간당 요금(USD), 2026-08 확인:
#    pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json
#    (Training/Hosting usagetype 단가가 동일해 하나로 둔다.)
#    비용 규모를 보여 주기 위한 값이며 실제 청구액과 다를 수 있습니다. 출력은
#    항상 'us-east-1 기준'이라고 밝히고, 현재 리전이 다르면 그 사실을 함께 찍는다.
#    이 kit 이 실제로 쓰는 인스턴스만 담는다(GEMMA4_PRESETS + config.yaml 의 instance_type).
_HOURLY_USD_US_EAST_1: dict[str, float] = {
    "ml.g5.2xlarge": 1.515,     # E2B/E4B 프리셋
    "ml.g5.12xlarge": 7.09,     # 12B/26B-A4B 프리셋
    "ml.g6.2xlarge": 1.222,     # config.yaml 기본값 (L4 24GB)
    "ml.g6.4xlarge": 1.654,
    "ml.g6.12xlarge": 5.752,
    "ml.g6e.2xlarge": 2.80,     # L40S 45GB, OOM 시 상향 후보
    "ml.g6e.12xlarge": 13.12,   # 31B 프리셋
}

_cost_warning_shown = False


# CLI 화면 색상
# 로그는 logging_utils 의 _ColorFormatter 가 칠한다. 여기는 print 로 내는 '화면'
# 출력 전용이며 터미널에서만 색상을 사용합니다.
def _c(text: str, code: str) -> str:
    from common.logging_utils import _color_enabled
    return f"\033[{code}m{text}\033[0m" if _color_enabled(sys.stdout) else text


def _bold(t: str) -> str:   return _c(t, "1")
def _dim(t: str) -> str:    return _c(t, "38;5;244")
def _warn(t: str) -> str:   return _c(t, "38;5;214")
def _danger(t: str) -> str: return _c(t, "1;38;5;196")


def print_billing_preview(*, what: str, instance_type: str, region: str,
                          cap_hours: float | None = None,
                          until_deleted: bool = False) -> None:
    """과금 리소스를 만들기 직전에 종류와 참고 비용을 출력합니다."""
    global _cost_warning_shown
    from common import aws_utils

    rate = _HOURLY_USD_US_EAST_1.get(instance_type)
    print(_danger("  과금 리소스 생성") + f": {what} [{_bold(instance_type)} x1]")
    if rate is None:
        # 내부 표에 없는 인스턴스는 임의로 추정하지 않습니다.
        print("     시간당 요금: 내부 표에 없는 인스턴스입니다. "
              "https://aws.amazon.com/sagemaker-ai/pricing/ 에서 확인하세요.")
    else:
        line = f"     요금: 약 ${rate:.3f}/시간"
        if cap_hours:
            # 실행 한도 기준 최대값이며 잡이 끝나면 과금도 멈춥니다.
            line += f". {cap_hours}시간 한도까지 사용하면 최대 약 ${rate * cap_hours:.2f}"
        if until_deleted:
            line += f". 삭제할 때까지 과금되며 하루 기준 약 ${rate * 24:.2f}"
        print(line)
        print("     [us-east-1 온디맨드 목록 가격이며 실제 청구액과 다를 수 있습니다]")
        if region != "us-east-1":
            print(f"     현재 리전은 {region}이며 리전별 요금이 다를 수 있습니다.")
    if not _cost_warning_shown:
        aws_utils.print_cost_warning()
        _cost_warning_shown = True


def _print_console_links(region: str, *, training_job: str | None = None,
                         endpoint_name: str | None = None) -> None:
    """CLI에서 사용할 CloudWatch와 SageMaker 콘솔 링크를 출력합니다."""
    if training_job:
        grp = "$252Faws$252Fsagemaker$252FTrainingJobs"
        print(f"    console: https://{region}.console.aws.amazon.com/sagemaker/home"
              f"?region={region}#/jobs/{training_job}")
        print(f"    logs   : https://{region}.console.aws.amazon.com/cloudwatch/home"
              f"?region={region}#logsV2:log-groups/log-group/{grp}"
              f"$3FlogStreamNameFilter$3D{training_job}")
        print("    (로그는 Job이 Training 단계에 들어간 뒤 생성됩니다.)")
    if endpoint_name:
        grp = "$252Faws$252Fsagemaker$252FEndpoints"
        print(f"    console: https://{region}.console.aws.amazon.com/sagemaker/home"
              f"?region={region}#/endpoints/{endpoint_name}")
        print(f"    logs   : https://{region}.console.aws.amazon.com/cloudwatch/home"
              f"?region={region}#logsV2:log-groups/log-group/{grp}$252F{endpoint_name}")


def _eta_guard(*, n_samples: int, epochs: float, accum: int, seconds_per_step: float,
               max_runtime_hours: float) -> None:
    """제출 전에 '이 설정이 한도 안에 끝나는지' 계산해 세운다(노트북 §3 의 assert 를 포팅).

    step = ceil(건수/accum) x epochs. 실측 g6.2xlarge: seq2048 ≈ 17s/step, seq512 ≈ 7s/step.
    한도에는 대기, 이미지 가져오기, 병합, 업로드가 포함됩니다.
    """
    steps = math.ceil(n_samples / max(1, accum)) * epochs
    eta_min = steps * seconds_per_step / 60
    print(f"  학습 {n_samples}건 x {epochs} epoch, 약 {steps:.0f} step: "
          f"학습 {eta_min:.0f}분 + 병합과 업로드 약 5분 (한도 {max_runtime_hours}시간)")
    if eta_min / 60 >= max_runtime_hours:
        raise RuntimeError(
            f"예상 학습 시간({eta_min:.0f}분)이 training.max_runtime_hours"
            f"({max_runtime_hours}시간)에 육박합니다. config.yaml 의 training.max_train_samples/"
            "epochs 를 줄이거나 max_runtime_hours 를 올리세요 "
            "(병합과 업로드용으로 최소 15분은 남겨 두세요).")


def _counts(cfg: PipelineConfig) -> tuple[int, int, int]:
    """시드, 합성, 평가 건수를 반환합니다."""
    if cfg.runtime.dry_run:
        return (cfg.data.dry_run_seed_samples, cfg.data.dry_run_synthetic,
                cfg.evaluation.dry_run_num_examples)
    return cfg.data.num_seed_samples, cfg.data.num_synthetic, cfg.evaluation.num_examples


def data_dir(course: CourseSpec, cfg: PipelineConfig) -> str:
    """dry-run과 실제 실행의 데이터 디렉토리를 분리해 반환합니다."""
    base = course.data_dir
    return os.path.join(base, "dryrun") if cfg.runtime.dry_run else base


# ===========================================================================
# 4) 스테이지
# ===========================================================================
def stage_data(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """시드 로드 + grounded 합성 + train/eval JSONL 작성 + S3 업로드.

    반환: {"train_path", "eval_path", "train_s3"}
    """
    td = load_track_data(course)
    n_seed, n_synth, n_eval = _counts(cfg)

    if course.multimodal:
        # 멀티모달 데이터는 학습 컨테이너에서 직접 읽으므로 로드 여부만 확인합니다.
        log.info(f"[data] {course.key}: 멀티모달 학습 데이터는 컨테이너에서 불러옵니다.")
        seeds = td.load_seed_examples(2, token=_hf_token())
        print(f"  시드 확인: {len(seeds)}건 (images/messages 컬럼)")
        state.mark_stage("data")
        return {"train_path": None, "eval_path": None, "train_s3": None}

    ddir = data_dir(course, cfg)
    os.makedirs(ddir, exist_ok=True)
    train_path = os.path.join(ddir, "train.jsonl")
    eval_path = os.path.join(ddir, "eval.jsonl")

    if os.path.isfile(train_path) and not force:
        n_lines = sum(1 for _ in open(train_path, encoding="utf-8"))
        log.info(f"[data] 건너뜀: {train_path}에 {n_lines}건이 있습니다. 다시 만들려면 --force를 사용하세요.")
    else:
        log.info(f"[data] {course.key}: 시드 {n_seed}건 + 합성 {n_synth}건")
        seeds = td.load_seed_examples(n_seed, token=_hf_token())
        print(f"  파싱한 시드: {len(seeds)}건")

        synth_msgs: list[list[dict[str, str]]] = []
        if course.has_synth and n_synth > 0:
            synth_msgs = _generate_synthetic(td, cfg, n_synth)

        rows = [td.to_messages(s) for s in seeds] + synth_msgs
        _write_messages_jsonl(rows, train_path)
        print(f"  학습 데이터: {len(rows)}건, {train_path}")

        # held-out은 학습 구간 뒤에서 선택해 데이터 누출을 막습니다.
        pool = td.load_seed_examples(n_seed + n_eval, token=_hf_token())
        heldout = pool[n_seed:n_seed + n_eval]
        if not heldout:
            print(f"  시드가 {len(pool)}건뿐이라 학습 구간 {n_seed}건 뒤에 평가 데이터가 없습니다. "
                  "data.num_seed_samples를 줄이거나 더 큰 데이터셋을 사용하세요.")
        else:
            with open(eval_path, "w", encoding="utf-8") as f:
                for ex in heldout:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  held-out {len(heldout)}건: {eval_path} "
                  f"(시드 인덱스 {n_seed}~{n_seed + len(heldout) - 1}, 학습 구간 제외)")

    if cfg.runtime.dry_run:
        train_s3 = f"{DRY_PREFIX}{cfg.data.s3_prefix}/{course.key}/data/train.jsonl"
        log.info(f"[data] dry-run: S3 업로드 생략 ({train_s3})")
    else:
        from common import aws_utils
        ctx = aws_context(cfg, state)
        key = f"{cfg.data.s3_prefix}/{course.key}/data/{os.path.basename(train_path)}"
        # 내용이 바뀐 경우에만 업로드합니다.
        train_s3 = aws_utils.upload_if_changed(train_path, ctx.bucket, key, region=ctx.region)

    state.set(train_s3=train_s3)
    state.mark_stage("data")
    return {"train_path": train_path, "eval_path": eval_path, "train_s3": train_s3}


def _hf_token() -> str | None:
    """호출 시점의 ``HF_HOME``과 토큰 설정을 반영합니다."""
    from common import config
    return config.get_hf_token()


def _write_messages_jsonl(rows: list[list[dict[str, str]]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for msgs in rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def _generate_synthetic(td: Any, cfg: PipelineConfig, n_synth: int) -> list[list[dict[str, str]]]:
    """Bedrock Converse + critique/refine 로 grounded 합성. 실패는 치명적으로 다루지 않는다."""
    from common import config
    from common.synth import bedrock_synth as bs

    # dry-run에서는 Bedrock을 호출하지 않고 시드를 복제해 데이터 형식만 검증합니다.
    if cfg.runtime.dry_run:
        n_seed = cfg.data.dry_run_seed_samples
        seeds = td.load_seed_examples(n_seed, token=_hf_token())
        rows = [td.to_messages(s) for s in seeds]
        if not rows:
            return []
        stub = [rows[i % len(rows)] for i in range(n_synth)]
        print(f"  dry-run: Bedrock 합성을 생략하고 시드 {len(rows)}건을 "
              f"{n_synth}건으로 복제해 형식만 확인합니다.")
        return stub

    model_id = cfg.aws.bedrock_model_id or config.BEDROCK_CLAUDE_MODEL_ID
    if not model_id or "claude" not in model_id:
        raise RuntimeError(
            f"aws.bedrock_model_id = {model_id!r}: inference-profile ID 여야 합니다"
            "(예: global.anthropic.claude-sonnet-5). 콘솔에서 현행 ID 를 확인하세요.")
    n_seed = cfg.data.dry_run_seed_samples if cfg.runtime.dry_run else cfg.data.num_seed_samples
    seeds = td.load_seed_examples(n_seed, token=_hf_token())
    examples = bs.generate_grounded(
        task_instruction=td.TASK_INSTRUCTION,
        seed_texts=td.seed_texts_for_synth(seeds),
        n_total=n_synth,
        model_id=model_id,
        region=config.BEDROCK_REGION,
        to_messages=td.to_messages,
        max_batches=3 if cfg.runtime.dry_run else None,
        max_workers=cfg.data.synth_max_workers,   # throttling(429) 이 나면 낮춘다
    )
    print(f"  채택한 합성 데이터: {len(examples)}건")
    return [ex.messages for ex in examples]


def stage_train(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                *, force: bool = False) -> dict[str, Any]:
    """SFT 학습 잡을 제출하고 S3 아티팩트를 반환합니다."""
    if state.get("model_data") and not force:
        log.info(f"[train] 건너뜀: model_data가 있습니다: {state.get('model_data')}\n"
              "        다시 학습하려면 --force (한 시간짜리 학습을 다시 돌립니다).")
        return {"model_data": state.get("model_data")}

    tr = cfg.training
    hyperparameters: dict[str, Any] = {
        "model_id": _model_id(),
        "epochs": tr.epochs,
        "per_device_train_batch_size": tr.per_device_train_batch_size,
        "gradient_accumulation_steps": tr.gradient_accumulation_steps,
        "learning_rate": tr.learning_rate,
        "max_seq_length": course.track.max_seq_length,
        "lora_r": tr.lora.r, "lora_alpha": tr.lora.alpha, "lora_dropout": tr.lora.dropout,
        "use_qlora": tr.use_qlora,
        # merge_adapter=False 면 아티팩트에 adapter/ 만 남아 배포가 불가능하다.
        "merge_adapter": tr.merge_adapter,
    }
    channels: list[tuple[str, str]] = []
    if course.multimodal:
        from common import config
        hyperparameters["seed_dataset"] = config.TRACKS[course.key].seed_dataset
        hyperparameters["freeze_vision"] = True   # vision tower 는 얼리고 language LoRA 만 학습
        n_samples = tr.max_train_samples or 0
    else:
        train_s3 = state.get("train_s3")
        train_path = os.path.join(data_dir(course, cfg), "train.jsonl")
        if not train_s3:
            raise RuntimeError(_missing(course, "train_s3", "data", state))
        n_lines = (sum(1 for _ in open(train_path, encoding="utf-8"))
                   if os.path.isfile(train_path) else 0)
        # max_train_samples 는 '앞에서부터 N건'이라 파일이 더 짧으면 파일 건수가 실제 학습량이다.
        # min 을 취하지 않으면 ETA 가 과대 추정돼 '한도 초과' 경고가 헛되게 뜬다.
        n_samples = min(tr.max_train_samples, n_lines) if tr.max_train_samples else n_lines
        channels = [("train", train_s3)]
    if tr.max_train_samples:
        hyperparameters["max_train_samples"] = tr.max_train_samples

    if n_samples:
        _eta_guard(n_samples=n_samples, epochs=tr.epochs,
                   accum=tr.gradient_accumulation_steps,
                   seconds_per_step=tr.seconds_per_step,
                   max_runtime_hours=tr.max_runtime_hours)

    # dry-run에서도 생성 예정 리소스와 참고 비용을 출력합니다.
    print_billing_preview(
        what=f"Training Job ({course.train_job_prefix}-*)",
        instance_type=tr.instance_type, region=_region(),
        cap_hours=tr.max_runtime_hours)

    if cfg.runtime.dry_run:
        log.info(f"[train] dry-run: 학습 Job을 제출하지 않습니다.\n"
              f"        entry={course.train_entry} source_dir={course.scripts_dir}\n"
              f"        instance={tr.instance_type} runtime_limit={tr.max_runtime_hours}h "
              f"channels={[c[0] for c in channels] or '(없음)'}\n"
              f"        hyperparameters={json.dumps(hyperparameters, ensure_ascii=False)}")
        model_data = f"{DRY_PREFIX}{course.train_job_prefix}/model.tar.gz"
        state.set(model_data=model_data)
        state.mark_stage("train")
        return {"model_data": model_data}

    ctx = aws_context(cfg, state)

    # 중복 제출을 막기 위해 이전 실행의 잡을 먼저 확인합니다.
    resumed = _resume_or_submit_guard(state, "training_job", label="train", cfg=cfg, force=force)
    if resumed:
        state.set(model_data=resumed)
        state.mark_stage("train")
        return {"model_data": resumed}

    log.info(f"[train] {course.key}: {course.train_entry}, {tr.instance_type}")
    job = _submit_training_job(
        cfg=cfg, ctx=ctx, base_job_name=course.train_job_prefix,
        entry_script=course.train_entry, source_dir=course.scripts_dir,
        hyperparameters=hyperparameters, max_runtime_hours=tr.max_runtime_hours,
        input_channels=channels,
        extra_environment=_mlflow_container_env(cfg, state))
    state.set(training_job=job.training_job_name)

    status = _wait_training_job(job, poll_seconds=cfg.runtime.poll_seconds)
    if status != "Completed":
        raise RuntimeError(
            f"학습 잡이 {status} 상태입니다 ({job.training_job_name}). CloudWatch 로그를 확인하세요.\n"
            "  Stopped 이고 FailureReason 이 비어 있으면 MaxRuntimeExceeded 일 수 있습니다. "
            "config.yaml 의 training.max_runtime_hours 를 올리세요.")
    model_data = job.model_artifacts.s3_model_artifacts
    log.info(f"[train] 완료: model_data={model_data}")
    state.set(model_data=model_data)
    state.mark_stage("train")
    return {"model_data": model_data}


def _model_id() -> str:
    from common import config
    return config.DEFAULT_MODEL_ID


def stage_grpo(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """SFT 산출물에서 GRPO 정련을 이어서 실행합니다.

    프로그램으로 보상을 계산할 수 있는 추출과 분류 코스만 지원합니다.
    """
    kind = course.grpo_reward_kind
    if not kind:
        # 보통은 resolve_stages 가 먼저 막는다(명시 요청은 거부, 'all' 은 계획에서 제외).
        # 스테이지 함수를 직접 호출하는 경우에도 같은 검증을 적용합니다.
        raise RuntimeError(unsupported_reason(course, "grpo") or "grpo 미지원")
    if state.get("grpo_model_data") and not force:
        log.info(f"[grpo] 건너뜀: grpo_model_data가 있습니다: {state.get('grpo_model_data')}\n"
              "       다시 돌리려면 --force.")
        return {"grpo_model_data": state.get("grpo_model_data")}

    sft_model_data = state.get("model_data")
    if not sft_model_data:
        raise RuntimeError(_missing(course, "model_data", "train", state))

    # Bedrock 합성 전에 기존 GRPO 잡을 확인해 중복 비용을 막습니다.
    if not cfg.runtime.dry_run:
        resumed = _resume_or_submit_guard(state, "grpo_job", label="grpo", cfg=cfg, force=force)
        if resumed:
            state.set(grpo_model_data=resumed, model_data=resumed)
            state.mark_stage("grpo")
            return {"grpo_model_data": resumed}

    g = cfg.grpo
    grpo_path = os.path.join(data_dir(course, cfg), "grpo_train.jsonl")

    from common import grpo_data as gd

    # 기존 프롬프트 파일을 재사용해 불필요한 Bedrock 호출을 막습니다.
    if os.path.isfile(grpo_path) and not force:
        n_prompts = sum(1 for _ in open(grpo_path, encoding="utf-8"))
        log.info(f"[grpo] 프롬프트 재사용: {grpo_path} ({n_prompts}건). "
                 "다시 만들려면 --force.")
    else:
        rows = _grpo_prompts(course, cfg, state, kind)
        gd.describe(rows, source=g.prompt_source)
        gd.write_grpo_jsonl(rows, grpo_path)   # 0건이면 여기서 즉시 실패(빈 파일 업로드 방지)
        n_prompts = len(rows)

    hyperparameters: dict[str, Any] = {
        # model_id 는 멀티모달 감지 폴백용. 실제 base 는 'model' 채널(SFT 산출물)에서 로드한다.
        "model_id": _model_id(),
        "reward_kind": kind,
        "epochs": g.epochs,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": g.learning_rate,
        "num_generations": g.num_generations,
        "max_completion_length": g.max_completion_length,
        "max_seq_length": course.track.max_seq_length,
        "lora_r": cfg.training.lora.r, "lora_alpha": cfg.training.lora.alpha,
        "lora_dropout": cfg.training.lora.dropout,
        "use_qlora": cfg.training.use_qlora, "merge_adapter": cfg.training.merge_adapter,
    }

    print_billing_preview(
        what=f"GRPO Training Job ({course.grpo_job_prefix}-*)",
        instance_type=cfg.training.instance_type, region=_region(),
        cap_hours=g.max_runtime_hours)

    if cfg.runtime.dry_run:
        log.info(f"[grpo] dry-run: 학습 Job을 제출하지 않습니다 (프롬프트 {n_prompts}건, "
              f"source={g.prompt_source}, runtime_limit={g.max_runtime_hours}h)")
        grpo_model_data = f"{DRY_PREFIX}{course.grpo_job_prefix}/model.tar.gz"
        state.set(grpo_model_data=grpo_model_data, model_data=grpo_model_data)
        state.mark_stage("grpo")
        return {"grpo_model_data": grpo_model_data}

    from common import aws_utils
    ctx = aws_context(cfg, state)
    key = f"{cfg.data.s3_prefix}/{course.key}/grpo/train.jsonl"
    train_s3 = aws_utils.upload_if_changed(grpo_path, ctx.bucket, key, region=ctx.region)

    # 학습 잡과 모델 아티팩트의 리전을 맞춥니다.
    sft_model_data = aws_utils.ensure_model_data_in_region(
        sft_model_data, ctx.region, job_prefix=course.train_job_prefix)

    log.info(f"[grpo] {course.key}: reward_kind={kind}, base={sft_model_data}")
    job = _submit_training_job(
        cfg=cfg, ctx=ctx, base_job_name=course.grpo_job_prefix,
        entry_script="train_grpo.py", source_dir=course.scripts_dir,
        hyperparameters=hyperparameters, max_runtime_hours=g.max_runtime_hours,
        extra_environment=_mlflow_container_env(cfg, state),
        # SFT 산출물은 model 채널을 통해 GRPO 학습의 base로 전달합니다.
        input_channels=[("train", train_s3), ("model", sft_model_data)])
    state.set(grpo_job=job.training_job_name)

    status = _wait_training_job(job, poll_seconds=cfg.runtime.poll_seconds)
    if status != "Completed":
        raise RuntimeError(f"GRPO 잡이 {status} 상태입니다 ({job.training_job_name}). "
                           "CloudWatch 로그를 확인하세요.")
    grpo_model_data = job.model_artifacts.s3_model_artifacts
    log.info(f"[grpo] 완료: grpo_model_data={grpo_model_data}")
    # 배포 대상은 GRPO 결과로 갱신하고 비교용 아티팩트도 따로 보관합니다.
    state.set(grpo_model_data=grpo_model_data, model_data=grpo_model_data)
    state.mark_stage("grpo")
    return {"grpo_model_data": grpo_model_data, "model_data": grpo_model_data}


def _grpo_prompts(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                  kind: str) -> list[dict]:
    """GRPO 프롬프트를 합성, holdout, 실패 사례에서 준비합니다.

    SFT 데이터를 그대로 쓰면 보상 차이가 사라져 학습 신호가 약해집니다.
    """
    from common import config, grpo_data as gd

    td = load_track_data(course)
    g = cfg.grpo
    n = g.num_prompts
    n_seed = cfg.data.dry_run_seed_samples if cfg.runtime.dry_run else cfg.data.num_seed_samples

    # dry-run에서는 synth 요청도 holdout으로 대체합니다.
    if cfg.runtime.dry_run and g.prompt_source == "synth":
        print("  dry-run: GRPO 합성을 생략하고 holdout을 사용합니다.")
        return gd.from_holdout(os.path.join(data_dir(course, cfg), "train.jsonl"), n, sft_used=n_seed)

    if g.prompt_source == "holdout":
        return gd.from_holdout(os.path.join(data_dir(course, cfg), "train.jsonl"), n, sft_used=n_seed)
    if g.prompt_source == "synth":
        # SFT에서 사용하지 않은 시드 구간으로 합성 분포의 중복을 줄입니다.
        pool = td.load_seed_examples(n_seed + n, token=_hf_token())
        fresh = pool[n_seed:]
        return gd.from_synth(task_instruction=td.TASK_INSTRUCTION,
                             seed_texts=td.seed_texts_for_synth(fresh), n=n,
                             model_id=cfg.aws.bedrock_model_id or config.BEDROCK_CLAUDE_MODEL_ID,
                             region=config.BEDROCK_REGION, to_messages=td.to_messages,
                             kind=kind)   # 난이도 제약을 생성 프롬프트에만 적용
    # failures 소스에는 평가 결과가 필요합니다.
    preds_path = _eval_preds_path(course, cfg)
    if not os.path.isfile(preds_path):
        raise RuntimeError(
            f"grpo.prompt_source='failures'에는 평가 예측이 필요합니다: {preds_path}\n"
            f"  실행: {_cmd(course.key, 'eval')}\n"
            "  (노트북은 같은 커널의 preds 변수를 썼지만, CLI 는 eval 스테이지가 남긴 파일을 읽습니다.)")
    with open(preds_path, encoding="utf-8") as f:
        saved = json.load(f)
    return gd.from_failures(saved["heldout"], saved["preds"], kind=kind,
                            to_messages=td.to_messages, max_n=n)


def stage_deploy(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                 *, force: bool = False) -> dict[str, Any]:
    """ModelBuilder로 실시간 엔드포인트를 배포하고 응답을 확인합니다."""
    prev_ep = state.get("endpoint_name")
    if prev_ep and not force:
        # 실제 상태를 확인해 중복 엔드포인트 생성을 막습니다.
        st = _endpoint_status(prev_ep)
        if st == "InService":
            log.info(f"[deploy] 건너뜀: 엔드포인트가 InService 상태입니다: {prev_ep}")
            return {"endpoint_name": prev_ep}
        if st == "Creating":
            log.info(f"[deploy] 이전 실행의 엔드포인트가 생성 중입니다: {prev_ep}\n"
                     "         새로 만들지 않고 InService까지 기다립니다.")
            _wait_endpoint(prev_ep, poll_seconds=cfg.runtime.poll_seconds)
            return {"endpoint_name": prev_ep}
        if st in ("Failed", "RollingBack", "Deleting", "OutOfService"):
            raise RuntimeError(
                f"이전 endpoint 가 {st} 상태입니다({prev_ep}).\n"
                "  CloudWatch 로그로 원인을 확인하고, 정리하려면 --stages cleanup 을 먼저 실행하세요.")
        if st is None:
            log.info(f"[deploy] 상태 파일의 엔드포인트가 없어 새로 만듭니다: {prev_ep}")
        else:
            log.info(f"[deploy] 건너뜀: 엔드포인트가 있습니다: {prev_ep} ({st})\n"
                     "         새로 만들려면 기존 엔드포인트를 정리한 뒤 --force를 사용하세요.")
            return {"endpoint_name": prev_ep}

    model_data = state.get("model_data")
    if not model_data:
        raise RuntimeError(_missing(course, "model_data", "train", state))

    engine = cfg.serving.engine
    endpoint_name = f"{course.endpoint_prefix}-{engine}-{int(time.time())}"

    from common import config, dlc

    serve_image = dlc.resolve_serving_image(_region(), engine)
    if not serve_image:
        raise RuntimeError(f"{engine} 서빙 이미지 해석 실패: config.yaml serving.images 또는 "
                           f".env 의 *_IMAGE_URI 를 확인하세요: {dlc.AVAILABLE_IMAGES_URL}")

    # 엔진별 환경변수 이름은 dlc.serving_env()에서 통합 관리합니다.
    serve_env = dlc.serving_env(
        engine,
        max_model_len=course.serve_max_model_len,
        max_num_seqs=cfg.serving.max_num_seqs,
        # 설정에 적은 문자열 표현을 그대로 전달합니다.
        gpu_memory_utilization=cfg.serving.gpu_memory_utilization,
        tensor_parallel=cfg.serving.tensor_parallel,
        # 멀티모달 코스는 이미지 입력을 허용한다. 텍스트 코스는 지정하지 않는다
        # (텍스트 전용 re-export 모델은 그냥 텍스트로 서빙된다).
        mm_limit=json.dumps({"image": 1}) if course.multimodal else None,
        hf_token=config.get_serving_hf_token(),   # gated 모델일 때만 채워진다
    )

    log.info(f"[deploy] {course.key}: engine={engine} image={serve_image}")
    print(f"         instance={cfg.serving.instance_type} max_model_len={course.serve_max_model_len}")
    print(f"         serve_env={serve_env}")

    # 엔드포인트는 삭제할 때까지 과금되므로 종료 시점 대신 삭제 필요성을 안내합니다.
    print_billing_preview(
        what=f"real-time endpoint ({endpoint_name})",
        instance_type=cfg.serving.instance_type, region=_region(),
        until_deleted=True)

    if cfg.runtime.dry_run:
        log.info("[deploy] dry-run: 엔드포인트를 만들지 않습니다.")
        endpoint_name = f"{DRY_PREFIX}{course.endpoint_prefix}-{engine}"
        state.set(endpoint_name=endpoint_name, engine=engine)
        state.mark_stage("deploy")
        return {"endpoint_name": endpoint_name}

    from common import aws_utils
    from sagemaker.serve import ModelBuilder

    ctx = aws_context(cfg, state)
    # 상태 파일의 모델 아티팩트와 현재 리전을 맞춥니다.
    model_data = aws_utils.ensure_model_data_in_region(
        model_data, ctx.region, job_prefix=course.train_job_prefix)
    state.set(model_data=model_data)

    mb = ModelBuilder(
        image_uri=serve_image,
        s3_model_data_url=model_data,      # v3: model_path 는 로컬 경로이므로 쓰지 않는다
        env_vars=serve_env,
        role_arn=ctx.role,
        sagemaker_session=ctx.session,
        instance_type=cfg.serving.instance_type,
    )
    mb.build()
    mb.deploy(endpoint_name=endpoint_name, initial_instance_count=1,
              instance_type=cfg.serving.instance_type, wait=False)  # 세션이 끊겨도 배포는 계속됩니다.
    state.set(endpoint_name=endpoint_name, engine=engine)
    log.info(f"[deploy] 생성 중: {endpoint_name}")
    _print_console_links(ctx.region, endpoint_name=endpoint_name)

    _wait_endpoint(endpoint_name, poll_seconds=cfg.runtime.poll_seconds)
    _deploy_smoke(course, cfg, endpoint_name)
    state.mark_stage("deploy")
    return {"endpoint_name": endpoint_name}


def _wait_endpoint(endpoint_name: str, *, poll_seconds: int) -> None:
    """엔드포인트가 InService 상태가 될 때까지 기다립니다."""
    from sagemaker.core.resources import Endpoint

    ep = Endpoint.get(endpoint_name)
    last = ""
    while True:
        ep.refresh()
        st = ep.endpoint_status
        if st != last:
            print(f"  endpoint: {st}")
            last = st
        if st == "InService":
            return
        if st in ("Failed", "OutOfService", "RollingBack"):
            raise RuntimeError(
                f"endpoint 가 {st} 입니다 ({endpoint_name}). "
                f"FailureReason: {getattr(ep, 'failure_reason', None)}\n"
                "  'did not pass the ping health check'만 보이면 CUDA OOM일 수 있습니다. "
                "실제 torch.OutOfMemoryError 는 CloudWatch endpoint 로그에만 남습니다. "
                "config.yaml 의 serving.max_num_seqs 를 낮추거나 더 큰 GPU 인스턴스를 쓰세요.")
        time.sleep(poll_seconds)


def _deploy_smoke(course: CourseSpec, cfg: PipelineConfig, endpoint_name: str) -> None:
    """배포 직후 응답과 절단 여부를 확인합니다."""
    td = load_track_data(course)
    region = _region()
    if course.multimodal:
        # 저장소의 샘플 영수증을 사용해 데이터셋 다운로드 없이 확인합니다.
        import base64
        import io

        sample = td.load_sample_receipts()[0]
        buf = io.BytesIO()
        sample["image"].save(buf, format="JPEG", quality=85)   # payload 가 PNG 의 1/8
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": td.INSTRUCTION}]}]
        label = f"멀티모달 스모크({sample['name']})"
    else:
        messages = _fold_system_messages(td.SYSTEM_PROMPT, course.track.deploy_smoke_user)
        label = "배포 스모크"

    max_tokens = course.gen_max_tokens
    out, finish = invoke_chat(endpoint_name, messages, region=region,
                             max_tokens=max_tokens, temperature=0.1)
    print(f"  {label}: finish_reason={finish}")
    print("  ---\n  " + (out or "(빈 응답)").strip().replace("\n", "\n  ")[:600])
    warn_if_truncated(finish, max_tokens, label=label)
    if not out:
        raise RuntimeError("빈 응답입니다. CloudWatch endpoint 로그를 확인하세요.")


def stage_eval(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """held-out 데이터로 코스별 지표를 계산합니다.

    학습에 사용하지 않은 시드 구간만 평가에 사용합니다.
    """
    if not course.has_eval_stage:
        # 평가 스테이지가 없는 코스는 성공으로 건너뜁니다.
        log.info("[eval] " + (unsupported_reason(course, "eval") or "평가를 지원하지 않습니다"))
        return {}

    endpoint_name = state.get("endpoint_name")
    if not endpoint_name:
        raise RuntimeError(_missing(course, "endpoint_name", "deploy", state))

    _, _, n_eval = _counts(cfg)
    td = load_track_data(course)
    heldout = _load_heldout(course, cfg, td, n_eval)
    log.info(f"[eval] {course.key}: held-out {len(heldout)}건, endpoint={endpoint_name}")

    if cfg.runtime.dry_run or _is_dry_value(endpoint_name):
        log.info("[eval] dry-run: 엔드포인트를 호출하지 않습니다.")
        state.mark_stage("eval")
        return {"n": len(heldout), "dry_run": True}

    region = _region()
    max_tokens = course.gen_max_tokens

    def predict(ex: dict) -> tuple[str, str | None]:
        # 모든 서빙 엔진이 messages를 받아 서버에서 chat template을 적용합니다.
        msgs = _fold_system_messages(td.SYSTEM_PROMPT, ex["input"])
        return invoke_chat(endpoint_name, msgs, region=region,
                          max_tokens=max_tokens, temperature=0.0)  # 재현성 위해 결정론적

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=cfg.evaluation.workers) as pool:
        # map은 입력 순서를 보존하므로 heldout과 예측의 인덱스가 일치합니다.
        results = list(pool.map(predict, heldout))
    preds = [r[0] for r in results]
    truncated = sum(1 for r in results if r[1] == "length")
    print(f"  예측: {len(preds)}건 (workers={cfg.evaluation.workers})")
    if truncated:
        # finish_reason을 집계해 낮은 지표의 원인을 구분합니다.
        print(f"  WARNING: {truncated}/{len(preds)}건이 max_tokens({max_tokens})에서 잘렸습니다. "
              "TrackSpec.gen_max_tokens를 올리세요.")

    scores = _score(course, cfg, td, heldout, preds)
    scores["truncated"] = truncated
    log.info("[eval] scores: " + json.dumps(scores, ensure_ascii=False, sort_keys=True))

    # GRPO 'failures' 소스가 읽을 수 있게 남긴다(노트북은 같은 커널의 변수를 썼다).
    with open(_eval_preds_path(course, cfg), "w", encoding="utf-8") as f:
        json.dump({"endpoint_name": endpoint_name, "heldout": heldout, "preds": preds,
                   "scores": scores}, f, ensure_ascii=False, indent=2)
    print(f"  예측 저장: {_eval_preds_path(course, cfg)} (grpo.prompt_source='failures'에서 사용)")
    state.mark_stage("eval")
    return scores


def _eval_preds_path(course: CourseSpec, cfg: PipelineConfig) -> str:
    return os.path.join(data_dir(course, cfg), "eval_preds.json")


def _load_heldout(course: CourseSpec, cfg: PipelineConfig, td: Any, n_eval: int) -> list[dict]:
    """eval.jsonl 이 있으면 그것을, 없으면 시드에서 학습 구간을 건너뛰고 잘라 쓴다."""
    eval_path = os.path.join(data_dir(course, cfg), "eval.jsonl")
    if os.path.isfile(eval_path):
        rows = [json.loads(line) for line in open(eval_path, encoding="utf-8") if line.strip()]
        if rows:
            return rows[:n_eval]
    n_seed = cfg.data.dry_run_seed_samples if cfg.runtime.dry_run else cfg.data.num_seed_samples
    pool = td.load_seed_examples(n_seed + n_eval, token=_hf_token())
    heldout = pool[n_seed:n_seed + n_eval]
    if not heldout:
        raise RuntimeError(
            f"시드가 {len(pool)}건뿐이라 학습 구간({n_seed}건) 뒤에 남는 예시가 없습니다. "
            "config.yaml 의 data.num_seed_samples 를 줄이거나 더 큰 시드를 쓰세요.")
    return heldout


def _score(course: CourseSpec, cfg: PipelineConfig, td: Any,
           heldout: list[dict], preds: list[str]) -> dict[str, Any]:
    """``common.eval_utils``로 코스별 지표를 계산합니다."""
    from common import config, eval_utils

    kind = course.track.eval_kind
    if kind == "extraction":
        # gold = 파싱된 {'name','arguments'} dict
        pairs = [(pred, json.loads(ex["output"])) for pred, ex in zip(preds, heldout)]
        return dict(eval_utils.eval_extraction(pairs))
    if kind == "classification":
        # 데이터 어댑터에서 현재 데이터셋의 라벨 이름을 읽습니다.
        label_names = td.load_label_names(token=_hf_token())
        pairs = [(pred, ex["output"]) for pred, ex in zip(preds, heldout)]
        return dict(eval_utils.eval_classification(pairs, label_names))

    # 요약과 도메인 QA는 ROUGE와 선택적 Bedrock 평가를 사용합니다.
    scores: dict[str, Any] = dict(eval_utils.eval_rouge(
        [(pred, ex["output"]) for pred, ex in zip(preds, heldout)]))
    limit = cfg.evaluation.judge_max_examples
    if limit <= 0:
        print("  (evaluation.judge_max_examples=0: LLM judge 생략)")
        return scores
    axes = (["groundedness", "coverage"] if kind == "summarization"
            else ["correctness", "helpfulness", "groundedness"])
    rubric = ("Rate the summary vs the source document." if kind == "summarization"
              else "Rate the answer for correctness, helpfulness, and (if context present) groundedness.")
    judged = [
        eval_utils.llm_judge(
            model_id=cfg.aws.bedrock_model_id or config.BEDROCK_CLAUDE_MODEL_ID,
            region=config.BEDROCK_REGION, source=ex["input"], prediction=pred,
            reference=ex["output"], rubric=rubric, axes=axes)
        for pred, ex in list(zip(preds, heldout))[:limit]
    ]
    scores.update(eval_utils.aggregate_judge(judged, axes))
    return scores


def stage_cleanup(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                  *, force: bool = False) -> dict[str, Any]:
    """엔드포인트, 엔드포인트 구성, 모델을 참조 순서에 맞춰 삭제합니다."""
    endpoint_name = state.get("endpoint_name")
    if not endpoint_name:
        log.info("[cleanup] 상태 파일에 삭제할 엔드포인트가 없습니다.")
        print(f"          prefix '{course.endpoint_prefix}'로 남은 리소스를 확인하세요. "
              "--force를 사용하면 해당 prefix를 함께 정리합니다.")
        if not force:
            return {}
    if cfg.runtime.dry_run or _is_dry_value(endpoint_name):
        log.info(f"[cleanup] dry-run: 삭제할 실제 리소스가 없습니다 ({endpoint_name}).")
        state.clear("endpoint_name")
        state.mark_stage("cleanup")
        return {}

    import boto3
    sm = boto3.client("sagemaker", region_name=_region())
    deleted: list[str] = []

    targets: list[str] = [endpoint_name] if endpoint_name else []
    if force:
        # 상태 파일에 없는 이전 배포 리소스도 prefix로 찾아 정리합니다.
        listed = [e["EndpointName"]
                  for e in sm.list_endpoints(NameContains=course.endpoint_prefix)["Endpoints"]]
        targets += [n for n in listed if n not in targets]
        log.info(f"[cleanup] --force: prefix '{course.endpoint_prefix}'로 찾은 엔드포인트: {listed}")

    for name in targets:
        model_names: list[str] = []
        try:
            cfg_desc = sm.describe_endpoint_config(EndpointConfigName=name)
            model_names = [v["ModelName"] for v in cfg_desc.get("ProductionVariants", [])
                           if v.get("ModelName")]
        except Exception as e:  # noqa: BLE001
            print(f"  endpoint-config 조회 생략 ({name}): {str(e)[:110]}")
        for fn, arg, target in ([(sm.delete_endpoint, "EndpointName", name),
                                 (sm.delete_endpoint_config, "EndpointConfigName", name)]
                                + [(sm.delete_model, "ModelName", m) for m in model_names]):
            try:
                fn(**{arg: target})
                print(f"  deleted: {arg}={target}")
                deleted.append(target)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped {arg}={target}: {str(e)[:110]}")

    state.clear("endpoint_name")
    state.mark_stage("cleanup")
    remaining = [e["EndpointName"]
                 for e in sm.list_endpoints(NameContains=course.endpoint_prefix)["Endpoints"]]
    log.info(f"[cleanup] 남은 엔드포인트: {remaining or '없음'}")
    if remaining and not force:
        print("          --force를 사용하면 같은 prefix의 잔여 리소스도 정리합니다.")
    return {"deleted": deleted, "remaining": remaining}


STAGES: dict[str, Callable[..., dict[str, Any]]] = {
    "data": stage_data,
    "train": stage_train,
    "grpo": stage_grpo,
    "deploy": stage_deploy,
    "eval": stage_eval,
    "cleanup": stage_cleanup,
}


# ===========================================================================
# 5) 드라이버
# ===========================================================================
def _cmd(course_key: str, stages: str) -> str:
    """현재 진입점 형식에 맞는 재실행 명령을 반환합니다."""
    argv0 = os.path.basename(sys.argv[0] or "")
    if argv0.startswith("run_") and argv0.endswith(".py"):
        return f"python pipelines/{argv0} --stages {stages}"
    return f"python -m pipelines._common --course {course_key} --stages {stages}"


def _missing(course: CourseSpec, key: str, producer_stage: str,
             state: StateStore | None = None) -> str:
    """빠진 선행조건을 '무엇이 없고, 무엇을 실행하면 생기는지'로 알려 준다.

    state 를 넘기면 **실제로 보고 있는 파일 경로**를 찍는다. --state-dir 를 바꿔 쓸 때
    기본 경로를 안내하면 사용자가 엉뚱한 파일을 열어 본다.
    """
    where = state.path if state is not None else os.path.join(STATE_DIR, f"{course.key}.json")
    return (f"선행조건 누락: 상태에 '{key}' 가 없습니다 ({where}).\n"
            f"  먼저 실행하세요: {_cmd(course.key, producer_stage)}")


def resolve_stages(requested: str | list[str] | None,
                   course: CourseSpec | None = None) -> list[str]:
    """--stages 값을 정규 순서로 정렬하고 중복을 제거합니다.

    'all' 에서 빼는 두 단계:
       - cleanup: 방금 만든 endpoint 를 자동으로 지우면 평가와 데모를 할 수 없다.
       - grpo: SFT 로 충분한 경우가 많고, GPU 시간이 한 번 더 든다. 필수가 아니므로
                   기본에서 빼고 원할 때만 요청하게 한다. 'all+grpo' 로 켠다.

    속도 측정(TTFT/TPOT/ITL)은 pipelines/run_benchmark.py 가 따로 맡는다.
       파이프라인은 '모델을 만들어 배포하는' 흐름이고 벤치마크는 '이미 있는 endpoint 를 재는'
       일이라 선행조건도 실행 주기도 다르다.

    course 를 주면 그 코스에 없는 스테이지를 걸러낸다. 두 경우를 **다르게** 다룬다:
      - 'all': 조용히 뺀다. 'all' 은 "이 코스의 전 단계"라는 뜻이고, 계획 줄에
                     없는 단계가 찍히면 사용자가 그 단계가 돌 것이라 오해한다.
      - 명시 요청: HARD_GAP_STAGES 면 거부한다(ValueError). `--stages grpo` 를 치고 "skip"
                     한 줄만 보면 오타인지, 선행조건이 빠진 것인지, 애초에 불가능한 것인지
                     알 수 없다. 그 외(eval)는 계획에 남겨 스테이지 함수가 이유를 찍게 한다.
                     `--stages deploy,eval` 은 다섯 코스에 공통으로 안내하는 명령이고, 코스마다
                     종료 코드가 갈리면 "한 번 배우면 되는 CLI" 라는 전제가 깨진다.
    """
    ALL_PLUS = ("all+grpo", ["all+grpo"])
    explicit = requested not in (None, "", "all", ["all"], *ALL_PLUS)
    if not explicit:
        want_grpo = requested in ALL_PLUS
        skip = {"cleanup"} if want_grpo else {"cleanup", "grpo"}
        names = [s for s in STAGE_ORDER if s not in skip]
    else:
        raw = (requested if isinstance(requested, list)
               else [p.strip() for p in requested.split(",")])
        names = [n for n in raw if n]
        unknown = [n for n in names if n not in STAGES]
        if unknown:
            raise ValueError(f"알 수 없는 스테이지 {unknown}. 허용: {list(STAGE_ORDER)} 또는 'all'")

    plan = [s for s in STAGE_ORDER if s in set(names)]
    if course is None:
        return plan
    for stage in list(plan):
        why = unsupported_reason(course, stage)
        if not why:
            continue
        if explicit and stage in HARD_GAP_STAGES:
            raise ValueError(why)
        if explicit:
            continue          # 스테이지 함수가 같은 이유를 찍고 아무것도 만들지 않는다
        plan.remove(stage)
    return plan


def _free_form_output(course: CourseSpec) -> str:
    """GRPO 거부 메시지에서 '채점 기준이 하나로 안 정해지는 산출물'을 이 코스의 말로 부른다.

    eval_kind 로 가른다. 그 값이 이 코스의 지표가 무엇인지 정하는 곳이고(_shared_build),
    새 코스를 넣을 때 지표를 정하면 이 문구도 함께 정해진다.
    """
    return {"summarization": "요약문", "domain_qa": "답변 문장"}.get(course.track.eval_kind, "생성 텍스트")


# 이름을 직접 적었을 때 거부할 스테이지입니다. GRPO만 해당합니다.
# 없는 코스에서 GRPO를 요청하는 것은
#    설정 착각이지 오타가 아니고(사람이 GPU 시간 몇 시간을 기대한다), 조용히 통과하면 "정련까지
#    끝냈다"고 믿게 된다. eval 은 반대다: `--stages deploy,eval` 이 다섯 코스 공통 안내 명령이라
#    거부하면 평가가 없는 코스에서만 종료 코드가 2가 되어 CLI 표면이 코스별로 달라진다.
HARD_GAP_STAGES = ("grpo",)


def unsupported_reason(course: CourseSpec, stage: str) -> str | None:
    """이 코스가 이 스테이지를 **가지고 있지 않은** 이유. 지원하면 None.

    노트북 구성이 진실의 근거다. 요약 코스에 02a_train_grpo 노트북이 없는 것과 같은 이유로
    여기에도 grpo 가 없다. 두 진입점이 다른 단계 목록을 갖게 두면 안 된다.
    """
    if stage == "grpo" and not course.grpo_reward_kind:
        if course.multimodal:
            # 멀티모달 코스에는 이미지 프롬프트를 처리할 GRPO 경로가 없습니다.
            return (
                f"'{course.key}' 코스는 GRPO를 지원하지 않습니다.\n"
                "  이유: 현재 GRPO 경로는 텍스트 프롬프트만 처리하며 이 코스는 이미지를 입력으로 사용합니다.\n"
                f"  지원 코스: extraction, classification\n"
                f"  대안: {_cmd(course.key, 'data,train')} 실행 후 "
                "config.yaml의 max_train_samples 또는 epochs를 조정하세요.")
        output = _free_form_output(course)
        return (
            f"'{course.key}' 코스는 GRPO를 지원하지 않습니다.\n"
            f"  이유: 자유형 {output}의 보상을 안정적으로 프로그램화하기 어렵습니다.\n"
            "  지원 코스: extraction, classification\n"
            f"  대안: SFT 데이터를 늘리거나 수정한 뒤 다시 학습하세요 "
            f"({_cmd(course.key, 'data,train')})")
    if stage == "eval" and not course.has_eval_stage:
        return (f"'{course.key}' 코스에는 eval 스테이지가 없습니다. "
                "배포 스모크에서 정답(samples/ground_truth.json)과 대조하는 것이 검증 지점입니다.")
    return None


def _announce_billing(plan: list[str], cfg: PipelineConfig) -> None:
    """과금 리소스가 포함된 실행이면 시작 전에 한 줄로 안내합니다."""
    if cfg.runtime.dry_run:
        return
    billable = [s for s in plan if s in ("train", "grpo", "deploy")]
    if billable:
        print(_danger("이 실행에는 과금 단계가 포함됩니다") + f": {_bold(', '.join(billable))}.\n"
              "   각 단계 직전에 만들 리소스와 대략 요금을 다시 알립니다.")


# ---------------------------------------------------------------------------
# 실험 추적 보조 함수
# ---------------------------------------------------------------------------
def _mlflow_setup(course: CourseSpec, cfg: PipelineConfig) -> tuple[Any, str, str]:
    """추적 대상과 experiment, run 이름을 반환합니다."""
    m = cfg.mlflow
    target = mlflow_utils.resolve_target(m.tracking_uri, local_uri=m.local_uri)
    # 코스별 메트릭이 달라 experiment를 분리합니다.
    experiment = course.key
    # 실행을 구분할 수 있도록 시각을 포함합니다.
    run_name = f"{course.key}-{time.strftime('%Y%m%d-%H%M%S')}"
    if target.enabled and target.is_managed and not cfg.runtime.dry_run:
        # 버전과 실행 상태를 확인합니다.
        info = mlflow_utils.describe_managed(target, region=_region())
        if info:
            log.info("[mlflow] resource: %s", json.dumps(info, ensure_ascii=False, default=str))
            mlflow_utils.warn_version_gap(info)
            if info.get("kind") == "tracking-server" and info.get("is_active") == "Inactive":
                log.warning("[mlflow] Tracking Server가 중지되어 있습니다. "
                            "`aws sagemaker start-mlflow-tracking-server`로 시작하세요.")
    return target, experiment, run_name


def _mlflow_tags(course: CourseSpec, cfg: PipelineConfig, plan: list[str],
                 force: bool) -> dict[str, Any]:
    return {
        "kit.course": course.key,
        "kit.stages": ",".join(plan),
        "kit.dry_run": cfg.runtime.dry_run,
        "kit.force": force,
        "kit.model_size": cfg.model.size,
        "kit.serving_engine": cfg.serving.engine,
        "kit.region": _region(),
    }


def _mlflow_log_config(mlf: Any, cfg: PipelineConfig) -> None:
    """비교에 필요한 설정만 파라미터로 기록합니다."""
    t, g, sv, d, e = cfg.training, cfg.grpo, cfg.serving, cfg.data, cfg.evaluation
    mlf.log_params({
        "model.size": cfg.model.size,
        "model.id": cfg.model.id or "(preset)",
        "train.instance_type": t.instance_type,
        "train.epochs": t.epochs,
        "train.max_train_samples": t.max_train_samples if t.max_train_samples else "all",
        "train.batch_size": t.per_device_train_batch_size,
        "train.grad_accum": t.gradient_accumulation_steps,
        "train.learning_rate": t.learning_rate,
        "train.use_qlora": t.use_qlora,
        "train.merge_adapter": t.merge_adapter,
        "lora.r": t.lora.r,
        "lora.alpha": t.lora.alpha,
        "lora.dropout": t.lora.dropout,
        "grpo.prompt_source": g.prompt_source,
        "grpo.num_generations": g.num_generations,
        "serving.engine": sv.engine,
        "serving.instance_type": sv.instance_type,
        "serving.max_num_seqs": sv.max_num_seqs,
        "data.num_seed_samples": d.num_seed_samples,
        "data.num_synthetic": d.num_synthetic,
        "eval.num_examples": e.num_examples,
    })


def _mlflow_log_stage(mlf: Any, stage: str, result: Any) -> None:
    """스테이지 결과의 숫자는 메트릭, 나머지는 파라미터로 기록합니다."""
    if not mlf.enabled:
        return
    mlf.set_tags({f"kit.stage.{stage}": "done"})
    if not isinstance(result, dict) or not result:
        return
    nums = {f"{stage}.{k}": v for k, v in result.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    strs = {f"{stage}.{k}": v for k, v in result.items()
            if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool))}
    # 평가 점수는 차트에서 바로 찾을 수 있도록 접두어 없이도 기록합니다.
    if stage == "eval":
        nums.update({k: v for k, v in result.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)})
    mlf.log_metrics(nums)
    mlf.log_params(strs)
    if stage == "eval" and result:
        mlf.log_json(result, "eval_scores.json")


def run_stages(course_key: str, stages: str | list[str] | None, cfg: PipelineConfig,
               *, dry_run: bool = False, force: bool = False,
               state_dir: str = STATE_DIR, state_path: str | None = None,
               endpoint_name: str | None = None) -> dict[str, Any]:
    """스테이지들을 순서대로 실행. 첫 실패에서 멈춘다(뒤 단계가 옛 산출물로 도는 것을 막는다).

    dry_run 인자는 호출 편의용입니다. 실제 스위치는 cfg.runtime.dry_run(= DRY_RUN env)이고,
    load_config(dry_run=...) 이 둘을 이미 일치시킨다. 여기서는 어긋났을 때만 경고한다.
    state_path 를 주면(--state) 그 파일을 쓰고 state_dir 는 무시된다.
    """
    courses = load_courses()
    if course_key not in courses:
        raise ValueError(f"알 수 없는 코스 {course_key!r}. 허용: {list(courses)}")
    course = courses[course_key]
    if dry_run and not cfg.runtime.dry_run:
        print("WARNING: dry_run=True이지만 cfg.runtime.dry_run은 False입니다. "
              "load_config(dry_run=True)로 만든 설정을 사용하세요.")

    state = StateStore(course_key, state_dir=state_dir, dry_run=cfg.runtime.dry_run,
                       path=state_path)
    if endpoint_name:
        # 외부 엔드포인트 이름도 상태 파일에 기록해 eval과 cleanup이 같은 값을 사용하게 합니다.
        previous = state.get("endpoint_name")
        if previous and previous != endpoint_name:
            log.warning("--endpoint-name이 상태 파일의 값을 덮어씁니다: %s -> %s",
                        previous, endpoint_name)
        state.set(endpoint_name=endpoint_name)
    plan = resolve_stages(stages, course)   # 이 코스에 없는 단계는 여기서 걸러지거나 거부된다

    print(_dim("=" * 78))
    print(f"{_bold('course')}    : {_bold(course.key)} ({course.dir_name})")
    print(cfg.summary())
    print(state.summary())
    print(f"{_bold('stages')}    : {', '.join(plan)}"
          + (_warn("   [--force]") if force else ""))
    print(_dim("=" * 78))
    # 실행 전에 과금 가능성을 안내합니다.
    _announce_billing(plan, cfg)

    # 파이프라인 실행 전체를 run 하나로 묶고 스테이지 결과를 같은 run에 기록합니다.
    mlf_target, mlf_experiment, mlf_run_name = _mlflow_setup(course, cfg)
    with mlflow_utils.run(mlf_target, mlf_experiment, run_name=mlf_run_name,
                          tags=_mlflow_tags(course, cfg, plan, force)) as mlf:
        _mlflow_log_config(mlf, cfg)
        _mlflow_print_ui(mlf_target, mlf)
        # 학습 컨테이너가 자식 run을 만들 수 있도록 부모 run ID를 상태에 저장합니다.
        state.set(mlflow_run_id=mlf.run_id, mlflow_experiment=mlf_experiment if mlf.enabled else None)

        results: dict[str, Any] = {}
        for stage in plan:
            print(f"\n──── {stage} ────")
            try:
                results[stage] = STAGES[stage](course, cfg, state, force=force)
            except Exception as e:  # noqa: BLE001
                log.error("스테이지 %s 실패: %s", stage, e)
                mlf.set_tags({f"kit.stage.{stage}": "failed"})
                print(f"\n'{stage}' 스테이지에서 중단했습니다: {e}")
                print(f"   상태는 {state.path}에 보존했습니다. 원인을 해결한 뒤 다시 실행하세요:")
                print("   " + _cmd(course.key, ",".join(plan[plan.index(stage):])))
                raise StageFailed(f"{stage}: {e}") from e
            _mlflow_log_stage(mlf, stage, results[stage])
    print("\n" + "=" * 78)
    print(state.summary())
    if state.get("endpoint_name") and not _is_dry_value(state.get("endpoint_name")):
        print("엔드포인트는 삭제할 때까지 시간당 과금됩니다. 사용 후 정리하세요:\n"
              "   " + _cmd(course.key, "cleanup"))
    return results


# ===========================================================================
# 6) CLI
# ===========================================================================
def build_parser(default_course: str | None = None) -> argparse.ArgumentParser:
    # 현재 진입점에 맞는 예시 명령을 표시합니다.
    argv0 = os.path.basename(sys.argv[0] or "")
    if argv0.startswith("run_") and argv0.endswith(".py"):
        base = f"python pipelines/{argv0}"
    else:
        base = f"python -m pipelines._common --course {default_course or '<course>'}"
    # 부분 실행에서 사용하는 상태 파일 경로를 도움말에 표시합니다.
    where = os.path.join(STATE_DIR, f"{default_course or '<course>'}.json")
    p = argparse.ArgumentParser(
        description="코스 E2E 파이프라인 (data, train, grpo, deploy, eval, cleanup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            f"  {base} --stages all           # 전 단계(cleanup 제외)\n"
            f"  {base} --stages data,train    # model_data 를 상태 파일에 기록\n"
            f"  {base} --stages deploy,eval   # 그 값을 읽어 배포와 평가\n"
            f"  {base} --dry-run              # 과금 리소스 0으로 전 경로 검증\n"
            f"  {base} --stages cleanup       # endpoint 삭제 및 과금 중지\n"
            "\n"
            f"스테이지: {', '.join(STAGE_ORDER)}\n"
            "  data    시드 로드 + grounded 합성 + train/eval JSONL + S3 업로드\n"
            "  train   SFT 학습 잡 제출과 대기, model_data 기록\n"
            "  grpo    SFT 결과를 이어받아 GRPO 정련 (추출, 분류 코스만)\n"
            "  deploy  real-time endpoint 배포와 스모크, endpoint_name 기록\n"
            "  eval    held-out 평가 (코스별 지표)\n"
            "  cleanup endpoint, config, model 삭제 ('all' 에는 포함되지 않는다)\n"
            "\n"
            f"상태 파일: {where}\n"
            "  스테이지 간 값(model_data / endpoint_name ...)을 여기에 남겨 부분 실행을 잇는다.\n"
            "  --dry-run 은 <course>.dryrun.json 을 따로 쓴다(가짜 산출물이 실제 실행에 섞이지 않게).\n"),
    )
    p.add_argument("--course", default=default_course,
                   help="코스 키 (extraction/classification/summarization/domain_qa/mm_extraction)")
    p.add_argument("--stages", default="all",
                   help=(f"콤마 구분 스테이지. 허용: {','.join(STAGE_ORDER)}. "
                         "기본 all = grpo와 cleanup 제외(둘 다 필수가 아니며 추가 비용이 든다). "
                         "GRPO 까지 돌리려면 all+grpo"))
    p.add_argument("--config", default=None, help="config.yaml 경로(기본 <repo>/config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="과금 리소스를 만들지 않고 전 경로를 밟는다(config.yaml runtime.dry_run 을 이긴다)")
    p.add_argument("--force", action="store_true",
                   help="이미 만들어진 산출물이 있어도 스테이지를 다시 실행한다"
                        " (진행 중인 Job과 endpoint는 --force로도 새로 만들지 않으며"
                        " 중복 과금을 막기 위해 이어받거나 중단한다)")
    p.add_argument("--endpoint-name", default=None, metavar="NAME",
                   help="평가할 endpoint 이름. 생략하면 deploy 가 상태 파일에 남긴 것을 쓴다. "
                        "이 kit 밖에서 만든 endpoint 를 평가하려면 여기에 이름을 준다 "
                        "(주면 상태 파일에도 기록되어 이후 스테이지가 같은 것을 본다)")
    p.add_argument("--state-dir", default=STATE_DIR, help=f"상태 파일 디렉토리(기본 {STATE_DIR})")
    p.add_argument("--state", default=None, metavar="PATH",
                   help="상태 파일 경로를 통째로 지정(--state-dir 보다 우선). "
                        "--dry-run 과 함께 주면 접미사 .dryrun 이 붙는다")
    p.add_argument("--quiet", action="store_true",
                   help="진행 로그를 줄인다(WARNING 이상만). CI 에서 유용하다")
    p.add_argument("--show-state", action="store_true", help="상태만 출력하고 종료")
    return p


def main(argv: list[str] | None = None, *, default_course: str | None = None) -> int:
    """CLI 진입점. 코스별 래퍼(pipelines/run_<course>.py)는 default_course 만 넘기면 된다."""
    args = build_parser(default_course).parse_args(argv)
    if not args.course:
        print("--course를 지정하거나 코스별 run_<course>.py를 사용하세요.")
        return 2

    # 인자 검증이 끝난 뒤 무거운 모듈을 불러옵니다.
    from common.logging_utils import setup_logging

    # 설정 오류는 트레이스백 대신 사용자용 메시지로 처리합니다.
    #    load_config 을 아래 try 밖에서 부르면 ConfigError(=ValueError) 가 그 핸들러를 못 만나고
    #    파이썬 기본 트레이스백으로 새어 나간다(실측: --config 에 model.size=NOPE 를 주면
    #    허용값을 담은 좋은 메시지가 5프레임 스택 밑에 묻혔다).
    try:
        cfg = load_config(args.config, dry_run=True if args.dry_run else None)
    except (ValueError, RuntimeError, FileNotFoundError, TypeError, AttributeError) as e:
        print(f"오류: {e}")
        return 2
    # 진입점에서 1회. 라이브러리는 핸들러를 건드리지 않는다.
    # 기본 INFO 인 이유: 학습이 수십 분 도는 동안 화면이 조용하면 멈춘 것으로 오해한다.
    setup_logging("WARNING" if args.quiet else cfg.runtime.log_level)

    if args.show_state:
        # --dry-run 과 함께 주면 dry-run 상태 파일을 본다(실제 실행 상태와 별개 파일).
        print(StateStore(args.course, state_dir=args.state_dir,
                         dry_run=cfg.runtime.dry_run,
                         path=args.state).summary(verbose=True))
        return 0
    try:
        run_stages(args.course, args.stages, cfg, dry_run=args.dry_run, force=args.force,
                   state_dir=args.state_dir, state_path=args.state,
                   endpoint_name=args.endpoint_name)
    except StageFailed:
        return 1                      # run_stages가 이미 원인과 재실행 명령을 출력했습니다.
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # 스테이지 진입 전 오류도 사용자에게 표시합니다.
        print(f"오류: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
