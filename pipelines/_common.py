"""
pipelines/_common.py — 코스별 E2E 스테이지 구현 + 상태 저장소 + 실행 드라이버

🔴 이 폴더의 코드는 **개발자 머신에서 돌며 SageMaker 를 조종한다**.
   컨테이너 안에서 도는 학습 코드는 tracks/*/scripts/ (train.py / train_grpo.py / train_mm.py)다.
   둘을 섞지 말 것 — 여기서는 그 스크립트를 SourceCode 로 '제출'만 한다.

이 파일은 노트북 로직의 **두 번째 진입점**이다(노트북을 대체하지 않는다). 노트북이 셀 사이에서
%store 로 넘기던 값(model_data / endpoint_name ...)을 IPython 없이 넘기기 위해 코스별 JSON
상태 파일을 쓴다: .pipeline_state/<course>.json (gitignore).
dry-run 은 .pipeline_state/<course>.dryrun.json 과 tracks/*/data/dryrun/ 을 따로 쓴다 —
가짜 산출물이 실제 실행의 재개 판단에 섞이지 않게 하려는 것이다(StateStore / data_dir 참고).

  python pipelines/run_extraction.py --stages data,train   # model_data 를 기록
  python pipelines/run_extraction.py --stages deploy       # 그 값을 읽어 배포
  python pipelines/run_extraction.py --dry-run             # 과금 리소스 0으로 전 경로 검증

코스별 진입 스크립트(pipelines/run_<course>.py)는 아래 3줄이면 된다:

    from pipelines._common import main
    if __name__ == "__main__":
        raise SystemExit(main(default_course="extraction"))

🔴 %store 를 파일로 바꾼 이유 — 노트북이 값비싸게 배운 교훈:
   %store 는 커널·코스·리전을 넘어 공유되는 **전역** 저장소다. 여러 코스를 돌리면 마지막 코스의
   endpoint_name 이 값을 덮어써, 요약 코스가 멀티모달 endpoint(max_model_len=2048)를 호출해
   "maximum context length is 2048" 400 에러가 났다(요약 endpoint 는 4096 이라 정상인데도).
   코스당 파일 하나면 이 충돌이 구조적으로 불가능하다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
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

from common.logging_utils import get_logger  # noqa: E402  (경로 세팅 이후여야 한다)
from pipelines._config import PipelineConfig, load_config  # noqa: E402

log = get_logger("pipelines")

STAGE_ORDER = ("data", "train", "grpo", "deploy", "eval", "cleanup")


class StageFailed(RuntimeError):
    """스테이지 실행 실패 — run_stages 가 이미 사람이 읽을 메시지를 출력했다는 표시.

    🔴 왜 별도 타입인가: CLI 가 예외를 뭉뚱그려 삼키면(except Exception: return 1) 설정·인자 오류가
       **아무 메시지 없이 종료 코드 1** 로만 나온다. 이미 출력한 실패와 아직 출력하지 않은 실패를
       구분해야 후자를 반드시 찍을 수 있다.
    """

# 🔴 에이전틱 단계(05_agentic_strands / 06_agentcore_deploy)는 일부러 노트북 전용으로 둔다 —
#    질문을 바꿔 가며 답을 보는 탐색적 작업이라 배치 실행에 얹을 이득이 없다.


# ===========================================================================
# 1) 코스 스펙 — tracks/_shared_build.TrackSpec 을 그대로 재사용한다
# ===========================================================================
@dataclass(frozen=True)
class CourseSpec:
    """한 코스를 파이프라인으로 돌리기 위한 정보.

    🔴 서빙 길이·생성 길이·GRPO 여부 같은 '측정으로 정한 값'은 여기서 다시 정의하지 않고
       tracks/_shared_build.TrackSpec 을 들고 온다(track 속성). 값을 두 곳에 두면 노트북과 CLI 가
       다른 컨텍스트 길이로 돌아, 노트북에서만 통과하는 400(context length exceeded)이 생긴다.
    """
    key: str                    # config.TRACKS 키 (extraction/classification/...)
    dir_name: str               # tracks/ 하위 디렉토리
    track: Any                  # tracks._shared_build.TrackSpec
    train_entry: str            # scripts/ 안의 학습 스크립트 (train.py | train_mm.py)
    multimodal: bool = False
    has_synth: bool = True      # 멀티모달 코스는 합성 단계가 없다(이미지 합성은 별개 문제)
    has_eval_stage: bool = True # 멀티모달 코스에는 04 평가 노트북이 없다 → 스테이지도 없음

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
        # 🔴 학습 길이와 분리한다. 학습은 '입력+정답'을 자르지만 서빙은 '입력+생성'이 함께 컨텍스트에
        #    들어가므로, 학습값을 서빙 컨텍스트로 쓰면 긴 입력 코스에서 vLLM 이 400 으로 거부한다
        #    (요약 실측: 프롬프트 max 2006 + 생성 256 > 2048). 지정이 없으면 학습 길이의 2배.
        return self.track.serve_max_model_len or self.track.max_seq_length * 2

    @property
    def gen_max_tokens(self) -> int:
        return self.track.gen_max_tokens

    @property
    def grpo_reward_kind(self) -> str:
        return self.track.grpo_reward_kind


def _mm_track_spec():
    """멀티모달 코스(05)의 TrackSpec — 노트북 빌더가 인라인으로 만들던 값을 그대로 옮긴 것.

    03_deploy_mm_endpoint 실측 근거:
      · serve_max_model_len=2048 — 이 코스는 입력이 이미지라 텍스트 컨텍스트가 짧다.
      · gen_max_tokens=768 — 정답 JSON 이 최대 592 토큰(실측 100건)이라 512 로는 잘린다.
    """
    from _shared_build import TrackSpec
    return TrackSpec(
        key="mm_extraction", dir_name="05_multimodal_extraction",
        title="멀티모달 추출(이미지→JSON)", endpoint_prefix="gemma-mm-extraction",
        max_seq_length=2048, use_qlora=True, eval_kind="extraction",
        tool_name="extract_receipt_json", tool_doc="", agent_system="",
        smoke_user="", deploy_smoke_user="",
        serve_max_model_len=2048, gen_max_tokens=768,
        has_local_serve=False,
    )


def load_courses() -> dict[str, CourseSpec]:
    """코스 레지스트리. TrackSpec 은 노트북 빌더가 쓰는 것과 **같은 객체**를 재사용한다."""
    import importlib.util

    # 01(플래그십)의 spec 은 그 코스의 빌더 안에 있다 → 파일에서 직접 로드(패키지가 아니다).
    flagship_path = os.path.join(TRACKS_ROOT, "01_extraction_to_json", "_build_notebooks.py")
    spec = importlib.util.spec_from_file_location("_bn_extraction", flagship_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    extraction = mod._flagship_spec()

    import build_all_tracks   # tracks/build_all_tracks.py — SPECS 에 02/03/04 가 있다

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
    """코스의 데이터 어댑터(tracks/<dir>/track_data.py) 를 파일 경로로 로드.

    🔴 노트북은 트랙 디렉토리를 cwd 로 두고 `import track_data` 하지만, CLI 는 리포 루트에서
       실행되므로 이름 충돌(다섯 코스가 모두 track_data.py)을 피하려면 코스별 모듈명으로
       파일에서 직접 로드해야 한다. tests/prepare_dryrun_data.py 와 같은 방식.
    """
    import importlib.util

    path = os.path.join(course.dir, "track_data.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} 가 없습니다 — 코스 디렉토리 구성을 확인하세요.")
    name = f"track_data_{course.key}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 2) 상태 저장소 — %store 대체
# ===========================================================================
# 노트북이 쓰던 키 이름을 그대로 유지한다(사람이 두 경로를 오갈 때 헷갈리지 않게).
STATE_KEYS = ("bucket", "role", "model_data", "grpo_model_data", "endpoint_name")
# 부가 기록(사람이 사후에 무엇이 어디로 갔는지 추적하기 위한 값 — 스테이지 판단에는 안 쓴다)
BOOKKEEPING_KEYS = ("train_s3", "training_job", "grpo_job", "region", "engine", "dry_run")


class StateStore:
    """.pipeline_state/<course>.json 읽기/쓰기.

    - 파일이 없거나 깨져 있으면 **빈 상태로 취급하고 경고**한다(실행을 멈추지 않는다 — 상태 파일은
      캐시일 뿐이고, 없으면 그 스테이지를 다시 돌리면 된다).
    - 쓰기는 tmp+rename 원자적 쓰기다. 학습 대기 중 Ctrl-C 로 끊겨도 반쯤 쓰인 JSON 이 남아
      다음 실행이 "corrupt" 로 시작하는 일을 막는다.
    """

    def __init__(self, course_key: str, state_dir: str = STATE_DIR,
                 *, dry_run: bool = False, path: str | None = None) -> None:
        # 🔴 dry-run 은 별도 파일(<course>.dryrun.json)에 쓴다. 같은 파일을 쓰면 dry-run 이 남긴
        #    model_data='dryrun://...' 를 다음 실제 실행이 "이미 학습됨"으로 보고 train 을 건너뛰고,
        #    그 가짜 URI 로 배포를 시도한다(ValidationException: Could not access model data).
        #    분리하면 그 혼입이 구조적으로 불가능하다 — 코스별 분리와 같은 논리다.
        # path 는 --state 로 파일을 통째로 지정한 경우다. 그때도 dry-run 접미사는 그대로 붙인다 —
        # 사용자가 고른 이름이라도 가짜 산출물과 실제 산출물이 한 파일에 섞이면 위 사고가 되살아난다.
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
            print(f"⚠️  상태 파일을 읽을 수 없어 빈 상태로 시작합니다: {self.path} ({e})")
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
        """스테이지 완료 시각 기록 — 상태 파일만 보고 '무엇이 언제 돌았나'를 알 수 있게."""
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
        os.replace(tmp, self.path)   # 원자적 교체 — 반쯤 쓰인 파일이 남지 않는다

    def summary(self, *, verbose: bool = False) -> str:
        """상태 요약. verbose=True 면 부가 기록(잡 이름·리전·엔진)까지 — `--show-state` 용."""
        lines = [f"state     : {self.path}"]
        # 🔴 첫 실행은 전 항목이 비어 있는 게 정상이다(bucket/role 은 첫 스테이지가 AWS 에서 해석하고,
        #    나머지는 해당 스테이지가 만든다). 그걸 '(없음)' 다섯 줄로 늘어놓으면 설정이 잘못된 것처럼
        #    보이므로, 아무것도 없을 때는 한 줄로 알린다.
        filled = {k: self._data.get(k) for k in STATE_KEYS if self._data.get(k)}
        if not filled and not verbose:
            lines.append("  (첫 실행 — bucket·role·산출물은 각 스테이지가 채웁니다)")
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
DRY_PREFIX = "dryrun://"   # dry-run 이 만든 가짜 산출물 표시 — AWS 경계에서 이 값을 걸러낸다


def _is_dry_value(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(DRY_PREFIX)


def _region() -> str:
    """현재 리전 — common.config 가 유일한 출처(env AWS_REGION)."""
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

    🔴 dry-run 에서는 AWS 를 아예 건드리지 않는다 — default_bucket() 은 없으면 버킷을 만들고,
       role 자동 탐지는 IAM 을 훑는다. 둘 다 "리소스를 만들지 않는다"는 dry-run 약속에 어긋난다.
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
    role = config.resolve_sagemaker_role(sess)      # env → get_execution_role → IAM 자동 탐지
    bucket = cfg.aws.s3_bucket or config.S3_BUCKET or sess.default_bucket()
    state.set(role=role, bucket=bucket, region=region, dry_run=False)
    return AwsContext(session=sess, role=role, bucket=bucket, region=region)


def _fold_system_messages(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    """추론 messages — system 지시문을 첫 user 턴에 병합한다.

    🔴 학습 데이터가 그렇게 만들어져 있기 때문이다(track_data.to_messages: Gemma instruct 템플릿이
       system role 을 거부해 fold 한다). 추론에서 system role 을 따로 보내면 학습과 표면형이 달라져
       같은 모델이 "학습이 안 된 것처럼" 답한다(실측: system·스키마를 빼면 일반 챗봇 응답).
    """
    return [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]


def invoke_chat(endpoint_name: str, messages: list[dict[str, Any]], *, region: str,
                max_tokens: int, temperature: float = 0.2) -> tuple[str, str | None]:
    """endpoint 호출 → (텍스트, finish_reason).

    🔴 max_tokens 다 — max_new_tokens 가 아니다. OpenAI 호환(messages) 스키마의 키는 max_tokens 이고,
       vLLM 은 모르는 키를 **조용히 무시**하므로 max_new_tokens 를 보내면 길이 제한이 아예 적용되지
       않는다(에러도 없다).
    🔴 finish_reason 을 함께 돌려주는 이유: 응답이 잘렸는지 알 수 있는 신호가 이것뿐이다.
       잘려도 예외가 없고 HTTP 200 이라, 확인하지 않으면 '모델이 이상한 JSON 을 낸다'로 오진한다.
       common.aws_utils.invoke_sagemaker_chat 은 텍스트만 반환하므로 여기서 raw 응답을 본다
       (스키마 파싱은 그 모듈의 _parse_endpoint_response 를 재사용 — 파서를 복제하지 않는다).
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
        print(f"⚠️  {label}이 max_tokens({max_tokens})에서 잘렸습니다(finish_reason='length'). "
              "정답이 이보다 길면 지표가 구조적으로 과소 측정됩니다 — "
              "TrackSpec.gen_max_tokens 를 올리거나 서빙 컨텍스트를 늘리세요.")
        return True
    return False


# 컨테이너 로거가 이미 붙인 타임스탬프·레벨·모듈. 우리 로거가 또 붙이므로 떼어낸다.
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
        # 🔴 filter_log_events 대신 get_log_events 를 쓴다. filter 쪽은 첫 호출에서 오래된
        #    페이지를 돌려줘 빈 결과가 나오는 일이 있다. get 쪽은 startFromHead=False 로
        #    '최신 40줄'을 확정적으로 준다.
        # 🔴 이어 읽을 때의 파라미터는 nextToken 이다(startFromToken 은 존재하지 않는다).
        #    첫 호출은 startFromHead=False 로 '최신 40줄'만 가져온다.
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
        if _LOG_NOISE.search(m):         # HF 텔레메트리·config.json HEAD 요청 등
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
                # 🔴 컨테이너 로그는 'Training Container Execution Completed' 에서 끝난다.
                #    그 뒤 S3 업로드는 SageMaker AI 가 하는 일이라 로그가 없어, 화면이 몇 분간
                #    조용해진다. 멈춘 것으로 오해하지 않게 무엇을 기다리는지 알려 준다.
                log.info("  (모델 아티팩트를 S3 로 올리는 중입니다. 컨테이너 로그는 여기서 끝나고,"
                         " 크기에 따라 수 분 걸립니다.)")
                upload_started = time.time()

        # 🔴 Training 단계에 들어가면 CloudWatch 로그를 함께 흘린다. 상태 문자열만으로는
        #    학습이 진행되는지 OOM 으로 멈춰 있는지 구분할 수 없다.
        if secondary in ("Training", "Uploading") or st in ("Failed", "Stopped"):
            if not tailing:
                log.info("  ── CloudWatch 로그 ──────────────────────────────")
                tailing = True
            try:
                msgs, token = _tail_training_logs(job.training_job_name, region=_region(),
                                                  token=token)
                for m in msgs:
                    log.info(f"  │ {m[:180]}")
            except Exception as e:      # 로그를 못 읽어도 학습 대기는 계속한다
                log.warning(f"  (로그 조회 실패: {type(e).__name__}. 상태 폴링은 계속합니다)")

        # 업로드는 로그가 없으므로 경과 시간만이라도 흘려 준다.
        if secondary == "Uploading" and upload_started:
            waited = int(time.time() - upload_started)
            if waited and waited // 30 != last_tick:
                last_tick = waited // 30
                log.info(f"  업로드 대기 중… {waited // 60}분 {waited % 60}초 경과")
        if st in ("Completed", "Failed", "Stopped"):
            if st != "Completed":
                # 🔴 MaxRuntimeExceeded 는 FailureReason 이 비어 있어(상태만 Stopped) 원인이 안 보인다.
                #    describe 로 secondary transition 메시지까지 찍어 준다.
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
    """이전 실행이 제출해 둔 학습 Job 을 이어받는다. (상태, model_data) 를 반환.

    🔴 왜 필요한가: `--stages all` 로 Job 을 제출한 뒤 Ctrl+C 를 누르거나 세션이 끊겨도
       **Job 은 AWS 에서 계속 돕니다**. 그런데 스킵 판단이 완료 산출물(model_data)만 본다면,
       다시 실행했을 때 "아직 없네" 하고 **같은 Job 을 또 제출**합니다. GPU 인스턴스가 두 대
       도는 것이고, 그만큼 청구됩니다.
       그래서 산출물이 없어도 **Job 이름이 남아 있으면 그 Job 의 상태를 먼저 조회**합니다.

    반환하는 상태:
      "InProgress" — 이어서 대기하고 완료되면 model_data 를 돌려준다
      "Completed"  — 이미 끝났다. model_data 만 회수한다(재학습 없음)
      "Failed"/"Stopped" — 왜 실패했는지 보여주고 호출자가 중단한다
      "NotFound"   — Job 이 사라졌다(콘솔에서 지웠거나 이름이 어긋남). 새로 제출해도 된다
    """
    from common import aws_utils
    from sagemaker.core.resources import TrainingJob

    try:
        job = TrainingJob.get(job_name)
    except Exception as e:                       # 이름이 없거나 권한/리전 불일치
        print(f"[{label}] 이전 Job '{job_name}' 을 조회할 수 없습니다({type(e).__name__}). "
              "새로 제출합니다.")
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
        print(f"[{label}] 🔴 이전 실행이 제출한 Job 이 아직 돌고 있습니다: {job_name} "
              f"({getattr(job, 'secondary_status', '?')}{elapsed})\n"
              f"        새로 제출하지 않고 이어서 대기합니다. "
              f"멈추려면 콘솔이나 `aws sagemaker stop-training-job` 을 쓰세요.")
        _print_console_links(region, training_job=job_name)
        st = _wait_training_job(job, poll_seconds=poll_seconds)

    if st == "Completed":
        md = job.model_artifacts.s3_model_artifacts
        print(f"[{label}] 이전 Job 이 이미 완료돼 있습니다: {job_name}\n"
              f"        학습을 다시 돌리지 않고 산출물을 씁니다: {md}")
        return "Completed", md

    # Failed / Stopped
    print(f"[{label}] 이전 Job 이 {st} 상태입니다: {job_name}")
    aws_utils.training_job_status(job_name, region)
    return st, None


def _resume_or_submit_guard(state: StateStore, key: str, *, label: str, cfg: PipelineConfig,
                            force: bool) -> str | None:
    """제출 전에 이전 Job 을 확인한다. model_data 를 얻었으면 그 값을, 아니면 None.

    force=True 라도 **돌고 있는 Job 을 그냥 두고 새로 제출하지는 않는다** — 두 대가 동시에
    과금되기 때문이다. 그 경우엔 무엇을 멈춰야 하는지 알려 주고 중단한다.
    """
    prev = state.get(key)
    if not prev or str(prev).startswith(DRY_PREFIX):
        return None

    st, md = _resume_training_job(prev, region=_region(), label=label,
                                 poll_seconds=cfg.runtime.poll_seconds)
    if st == "Completed":
        return md
    if st == "NotFound":
        return None
    if st == "InProgress":     # _resume_training_job 이 끝까지 기다렸으므로 여기 오면 실패로 끝난 것
        raise RuntimeError(f"이어받은 Job 이 완료되지 못했습니다: {prev}")
    raise RuntimeError(
        f"이전 Job 이 {st} 상태입니다({prev}). 위 로그에서 원인을 확인하세요.\n"
        f"  같은 설정으로 다시 제출하려면: state 에서 {key} 를 지우거나 --force 를 붙이세요.")


def _submit_training_job(*, cfg: PipelineConfig, ctx: AwsContext, base_job_name: str,
                         entry_script: str, source_dir: str, hyperparameters: dict[str, Any],
                         max_runtime_hours: float,
                         input_channels: list[tuple[str, str]]) -> Any:
    """ModelTrainer 구성 + 비동기 제출 → TrainingJob 객체.

    input_channels: [(channel_name, s3_uri)]. 빈 리스트면 채널 없이 제출한다
                    (멀티모달 코스는 train_mm.py 가 컨테이너 안에서 시드 데이터셋을 받는다).
    """
    from common import config, dlc
    from sagemaker.core.training.configs import Compute, InputData, SourceCode, StoppingCondition
    from sagemaker.train.model_trainer import ModelTrainer

    image_uri = dlc.resolve_training_image(ctx.region)
    if not image_uri:
        raise RuntimeError(
            "학습 이미지 해석 실패 — config.yaml training.dlc(repository/tag) 또는 .env 의 "
            f"DLC_IMAGE_URI 를 확인하세요(리전 포함 완전 URI). 태그 목록: {dlc.AVAILABLE_IMAGES_URL}")
    print(f"  training image: {image_uri}")

    environment = {"HF_TOKEN": config.get_hf_token()} if config.get_hf_token() else {}

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
        # 🔴 반드시 명시한다. 생략하면 SDK 가 3600초(1시간)를 조용히 넣고
        #    (sagemaker/train/defaults.py DEFAULT_MAX_RUNTIME_IN_SECONDS), 그 한도는
        #    Pending(용량 대기) + Downloading + Training + merge/업로드 전체를 센다.
        #    실측: 학습 189 step 을 전부 끝낸 뒤 merge 도중 잘려 서빙용 머지 모델이 없었고,
        #    FailureReason 도 비어 있어(상태만 Stopped) 원인을 찾기 어려웠다.
        stopping_condition=StoppingCondition(max_runtime_in_seconds=int(max_runtime_hours * 3600)),
    )
    kwargs: dict[str, Any] = {"wait": False, "logs": False}   # 비동기 제출 — 끊겨도 잡은 계속 돈다
    if input_channels:
        kwargs["input_data_config"] = [InputData(channel_name=name, data_source=uri)
                                       for name, uri in input_channels]
    trainer.train(**kwargs)
    job = trainer._latest_training_job
    print(f"  submitted: {job.training_job_name}")
    _print_console_links(ctx.region, training_job=job.training_job_name)
    return job


# 🔴 us-east-1 온디맨드 시간당 요금(USD) — AWS Price List API 에서 확인한 값이다(라이브 검증 2026-08):
#    pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json
#    (Training/Hosting usagetype 단가가 동일해 하나로 둔다.)
#    감을 주기 위한 값일 뿐 청구 예측이 아니다 — 리전마다 다르고 요금표는 바뀐다. 그래서 출력은
#    항상 'us-east-1 기준'이라고 밝히고, 현재 리전이 다르면 그 사실을 함께 찍는다.
#    이 kit 이 실제로 쓰는 인스턴스만 담는다(GEMMA4_PRESETS + config.yaml 의 instance_type).
_HOURLY_USD_US_EAST_1: dict[str, float] = {
    "ml.g5.2xlarge": 1.515,     # E2B/E4B 프리셋
    "ml.g5.12xlarge": 7.09,     # 12B/26B-A4B 프리셋
    "ml.g6.2xlarge": 1.222,     # config.yaml 기본값 (L4 24GB)
    "ml.g6.4xlarge": 1.654,
    "ml.g6.12xlarge": 5.752,
    "ml.g6e.2xlarge": 2.80,     # L40S 45GB — OOM 시 올려 가는 후보
    "ml.g6e.12xlarge": 13.12,   # 31B 프리셋
}

_cost_warning_shown = False


# ── CLI 화면 컬러 ───────────────────────────────────────────────────────────
# 로그는 logging_utils 의 _ColorFormatter 가 칠한다. 여기는 print 로 내는 '화면'
# (실행 시작 헤더·과금 확인) 전용이다. 같은 규칙으로 터미널에서만 켠다.
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
    """과금이 시작되는 스테이지 직전에 '무엇을 만들고 대략 얼마인지'를 찍는다.

    🔴 노트북은 셀 위 마크다운의 비용 경고를 사람이 읽고 한 셀씩 실행하지만, CLI 는
       `--stages all` 한 줄로 학습·배포가 연달아 돈다 — 만들기 직전이 사람이 볼 유일한 지점이다.
       dry-run 에서도 찍는다(무엇이 얼마에 만들어질지 미리 보는 것이 dry-run 의 목적).
    서술형 경고 본문은 common.aws_utils.COST_WARNING 이 유일한 출처다(복제 금지) — 실행당 1회만
    찍는다. train·grpo·deploy 에서 같은 문단이 세 번 반복되면 사람이 읽지 않는다.
    """
    global _cost_warning_shown
    from common import aws_utils

    rate = _HOURLY_USD_US_EAST_1.get(instance_type)
    print(_danger(f"  💳 과금 시작") + f" — 만들 것: {what}  [{_bold(instance_type)} x1]")
    if rate is None:
        # 표에 없는 인스턴스로 바꿨다는 뜻 → 추정치를 꾸며내지 않고 어디서 확인하는지 알려 준다.
        print(f"     시간당 요금: 요금표에 없는 인스턴스입니다 — "
              "https://aws.amazon.com/sagemaker-ai/pricing/ 에서 확인하세요.")
    else:
        line = f"     요금: 약 ${rate:.3f}/시간"
        if cap_hours:
            # 상한이지 예상 청구액이 아니다 — 잡이 먼저 끝나면 그 시점에 과금이 멈춘다.
            line += f" → 한도 {cap_hours}시간을 다 쓰면 최대 약 ${rate * cap_hours:.2f}"
        if until_deleted:
            line += f" → 삭제할 때까지 계속(하루 방치 시 약 ${rate * 24:.2f})"
        print(line)
        print("     [us-east-1 온디맨드 기준 — 참고용, 청구 예측이 아닙니다]")
        if region != "us-east-1":
            print(f"     ⚠️ 현재 리전은 {region} 입니다 — 실제 요금은 리전마다 다릅니다.")
    if not _cost_warning_shown:
        aws_utils.print_cost_warning()
        _cost_warning_shown = True


def _print_console_links(region: str, *, training_job: str | None = None,
                         endpoint_name: str | None = None) -> None:
    """CloudWatch/콘솔 링크를 평문으로 출력.

    🔴 common.aws_utils.cw_links 는 IPython HTML 객체를 돌려주므로 CLI 에서 쓸 수 없다
       (IPython 의존을 여기 들이지 않는다는 것이 이 폴더의 규칙). URL 규약은 같다:
       하위 경로의 '/'는 $252F, 쿼리 '?'는 $3F 로 이중 인코딩.
    """
    if training_job:
        grp = "$252Faws$252Fsagemaker$252FTrainingJobs"
        print(f"    console: https://{region}.console.aws.amazon.com/sagemaker/home"
              f"?region={region}#/jobs/{training_job}")
        print(f"    logs   : https://{region}.console.aws.amazon.com/cloudwatch/home"
              f"?region={region}#logsV2:log-groups/log-group/{grp}"
              f"$3FlogStreamNameFilter$3D{training_job}")
        print("    (로그는 잡이 Training 단계에 들어간 뒤 생깁니다 — 그 전엔 로그 그룹이 없는 게 정상)")
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
    한도는 학습 시간만이 아니라 대기·pull·merge 까지 포함하므로 merge/업로드용 여유를 남겨야 한다.
    """
    steps = math.ceil(n_samples / max(1, accum)) * epochs
    eta_min = steps * seconds_per_step / 60
    print(f"  학습 {n_samples}건 x {epochs}epoch ≈ {steps:.0f} step → "
          f"학습 ~{eta_min:.0f}분 + 머지/업로드 ~5분 (한도 {max_runtime_hours}시간)")
    if eta_min / 60 >= max_runtime_hours:
        raise RuntimeError(
            f"예상 학습 시간({eta_min:.0f}분)이 training.max_runtime_hours"
            f"({max_runtime_hours}시간)에 육박합니다. config.yaml 의 training.max_train_samples/"
            "epochs 를 줄이거나 max_runtime_hours 를 올리세요 "
            "(머지·업로드용으로 최소 15분은 남겨 두세요).")


def _counts(cfg: PipelineConfig) -> tuple[int, int, int]:
    """(시드 건수, 합성 건수, 평가 건수) — dry-run 이면 소량 값."""
    if cfg.runtime.dry_run:
        return (cfg.data.dry_run_seed_samples, cfg.data.dry_run_synthetic,
                cfg.evaluation.dry_run_num_examples)
    return cfg.data.num_seed_samples, cfg.data.num_synthetic, cfg.evaluation.num_examples


def data_dir(course: CourseSpec, cfg: PipelineConfig) -> str:
    """이 실행이 데이터 파일을 쓸 디렉토리.

    🔴 dry-run 은 별도 하위 디렉토리(data/dryrun/)를 쓴다. 같은 data/train.jsonl 에 쓰면
       dry-run 이 만든 **소량 stub 파일**(시드 8건 복제)이 남고, 그 뒤 실제 실행은 "파일이 이미
       있으니 skip" 하며 그 stub 으로 학습을 제출한다 — 아무 경고 없이 8건짜리 모델이 나온다.
       노트북에는 없던 위험이다(노트북은 셀을 순서대로 다시 돌리므로 항상 덮어썼다).
       두 경로 모두 tracks/*/data/ 아래라 .gitignore 가 이미 덮는다.
    """
    base = course.data_dir
    return os.path.join(base, "dryrun") if cfg.runtime.dry_run else base


# ===========================================================================
# 4) 스테이지 — 각각 (course, cfg, state) 를 받아 '만든 것'을 반환한다
# ===========================================================================
def stage_data(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """시드 로드 + grounded 합성 + train/eval JSONL 작성 + S3 업로드.

    반환: {"train_path", "eval_path", "train_s3"}
    """
    td = load_track_data(course)
    n_seed, n_synth, n_eval = _counts(cfg)

    if course.multimodal:
        # 🔴 멀티모달 코스는 학습 데이터를 S3 로 올리지 않는다 — train_mm.py 가 컨테이너 안에서
        #    seed_dataset(cord-v2)을 직접 받는다(이미지가 parquet 에 내장돼 로컬 왕복이 비싸다).
        #    그래서 이 스테이지는 '시드가 실제로 로드되는지' 확인만 한다.
        log.info(f"[data] {course.key}: 멀티모달 코스 — 학습 데이터는 컨테이너 안에서 로드합니다.")
        seeds = td.load_seed_examples(2, token=_hf_token())
        print(f"  시드 로드 확인: {len(seeds)}건 (images/messages 컬럼 = TRL VLM 포맷)")
        state.mark_stage("data")
        return {"train_path": None, "eval_path": None, "train_s3": None}

    ddir = data_dir(course, cfg)
    os.makedirs(ddir, exist_ok=True)
    train_path = os.path.join(ddir, "train.jsonl")
    eval_path = os.path.join(ddir, "eval.jsonl")

    if os.path.isfile(train_path) and not force:
        n_lines = sum(1 for _ in open(train_path, encoding="utf-8"))
        log.info(f"[data] skip — {train_path} 가 이미 있습니다({n_lines}건). 다시 만들려면 --force.")
    else:
        log.info(f"[data] {course.key}: 시드 {n_seed}건 + 합성 {n_synth}건")
        seeds = td.load_seed_examples(n_seed, token=_hf_token())
        print(f"  파싱된 시드: {len(seeds)}건")

        synth_msgs: list[list[dict[str, str]]] = []
        if course.has_synth and n_synth > 0:
            synth_msgs = _generate_synthetic(td, cfg, n_synth)

        rows = [td.to_messages(s) for s in seeds] + synth_msgs
        _write_messages_jsonl(rows, train_path)
        print(f"  학습셋 {len(rows)}건 → {train_path}")

        # 🔴 held-out 은 학습에 쓴 '앞 n_seed 건' **뒤** 인덱스에서 뽑는다. 넉넉히 로드해 뒤쪽 N건을
        #    쓰는 방식은 held-out 이 학습 구간 안쪽으로 들어가 점수가 부풀려진다(실측).
        pool = td.load_seed_examples(n_seed + n_eval, token=_hf_token())
        heldout = pool[n_seed:n_seed + n_eval]
        if not heldout:
            print(f"  ⚠️ 시드가 {len(pool)}건뿐이라 학습 구간({n_seed}건) 뒤에 남는 예시가 없습니다 — "
                  "data.num_seed_samples 를 줄이거나 더 큰 시드를 쓰세요.")
        else:
            with open(eval_path, "w", encoding="utf-8") as f:
                for ex in heldout:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  held-out {len(heldout)}건 → {eval_path} "
                  f"(시드 인덱스 {n_seed}~{n_seed + len(heldout) - 1}, 학습 구간 제외)")

    if cfg.runtime.dry_run:
        train_s3 = f"{DRY_PREFIX}{cfg.data.s3_prefix}/{course.key}/data/train.jsonl"
        log.info(f"[data] dry-run — S3 업로드 생략 ({train_s3})")
    else:
        from common import aws_utils
        ctx = aws_context(cfg, state)
        key = f"{cfg.data.s3_prefix}/{course.key}/data/{os.path.basename(train_path)}"
        # 내용 해시 비교로 조건부 업로드 — 여러 번 돌려도 바뀐 게 없으면 재업로드하지 않는다.
        train_s3 = aws_utils.upload_if_changed(train_path, ctx.bucket, key, region=ctx.region)

    state.set(train_s3=train_s3)
    state.mark_stage("data")
    return {"train_path": train_path, "eval_path": eval_path, "train_s3": train_s3}


def _hf_token() -> str | None:
    """호출 시점에 새로 해석 — HF_HOME 을 나중에 설정해도 반영된다(common.config 규약)."""
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

    # 🔴 dry-run 은 Bedrock 을 호출하지 않는다. 건수만 줄이면 호출은 그대로 일어나고,
    #    Bedrock 은 토큰당 과금이다 — 100건이면 생성 10회 + critique ~100회가 실제로 청구된다.
    #    --dry-run 의 계약은 '과금 리소스를 만들지 않는다' 이므로 여기서 끊고,
    #    시드를 복제해 파이프라인 뒷단(포맷·업로드·학습 인자)이 밟히는지만 확인한다.
    if cfg.runtime.dry_run:
        n_seed = cfg.data.dry_run_seed_samples
        seeds = td.load_seed_examples(n_seed, token=_hf_token())
        rows = [td.to_messages(s) for s in seeds]
        if not rows:
            return []
        stub = [rows[i % len(rows)] for i in range(n_synth)]
        print(f"  dry-run — Bedrock 합성을 건너뜁니다(과금). 시드 {len(rows)}건을 "
              f"{n_synth}건으로 복제해 형식만 검증합니다.")
        return stub

    model_id = cfg.aws.bedrock_model_id or config.BEDROCK_CLAUDE_MODEL_ID
    if not model_id or "claude" not in model_id:
        raise RuntimeError(
            f"aws.bedrock_model_id = {model_id!r} — inference-profile ID 여야 합니다"
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
    print(f"  채택된 합성 예시: {len(examples)}건")
    return [ex.messages for ex in examples]


def stage_train(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                *, force: bool = False) -> dict[str, Any]:
    """SFT 학습 잡 제출 + 완료 대기 → model_data(S3 아티팩트)."""
    if state.get("model_data") and not force:
        log.info(f"[train] skip — model_data 가 이미 있습니다: {state.get('model_data')}\n"
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

    # 🔴 dry-run 에서도 출력한다 — '무엇이 얼마에 만들어질지'를 미리 보는 것이 dry-run 의 목적이다.
    print_billing_preview(
        what=f"Training Job ({course.train_job_prefix}-*)",
        instance_type=tr.instance_type, region=_region(),
        cap_hours=tr.max_runtime_hours)

    if cfg.runtime.dry_run:
        log.info(f"[train] dry-run — 학습 잡을 제출하지 않습니다.\n"
              f"        entry={course.train_entry} source_dir={course.scripts_dir}\n"
              f"        instance={tr.instance_type} runtime_limit={tr.max_runtime_hours}h "
              f"channels={[c[0] for c in channels] or '(없음)'}\n"
              f"        hyperparameters={json.dumps(hyperparameters, ensure_ascii=False)}")
        model_data = f"{DRY_PREFIX}{course.train_job_prefix}/model.tar.gz"
        state.set(model_data=model_data)
        state.mark_stage("train")
        return {"model_data": model_data}

    ctx = aws_context(cfg, state)

    # 🔴 제출 전에 이전 실행이 남긴 Job 을 확인한다(중복 제출 = 중복 과금).
    resumed = _resume_or_submit_guard(state, "training_job", label="train", cfg=cfg, force=force)
    if resumed:
        state.set(model_data=resumed)
        state.mark_stage("train")
        return {"model_data": resumed}

    log.info(f"[train] {course.key}: {course.train_entry} → {tr.instance_type}")
    job = _submit_training_job(
        cfg=cfg, ctx=ctx, base_job_name=course.train_job_prefix,
        entry_script=course.train_entry, source_dir=course.scripts_dir,
        hyperparameters=hyperparameters, max_runtime_hours=tr.max_runtime_hours,
        input_channels=channels)
    state.set(training_job=job.training_job_name)

    status = _wait_training_job(job, poll_seconds=cfg.runtime.poll_seconds)
    if status != "Completed":
        raise RuntimeError(
            f"학습 잡이 {status} 상태입니다 ({job.training_job_name}). CloudWatch 로그를 확인하세요.\n"
            "  Stopped 이고 FailureReason 이 비어 있으면 MaxRuntimeExceeded 입니다 — "
            "config.yaml 의 training.max_runtime_hours 를 올리세요.")
    model_data = job.model_artifacts.s3_model_artifacts
    log.info(f"[train] 완료 — model_data: {model_data}")
    state.set(model_data=model_data)
    state.mark_stage("train")
    return {"model_data": model_data}


def _model_id() -> str:
    from common import config
    return config.DEFAULT_MODEL_ID


def stage_grpo(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """SFT 산출물을 base 로 이어받아 GRPO 정련 → grpo_model_data (그리고 model_data 갱신).

    🔴 추출·분류 코스만 지원한다. reward 를 프로그램으로 채점할 수 있어야 GRPO 가 의미가 있고,
       요약·자유서술은 채점 기준이 애매해 이 kit 이 제공하지 않는다.
    """
    kind = course.grpo_reward_kind
    if not kind:
        # 보통은 resolve_stages 가 먼저 막는다(명시 요청은 거부, 'all' 은 계획에서 제외).
        # 스테이지 함수를 직접 호출하는 경로를 위한 방어선 — 같은 이유를 같은 문구로 낸다.
        raise RuntimeError(unsupported_reason(course, "grpo") or "grpo 미지원")
    if state.get("grpo_model_data") and not force:
        log.info(f"[grpo] skip — grpo_model_data 가 이미 있습니다: {state.get('grpo_model_data')}\n"
              "       다시 돌리려면 --force.")
        return {"grpo_model_data": state.get("grpo_model_data")}

    sft_model_data = state.get("model_data")
    if not sft_model_data:
        raise RuntimeError(_missing(course, "model_data", "train", state))

    # 🔴 프롬프트 준비보다 **먼저** 확인한다. _grpo_prompts 가 Bedrock 합성을 돌리므로,
    #    이미 제출된 Job 이 있는데 여기까지 오면 토큰당 과금이 또 발생한다(실측 사고).
    if not cfg.runtime.dry_run:
        resumed = _resume_or_submit_guard(state, "grpo_job", label="grpo", cfg=cfg, force=force)
        if resumed:
            state.set(grpo_model_data=resumed, model_data=resumed)
            state.mark_stage("grpo")
            return {"grpo_model_data": resumed}

    g = cfg.grpo
    grpo_path = os.path.join(data_dir(course, cfg), "grpo_train.jsonl")

    from common import grpo_data as gd

    # 🔴 data 스테이지와 같은 규칙 — 파일이 있으면 다시 만들지 않는다.
    #    prompt_source=synth 면 이 준비 과정이 Bedrock 을 부르므로(토큰당 과금),
    #    중단 후 재실행할 때마다 합성이 다시 도는 것은 그대로 요금이다.
    if os.path.isfile(grpo_path) and not force:
        n_prompts = sum(1 for _ in open(grpo_path, encoding="utf-8"))
        log.info(f"[grpo] 프롬프트 재사용 — {grpo_path} ({n_prompts}건). "
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
        log.info(f"[grpo] dry-run — 학습 잡을 제출하지 않습니다 (prompts {n_prompts}건, "
              f"source={g.prompt_source}, runtime_limit={g.max_runtime_hours}h)")
        grpo_model_data = f"{DRY_PREFIX}{course.grpo_job_prefix}/model.tar.gz"
        state.set(grpo_model_data=grpo_model_data, model_data=grpo_model_data)
        state.mark_stage("grpo")
        return {"grpo_model_data": grpo_model_data}

    from common import aws_utils
    ctx = aws_context(cfg, state)
    key = f"{cfg.data.s3_prefix}/{course.key}/grpo/train.jsonl"
    train_s3 = aws_utils.upload_if_changed(grpo_path, ctx.bucket, key, region=ctx.region)

    # 🔴 리전 가드: 상태 파일 값은 리전을 바꿔도 남으므로 옛 리전 버킷을 가리킬 수 있다.
    #    학습 잡도 같은 리전 S3 만 읽는다.
    sft_model_data = aws_utils.ensure_model_data_in_region(
        sft_model_data, ctx.region, job_prefix=course.train_job_prefix)

    log.info(f"[grpo] {course.key}: reward_kind={kind}, base={sft_model_data}")
    job = _submit_training_job(
        cfg=cfg, ctx=ctx, base_job_name=course.grpo_job_prefix,
        entry_script="train_grpo.py", source_dir=course.scripts_dir,
        hyperparameters=hyperparameters, max_runtime_hours=g.max_runtime_hours,
        # 🔴 'model' 채널로 SFT 산출물을 마운트하면 컨테이너의 SM_CHANNEL_MODEL 에 풀리고,
        #    train_grpo.py 가 --base_model_dir 로 받아 그것을 base 로 이어서 학습한다(정석 SFT→GRPO).
        input_channels=[("train", train_s3), ("model", sft_model_data)])
    state.set(grpo_job=job.training_job_name)

    status = _wait_training_job(job, poll_seconds=cfg.runtime.poll_seconds)
    if status != "Completed":
        raise RuntimeError(f"GRPO 잡이 {status} 상태입니다 ({job.training_job_name}). "
                           "CloudWatch 로그를 확인하세요.")
    grpo_model_data = job.model_artifacts.s3_model_artifacts
    log.info(f"[grpo] 완료 — grpo_model_data: {grpo_model_data}")
    # deploy 는 model_data 를 서빙한다 → GRPO 결과를 배포하도록 갱신하고, SFT 와 비교할 수 있게
    # grpo_model_data 는 따로 보관한다.
    state.set(grpo_model_data=grpo_model_data, model_data=grpo_model_data)
    state.mark_stage("grpo")
    return {"grpo_model_data": grpo_model_data, "model_data": grpo_model_data}


def _grpo_prompts(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                  kind: str) -> list[dict]:
    """GRPO prompt 소스 3종 — 🔴 SFT 데이터를 그대로 쓰면 안 된다.

    같은 데이터를 주면 rollout 이 전부 만점이 되어 advantage ≈ 0 → gradient 가 흐르지 않는다
    (GPU 시간만 쓰고 배우는 게 없다). 같은 분포에서 슬라이스만 나눠도 이 문제는 남는다.
    """
    from common import config, grpo_data as gd

    td = load_track_data(course)
    g = cfg.grpo
    n = g.num_prompts
    n_seed = cfg.data.dry_run_seed_samples if cfg.runtime.dry_run else cfg.data.num_seed_samples

    # 🔴 dry-run 은 Bedrock 을 부르지 않는다(토큰당 과금). synth 소스라도 무료 경로인
    #    holdout 으로 대체해 prompt 형식과 뒷단만 검증한다. 실제 실행에서는 아래 분기를 그대로 탄다.
    if cfg.runtime.dry_run and g.prompt_source == "synth":
        print("  dry-run — GRPO 합성 프롬프트를 건너뛰고 holdout 으로 대체합니다(과금).")
        return gd.from_holdout(os.path.join(data_dir(course, cfg), "train.jsonl"), n, sft_used=n_seed)

    if g.prompt_source == "holdout":
        return gd.from_holdout(os.path.join(data_dir(course, cfg), "train.jsonl"), n, sft_used=n_seed)
    if g.prompt_source == "synth":
        # ⚠️ SFT 합성과 같은 시드를 주면 분포가 또 겹친다 → SFT 미사용 구간을 넘긴다.
        pool = td.load_seed_examples(n_seed + n, token=_hf_token())
        fresh = pool[n_seed:]
        return gd.from_synth(task_instruction=td.TASK_INSTRUCTION,
                             seed_texts=td.seed_texts_for_synth(fresh), n=n,
                             model_id=cfg.aws.bedrock_model_id or config.BEDROCK_CLAUDE_MODEL_ID,
                             region=config.BEDROCK_REGION, to_messages=td.to_messages,
                             kind=kind)   # 난이도 제약을 생성 프롬프트에만 적용
    # failures — 가장 강한 소스지만 eval 결과가 필요하다.
    preds_path = _eval_preds_path(course, cfg)
    if not os.path.isfile(preds_path):
        raise RuntimeError(
            f"grpo.prompt_source='failures' 는 평가 예측이 필요합니다 — {preds_path} 가 없습니다.\n"
            f"  → {_cmd(course.key, 'eval')}\n"
            "  (노트북은 같은 커널의 preds 변수를 썼지만, CLI 는 eval 스테이지가 남긴 파일을 읽습니다.)")
    with open(preds_path, encoding="utf-8") as f:
        saved = json.load(f)
    return gd.from_failures(saved["heldout"], saved["preds"], kind=kind,
                            to_messages=td.to_messages, max_n=n)


def stage_deploy(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
                 *, force: bool = False) -> dict[str, Any]:
    """ModelBuilder 로 real-time endpoint 배포 + invoke 스모크 → endpoint_name."""
    prev_ep = state.get("endpoint_name")
    if prev_ep and not force:
        # 🔴 이름만 보고 끝내지 않는다. Creating 중에 끊겼거나 콘솔에서 지웠을 수 있으므로
        #    실제 상태를 확인한다. 이름에 타임스탬프가 붙어(endpoint_prefix-engine-<epoch>)
        #    재제출하면 **다른 이름의 endpoint 가 하나 더 생기고 둘 다 과금**된다.
        st = _endpoint_status(prev_ep)
        if st == "InService":
            log.info(f"[deploy] skip — endpoint 가 이미 서비스 중입니다: {prev_ep}")
            return {"endpoint_name": prev_ep}
        if st == "Creating":
            log.info(f"[deploy] 🔴 이전 실행이 만든 endpoint 가 생성 중입니다: {prev_ep}\n"
                  "         새로 만들지 않고 InService 까지 기다립니다.")
            _wait_endpoint(prev_ep, poll_seconds=cfg.runtime.poll_seconds)
            return {"endpoint_name": prev_ep}
        if st in ("Failed", "RollingBack", "Deleting", "OutOfService"):
            raise RuntimeError(
                f"이전 endpoint 가 {st} 상태입니다({prev_ep}).\n"
                "  CloudWatch 로그로 원인을 확인하고, 정리하려면 --stages cleanup 을 먼저 실행하세요.")
        if st is None:
            log.info(f"[deploy] state 의 endpoint '{prev_ep}' 가 존재하지 않습니다(삭제됨). 새로 만듭니다.")
        else:
            log.info(f"[deploy] skip — endpoint_name 이 이미 있습니다: {prev_ep} ({st})\n"
                  "         새로 띄우려면 --force (⚠️ 이전 endpoint 는 계속 과금됩니다 — cleanup 먼저).")
            return {"endpoint_name": prev_ep}

    model_data = state.get("model_data")
    if not model_data:
        raise RuntimeError(_missing(course, "model_data", "train", state))

    engine = cfg.serving.engine
    endpoint_name = f"{course.endpoint_prefix}-{engine}-{int(time.time())}"

    from common import config, dlc

    serve_image = dlc.resolve_serving_image(_region(), engine)
    if not serve_image:
        raise RuntimeError(f"{engine} 서빙 이미지 해석 실패 — config.yaml serving.images 또는 "
                           f".env 의 *_IMAGE_URI 를 확인하세요: {dlc.AVAILABLE_IMAGES_URL}")

    # 🔴 엔진별 env 키는 dlc.serving_env() 가 유일한 출처다. SM_VLLM_* / SM_SGLANG_* / OPTION_* 를
    #    손으로 쓰면 값 하나를 바꿀 때 한 분기를 빼먹는다(실측: max_num_seqs 를 vLLM 분기에만 넣고
    #    LMI 분기를 놓쳐 OOM 이 재발했다).
    serve_env = dlc.serving_env(
        engine,
        max_model_len=course.serve_max_model_len,
        max_num_seqs=cfg.serving.max_num_seqs,
        # 문자열 그대로 넘긴다 — float 0.90 을 str() 하면 "0.9" 가 되어 컨테이너 env 표시가 달라진다.
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

    # 🔴 endpoint 는 이 kit 에서 유일한 '상시 과금' 리소스다(학습 잡은 끝나면 멈춘다) →
    #    cap_hours 대신 until_deleted 로 '지울 때까지'를 알린다.
    print_billing_preview(
        what=f"real-time endpoint ({endpoint_name})",
        instance_type=cfg.serving.instance_type, region=_region(),
        until_deleted=True)

    if cfg.runtime.dry_run:
        log.info("[deploy] dry-run — endpoint 를 만들지 않습니다(시간당 과금 리소스).")
        endpoint_name = f"{DRY_PREFIX}{course.endpoint_prefix}-{engine}"
        state.set(endpoint_name=endpoint_name, engine=engine)
        state.mark_stage("deploy")
        return {"endpoint_name": endpoint_name}

    from common import aws_utils
    from sagemaker.serve import ModelBuilder

    ctx = aws_context(cfg, state)
    # 리전 가드 — 상태 파일 값이 옛 리전을 가리키면 같은 리전 최신 산출물로 교체한다.
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
              instance_type=cfg.serving.instance_type, wait=False)  # 비동기 — 끊겨도 배포는 계속된다
    state.set(endpoint_name=endpoint_name, engine=engine)
    log.info(f"[deploy] 생성 중: {endpoint_name}")
    _print_console_links(ctx.region, endpoint_name=endpoint_name)

    _wait_endpoint(endpoint_name, poll_seconds=cfg.runtime.poll_seconds)
    _deploy_smoke(course, cfg, endpoint_name)
    state.mark_stage("deploy")
    return {"endpoint_name": endpoint_name}


def _wait_endpoint(endpoint_name: str, *, poll_seconds: int) -> None:
    """Creating → InService 대기. Failed 면 FailureReason 을 함께 보여 준다."""
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
                "  🔴 'did not pass the ping health check' 만 보이면 대개 CUDA OOM 입니다 — "
                "실제 torch.OutOfMemoryError 는 CloudWatch endpoint 로그에만 남습니다. "
                "config.yaml 의 serving.max_num_seqs 를 낮추거나 더 큰 GPU 인스턴스를 쓰세요.")
        time.sleep(poll_seconds)


def _deploy_smoke(course: CourseSpec, cfg: PipelineConfig, endpoint_name: str) -> None:
    """배포 직후 최소 호출 — 실제로 응답하는지, 잘리지 않는지 확인."""
    td = load_track_data(course)
    region = _region()
    if course.multimodal:
        # samples/ 에 넣어 둔 영수증으로 즉시 확인한다(데이터셋 다운로드 없음 — 실측 0.0초).
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
        raise RuntimeError("빈 응답입니다 — CloudWatch endpoint 로그를 확인하세요.")


def stage_eval(course: CourseSpec, cfg: PipelineConfig, state: StateStore,
               *, force: bool = False) -> dict[str, Any]:
    """held-out 평가 — endpoint 로 예측을 모아 코스별 지표를 출력한다.

    🔴 합성 데이터로 평가하면 teacher 모방도를 재는 것에 그친다. 학습에 쓴 앞 구간을 건너뛴
       시드 슬라이스만 쓴다(stage_data 가 eval.jsonl 로 남겨 둔 것).
    """
    if not course.has_eval_stage:
        # 🔴 실패로 다루지 않는다(grpo 와 다른 점). eval 이 없는 것은 이 코스의 검증 방식이 다르다는
        #    뜻이고, `--stages deploy,eval` 은 다섯 코스에 공통으로 안내하는 명령이다 — 여기서 예외를
        #    던지면 deploy 가 성공했는데도 종료 코드가 1이 되어 뒤 작업(스크립트·CI)이 실패로 읽는다.
        log.info("[eval] " + (unsupported_reason(course, "eval") or "eval 미지원"))
        return {}

    endpoint_name = state.get("endpoint_name")
    if not endpoint_name:
        raise RuntimeError(_missing(course, "endpoint_name", "deploy", state))

    _, _, n_eval = _counts(cfg)
    td = load_track_data(course)
    heldout = _load_heldout(course, cfg, td, n_eval)
    log.info(f"[eval] {course.key}: held-out {len(heldout)}건 → {endpoint_name}")

    if cfg.runtime.dry_run or _is_dry_value(endpoint_name):
        log.info("[eval] dry-run — endpoint 를 호출하지 않습니다(호출당 과금 + 실제 endpoint 필요).")
        state.mark_stage("eval")
        return {"n": len(heldout), "dry_run": True}

    region = _region()
    max_tokens = course.gen_max_tokens

    def predict(ex: dict) -> tuple[str, str | None]:
        # vllm/sglang/lmi 공통 — messages 를 그대로 보내면 서버가 chat template 을 적용한다.
        msgs = _fold_system_messages(td.SYSTEM_PROMPT, ex["input"])
        return invoke_chat(endpoint_name, msgs, region=region,
                          max_tokens=max_tokens, temperature=0.0)  # 재현성 위해 결정론적

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=cfg.evaluation.workers) as pool:
        results = list(pool.map(predict, heldout))   # map 은 순서 보존 → heldout[i] ↔ preds[i]
    preds = [r[0] for r in results]
    truncated = sum(1 for r in results if r[1] == "length")
    print(f"  예측 {len(preds)}건 (workers={cfg.evaluation.workers})")
    if truncated:
        # finish_reason 이 유일한 절단 신호다 — 여기서 세어 두지 않으면 지표가 왜 낮은지 모른다.
        print(f"  ⚠️ {truncated}/{len(preds)}건이 max_tokens({max_tokens})에서 잘렸습니다 "
              "→ 지표가 실제보다 낮게 나옵니다(TrackSpec.gen_max_tokens 를 올리세요).")

    scores = _score(course, cfg, td, heldout, preds)
    scores["truncated"] = truncated
    log.info("[eval] scores: " + json.dumps(scores, ensure_ascii=False, sort_keys=True))

    # GRPO 'failures' 소스가 읽을 수 있게 남긴다(노트북은 같은 커널의 변수를 썼다).
    with open(_eval_preds_path(course, cfg), "w", encoding="utf-8") as f:
        json.dump({"endpoint_name": endpoint_name, "heldout": heldout, "preds": preds,
                   "scores": scores}, f, ensure_ascii=False, indent=2)
    print(f"  예측 저장: {_eval_preds_path(course, cfg)}  (grpo.prompt_source='failures' 가 읽습니다)")
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
            f"시드가 {len(pool)}건뿐이라 학습 구간({n_seed}건) 뒤에 남는 예시가 없습니다 — "
            "config.yaml 의 data.num_seed_samples 를 줄이거나 더 큰 시드를 쓰세요.")
    return heldout


def _score(course: CourseSpec, cfg: PipelineConfig, td: Any,
           heldout: list[dict], preds: list[str]) -> dict[str, Any]:
    """코스별 지표 — common/eval_utils.py 를 그대로 쓴다(지표 구현을 복제하지 않는다)."""
    from common import config, eval_utils

    kind = course.track.eval_kind
    if kind == "extraction":
        # gold = 파싱된 {'name','arguments'} dict
        pairs = [(pred, json.loads(ex["output"])) for pred, ex in zip(preds, heldout)]
        return dict(eval_utils.eval_extraction(pairs))
    if kind == "classification":
        # 🔴 라벨 이름은 track_data 헬퍼로 가져온다 — PolyAI/banking77 을 직접 부르면
        #    스크립트 기반 데이터셋이라 datasets>=5.0.0 에서 실패한다.
        label_names = td.load_label_names(token=_hf_token())
        pairs = [(pred, ex["output"]) for pred, ex in zip(preds, heldout)]
        return dict(eval_utils.eval_classification(pairs, label_names))

    # summarization / domain_qa — ROUGE + (선택) Bedrock LLM-judge
    scores: dict[str, Any] = dict(eval_utils.eval_rouge(
        [(pred, ex["output"]) for pred, ex in zip(preds, heldout)]))
    limit = cfg.evaluation.judge_max_examples
    if limit <= 0:
        print("  (evaluation.judge_max_examples=0 → LLM-judge 생략)")
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
    """endpoint + endpoint-config + model 삭제 (🔴 시간당 과금은 endpoint 에서 발생).

    🔴 model 이름은 endpoint_name 과 다르다 — ModelBuilder 가 model-42c30d1e 같은 임의 이름을
       자동 생성한다. endpoint_name 으로 지우려 하면 model 만 조용히 남는다(실측).
       그래서 endpoint-config 에서 실제 ModelName 을 먼저 조회한다(삭제하면 조회 불가 → 순서 중요).
    🔴 삭제 순서는 endpoint → config → model (사용 중이면 삭제가 거부된다).
    """
    endpoint_name = state.get("endpoint_name")
    if not endpoint_name:
        log.info("[cleanup] 지울 endpoint 가 상태에 없습니다 — 이미 정리됐거나 배포하지 않았습니다.")
        print(f"          계정에 남은 것이 있는지 확인: prefix '{course.endpoint_prefix}' "
              "(콘솔 또는 --force 로 prefix 일괄 정리)")
        if not force:
            return {}
    if cfg.runtime.dry_run or _is_dry_value(endpoint_name):
        log.info(f"[cleanup] dry-run — 삭제할 실제 리소스가 없습니다 ({endpoint_name}).")
        state.clear("endpoint_name")
        state.mark_stage("cleanup")
        return {}

    import boto3
    sm = boto3.client("sagemaker", region_name=_region())
    deleted: list[str] = []

    targets: list[str] = [endpoint_name] if endpoint_name else []
    if force:
        # 여러 번 배포했다면 상태 파일은 마지막 것만 가리킨다 → prefix 로 훑어 잔여 리소스도 정리.
        listed = [e["EndpointName"]
                  for e in sm.list_endpoints(NameContains=course.endpoint_prefix)["Endpoints"]]
        targets += [n for n in listed if n not in targets]
        log.info(f"[cleanup] --force: prefix '{course.endpoint_prefix}' 로 찾은 endpoint {listed}")

    for name in targets:
        model_names: list[str] = []
        try:
            cfg_desc = sm.describe_endpoint_config(EndpointConfigName=name)
            model_names = [v["ModelName"] for v in cfg_desc.get("ProductionVariants", [])
                           if v.get("ModelName")]
        except Exception as e:  # noqa: BLE001
            print(f"  endpoint-config 조회 생략({name}): {str(e)[:110]}")
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
    log.info(f"[cleanup] 남은 이 코스 endpoint: {remaining or 'none ✅ (이 코스 과금 멈춤)'}")
    if remaining and not force:
        print("          --force 를 주면 prefix 로 찾은 잔여 리소스까지 정리합니다.")
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
    """다시 실행할 명령 문자열 — 사용자가 **방금 입력한 형태**로 돌려준다.

    🔴 안내 명령이 실제로 붙여넣어 동작해야 한다. run_extraction.py 로 실행한 사람에게
       `python -m pipelines._common --course ...` 를 보여 주면 한 번 더 번역해야 한다.
    """
    argv0 = os.path.basename(sys.argv[0] or "")
    if argv0.startswith("run_") and argv0.endswith(".py"):
        return f"python pipelines/{argv0} --stages {stages}"
    return f"python -m pipelines._common --course {course_key} --stages {stages}"


def _missing(course: CourseSpec, key: str, producer_stage: str,
             state: StateStore | None = None) -> str:
    """빠진 선행조건을 '무엇이 없고, 무엇을 실행하면 생기는지'로 알려 준다.

    state 를 넘기면 **실제로 보고 있는 파일 경로**를 찍는다 — --state-dir 를 바꿔 쓸 때
    기본 경로를 안내하면 사용자가 엉뚱한 파일을 열어 본다.
    """
    where = state.path if state is not None else os.path.join(STATE_DIR, f"{course.key}.json")
    return (f"선행조건 누락: 상태에 '{key}' 가 없습니다 ({where}).\n"
            f"  → 먼저 실행하세요: {_cmd(course.key, producer_stage)}")


def resolve_stages(requested: str | list[str] | None,
                   course: CourseSpec | None = None) -> list[str]:
    """--stages 값을 정규 순서로 정렬·중복 제거.

    🔴 'all' 에서 빼는 두 단계:
       · cleanup — 방금 만든 endpoint 를 자동으로 지우면 평가·데모를 할 수 없다.
       · grpo    — SFT 로 충분한 경우가 많고, GPU 시간이 한 번 더 든다. 필수가 아니므로
                   기본에서 빼고 원할 때만 요청하게 한다. 'all+grpo' 로 켠다.

    🔴 속도 측정(TTFT/TPOT/ITL)은 여기 없다 — pipelines/run_benchmark.py 가 따로 맡는다.
       파이프라인은 '모델을 만들어 배포하는' 흐름이고 벤치마크는 '이미 있는 endpoint 를 재는'
       일이라 선행조건도 실행 주기도 다르다.

    course 를 주면 그 코스에 없는 스테이지를 걸러낸다. 두 경우를 **다르게** 다룬다:
      · 'all'      → 조용히 뺀다. 'all' 은 "이 코스의 전 단계"라는 뜻이고, 계획 줄에
                     없는 단계가 찍히면 사용자가 그 단계가 돌 것이라 오해한다.
      · 명시 요청  → HARD_GAP_STAGES 면 거부한다(ValueError). `--stages grpo` 를 치고 "skip"
                     한 줄만 보면 오타인지, 선행조건이 빠진 것인지, 애초에 불가능한 것인지
                     알 수 없다. 그 외(eval)는 계획에 남겨 스테이지 함수가 이유를 찍게 한다 —
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
            raise ValueError(f"알 수 없는 스테이지 {unknown} — 허용: {list(STAGE_ORDER)} (또는 'all')")

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

    eval_kind 로 가른다 — 그 값이 이 코스의 지표가 무엇인지 정하는 곳이고(_shared_build),
    새 코스를 넣을 때 지표를 정하면 이 문구도 함께 정해진다.
    """
    return {"summarization": "요약문", "domain_qa": "답변 문장"}.get(course.track.eval_kind, "생성 텍스트")


# 🔴 이름을 직접 적었을 때 **거부**할 스테이지. grpo 만 넣는다 — 없는 코스에서 GRPO 를 요청하는 것은
#    설정 착각이지 오타가 아니고(사람이 GPU 시간 몇 시간을 기대한다), 조용히 통과하면 "정련까지
#    끝냈다"고 믿게 된다. eval 은 반대다: `--stages deploy,eval` 이 다섯 코스 공통 안내 명령이라
#    거부하면 평가가 없는 코스에서만 종료 코드가 2가 되어 CLI 표면이 코스별로 달라진다.
HARD_GAP_STAGES = ("grpo",)


def unsupported_reason(course: CourseSpec, stage: str) -> str | None:
    """이 코스가 이 스테이지를 **가지고 있지 않은** 이유. 지원하면 None.

    노트북 구성이 진실의 근거다 — 요약 코스에 02a_train_grpo 노트북이 없는 것과 같은 이유로
    여기에도 grpo 가 없다. 두 진입점이 다른 단계 목록을 갖게 두면 안 된다.
    """
    if stage == "grpo" and not course.grpo_reward_kind:
        if course.multimodal:
            # 🔴 이 코스는 '자유형 출력이라서'가 아니다 — 정답은 JSON 스키마다. 없는 이유가 구조적이므로
            #    그 사실 그대로 말한다. 텍스트 코스용 문구를 재사용하면 "지표는 ROUGE" 처럼 이 코스에
            #    해당하지 않는 말이 찍힌다(이 코스 eval_kind 는 extraction 이다).
            return (
                f"'{course.key}' 코스에는 grpo 스테이지가 없습니다 — 요청을 거부합니다.\n"
                "  🔴 왜: GRPO 는 rollout 마다 reward 를 프로그램으로 채점해야 하고, 그 rollout 을\n"
                "     만들 prompt 가 필요하다. 이 코스의 prompt 는 **이미지**다 —\n"
                f"     · {course.dir_name}/scripts/ 에 train_grpo.py 가 없다(train_mm.py 만 있다).\n"
                "       그 스크립트는 텍스트 prompt 를 받는 전제로 쓰여 있어 이미지를 실을 자리가 없다.\n"
                "     · prompt 소스 3종이 전부 성립하지 않는다: holdout 은 train.jsonl 을 자르는데 이\n"
                "       코스는 그 파일을 만들지 않고(이미지는 컨테이너 안에서 로드), synth 는\n"
                "       track_data.TASK_INSTRUCTION/seed_texts_for_synth 를 쓰는데 이 코스 어댑터에는\n"
                "       그 둘이 없다(이미지 합성은 별개 문제), failures 는 eval 스테이지의 예측이\n"
                "       필요한데 이 코스에는 eval 스테이지가 없다.\n"
                "  → GRPO 를 제공하는 코스: extraction(JSON 스키마 일치), classification(라벨 일치)\n"
                f"  → 이 코스에서 품질을 올리려면: SFT 를 다시 돌리세요 "
                f"({_cmd(course.key, 'data,train')}) — max_train_samples/epochs 를 올리는 것이\n"
                "     이 코스에서 가장 효과가 큰 손잡이다(config.yaml training 섹션).")
        # 🔴 '무엇을 만드는 코스인지'로 말한다. 이 문구를 한 코스(요약)에 맞춰 쓰면 다른 코스에서
        #    "좋은 요약의 기준" 같은 남의 말이 찍혀, 거부 이유가 이 코스의 것으로 읽히지 않는다.
        output = _free_form_output(course)
        return (
            f"'{course.key}' 코스에는 grpo 스테이지가 없습니다 — 요청을 거부합니다.\n"
            "  🔴 왜: GRPO 는 rollout 마다 reward 를 **프로그램으로** 채점해야 학습이 된다.\n"
            f"     이 코스의 정답은 자유형 {output}이고 지표는 ROUGE + LLM-judge 다 — 무엇이 좋은\n"
            f"     {output}인지가 하나로 정해지지 않는다. 채점 함수를 억지로 만들면 그 함수가\n"
            "     좋아하는 표면형(길이·특정 어휘)으로만 수렴해, 지표는 올라가고 품질은 나빠진다.\n"
            "     LLM-judge 를 reward 로 쓰는 길은 있지만 rollout 당 Bedrock 호출이 붙어\n"
            "     (num_generations=8 x prompts) 비용이 학습보다 커진다 — 이 kit 은 제공하지 않는다.\n"
            "  → reward 를 프로그램으로 채점할 수 있는 코스: "
            "extraction(JSON 스키마 일치), classification(라벨 일치)\n"
            f"  → 이 코스에서 품질을 올리려면: SFT 데이터를 늘리거나 고치세요 "
            f"({_cmd(course.key, 'data,train')})")
    if stage == "eval" and not course.has_eval_stage:
        return (f"'{course.key}' 코스에는 eval 스테이지가 없습니다(04 평가 노트북이 없는 코스) — "
                "배포 스모크에서 정답(samples/ground_truth.json)과 대조하는 것이 검증 지점입니다.")
    return None


def _announce_billing(plan: list[str], cfg: PipelineConfig) -> None:
    """이 실행이 과금 리소스를 만들 예정이면 계획 단계에서 한 줄 알린다.

    🔴 스테이지별 상세·요금은 print_billing_preview 가 **만들기 직전에** 찍는다(유일한 출처).
       여기서 같은 내용을 되풀이하지 않는 이유: data 가 몇 분 도는 동안 화면이 밀려 올라가면
       사람이 위쪽 경고를 다시 보지 않는다. 여기서는 '뒤에 과금 단계가 있다'는 예고만 한다.
    """
    if cfg.runtime.dry_run:
        return
    billable = [s for s in plan if s in ("train", "grpo", "deploy")]
    if billable:
        print(_danger("🔴 이 실행에는 과금 단계가 있습니다") + f": {_bold(', '.join(billable))}.\n"
              "   각 단계 직전에 만들 리소스와 대략 요금을 다시 알립니다.")


def run_stages(course_key: str, stages: str | list[str] | None, cfg: PipelineConfig,
               *, dry_run: bool = False, force: bool = False,
               state_dir: str = STATE_DIR, state_path: str | None = None,
               endpoint_name: str | None = None) -> dict[str, Any]:
    """스테이지들을 순서대로 실행. 첫 실패에서 멈춘다(뒤 단계가 옛 산출물로 도는 것을 막는다).

    dry_run 인자는 호출 편의용이다 — 실제 스위치는 cfg.runtime.dry_run(= DRY_RUN env)이고,
    load_config(dry_run=...) 이 둘을 이미 일치시킨다. 여기서는 어긋났을 때만 경고한다.
    state_path 를 주면(--state) 그 파일을 쓰고 state_dir 는 무시된다.
    """
    courses = load_courses()
    if course_key not in courses:
        raise ValueError(f"알 수 없는 코스 {course_key!r} — 허용: {list(courses)}")
    course = courses[course_key]
    if dry_run and not cfg.runtime.dry_run:
        print("⚠️  dry_run=True 로 호출됐지만 cfg.runtime.dry_run 이 False 입니다 — "
              "load_config(dry_run=True) 로 만든 cfg 를 넘기세요(DRY_RUN env 와 함께 맞춰집니다).")

    state = StateStore(course_key, state_dir=state_dir, dry_run=cfg.runtime.dry_run,
                       path=state_path)
    if endpoint_name:
        # 🔴 스테이지 함수는 (course, cfg, state, force) 만 받는다. endpoint 이름을 인자로 더 흘리는
        #    대신 상태 파일에 넣는다 — deploy 가 쓰는 것과 같은 키라 eval·cleanup 이 이미 읽는다.
        #    부작용이 하나 있다: cleanup 이 이 값을 지운다. 남의 endpoint 를 실수로 지우는 쪽보다
        #    낫다고 보고, 덮어쓸 때 무엇을 덮는지 알린다.
        previous = state.get("endpoint_name")
        if previous and previous != endpoint_name:
            log.warning("--endpoint-name 이 상태 파일의 값을 덮어씁니다: %s → %s",
                        previous, endpoint_name)
        state.set(endpoint_name=endpoint_name)
    plan = resolve_stages(stages, course)   # 이 코스에 없는 단계는 여기서 걸러지거나 거부된다

    print(_dim("=" * 78))
    print(f"{_bold('course')}    : {_bold(course.key)} ({course.dir_name})")
    print(cfg.summary())
    print(state.summary())
    print(f"{_bold('stages')}    : {' → '.join(plan)}"
          + (_warn("   [--force]") if force else ""))
    print(_dim("=" * 78))
    # 🔴 첫 스테이지 전에 알린다 — data 가 몇 분 돈 뒤에 경고해도 이미 늦다.
    _announce_billing(plan, cfg)

    results: dict[str, Any] = {}
    for stage in plan:
        print(f"\n──── {stage} ────")
        try:
            results[stage] = STAGES[stage](course, cfg, state, force=force)
        except Exception as e:  # noqa: BLE001
            log.error("stage %s 실패: %s", stage, e)
            print(f"\n🔴 '{stage}' 스테이지에서 중단했습니다: {e}")
            print(f"   상태는 보존됐습니다({state.path}) — 고친 뒤 그 스테이지부터 다시 실행하세요:")
            print("   " + _cmd(course.key, ",".join(plan[plan.index(stage):])))
            raise StageFailed(f"{stage}: {e}") from e
    print("\n" + "=" * 78)
    print(state.summary())
    if state.get("endpoint_name") and not _is_dry_value(state.get("endpoint_name")):
        print("🔴 endpoint 는 삭제 전까지 시간당 과금됩니다 — 실습이 끝나면:\n"
              "   " + _cmd(course.key, "cleanup"))
    return results


# ===========================================================================
# 6) CLI
# ===========================================================================
def build_parser(default_course: str | None = None) -> argparse.ArgumentParser:
    # 🔴 -h 의 예시는 **사용자가 방금 입력한 명령 형태**로 보여 준다(_cmd 와 같은 이유).
    #    run_summarization.py 를 실행한 사람에게 `python -m pipelines._common --course extraction`
    #    을 보여 주면 한 번 더 번역해야 하고, 코스 이름까지 남의 것이라 그대로 붙여넣으면 틀린다.
    argv0 = os.path.basename(sys.argv[0] or "")
    if argv0.startswith("run_") and argv0.endswith(".py"):
        base = f"python pipelines/{argv0}"
    else:
        base = f"python -m pipelines._common --course {default_course or '<course>'}"
    # 상태 파일 경로도 -h 에 적는다 — 스테이지 간 값이 어디에 남는지 모르면 부분 실행을 신뢰할 수 없다.
    where = os.path.join(STATE_DIR, f"{default_course or '<course>'}.json")
    p = argparse.ArgumentParser(
        description="코스 E2E 파이프라인 (data → train → grpo → deploy → eval → cleanup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            f"  {base} --stages all           # 전 단계(cleanup 제외)\n"
            f"  {base} --stages data,train    # model_data 를 상태 파일에 기록\n"
            f"  {base} --stages deploy,eval   # 그 값을 읽어 배포·평가\n"
            f"  {base} --dry-run              # 과금 리소스 0으로 전 경로 검증\n"
            f"  {base} --stages cleanup       # 🔴 endpoint 삭제(시간당 과금 정지)\n"
            "\n"
            f"스테이지: {' → '.join(STAGE_ORDER)}\n"
            "  data    시드 로드 + grounded 합성 + train/eval JSONL + S3 업로드\n"
            "  train   SFT 학습 잡 제출·대기 → model_data\n"
            "  grpo    SFT 결과를 이어받아 GRPO 정련 (추출·분류 코스만)\n"
            "  deploy  real-time endpoint 배포 + 스모크 → endpoint_name\n"
            "  eval    held-out 평가 (코스별 지표)\n"
            "  cleanup endpoint·config·model 삭제 ('all' 에는 포함되지 않는다)\n"
            "\n"
            f"상태 파일: {where}\n"
            "  스테이지 간 값(model_data / endpoint_name ...)을 여기에 남겨 부분 실행을 잇는다.\n"
            "  --dry-run 은 <course>.dryrun.json 을 따로 쓴다(가짜 산출물이 실제 실행에 섞이지 않게).\n"),
    )
    p.add_argument("--course", default=default_course,
                   help="코스 키 (extraction/classification/summarization/domain_qa/mm_extraction)")
    p.add_argument("--stages", default="all",
                   help=(f"콤마 구분 스테이지. 허용: {','.join(STAGE_ORDER)}. "
                         "기본 all = grpo·cleanup 제외(둘 다 필수가 아니고 GPU 시간·과금이 더 든다). "
                         "GRPO 까지 돌리려면 all+grpo"))
    p.add_argument("--config", default=None, help="config.yaml 경로(기본 <repo>/config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="과금 리소스를 만들지 않고 전 경로를 밟는다(config.yaml runtime.dry_run 을 이긴다)")
    p.add_argument("--force", action="store_true",
                   help="이미 만들어진 산출물이 있어도 스테이지를 다시 실행한다"
                        " (진행 중인 Job·endpoint 는 --force 로도 새로 만들지 않는다 —"
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
        print("--course 를 지정하세요 (또는 코스별 run_<course>.py 를 쓰세요).")
        return 2

    # 🔴 인자 파싱 실패는 밀리초 안에 끝나야 한다 → 무거운 import 는 여기서부터.
    from common.logging_utils import setup_logging

    # 🔴 config 검증 실패는 '설정이 틀렸다'는 사용자 오류다 → 트레이스백이 아니라 메시지로 끝낸다.
    #    load_config 을 아래 try 밖에서 부르면 ConfigError(=ValueError) 가 그 핸들러를 못 만나고
    #    파이썬 기본 트레이스백으로 새어 나간다(실측: --config 에 model.size=NOPE 를 주면
    #    허용값을 담은 좋은 메시지가 5프레임 스택 밑에 묻혔다).
    try:
        cfg = load_config(args.config, dry_run=True if args.dry_run else None)
    except (ValueError, RuntimeError, FileNotFoundError, TypeError, AttributeError) as e:
        print(f"🔴 {e}")
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
        return 1                      # run_stages 가 이미 원인·재실행 명령을 출력했다
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # 스테이지 진입 전 실패(알 수 없는 코스/스테이지, config 오류 등) — 조용히 죽으면 안 된다.
        print(f"🔴 {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
