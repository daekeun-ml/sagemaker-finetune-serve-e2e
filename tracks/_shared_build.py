"""
tracks/_shared_build.py — 4개 트랙 공용 노트북 빌더 (중복 제거의 핵심)

각 트랙은 TrackSpec만 정의하고 build_track(spec)을 호출하면 00~05,99 노트북 7종이 생성된다.
트랙 간 차이는 데이터 어댑터(track_data.py)와 몇 개 문자열(task 이름/프롬프트/예시)뿐이므로
이를 TrackSpec으로 파라미터화한다. 노트북 본문 로직(설치·설정·학습·배포·agentic·cleanup)은 공유.

규약(aws-ml-lab-code): 상단 TL;DR/Why/Pain → 설치(pin) → 설정(플레이스홀더/env) →
데이터 → 학습/배포 → CloudWatch 링크 → 결과확인 → 🔴 cleanup.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class TrackSpec:
    key: str                 # config.TRACKS 키 (extraction/classification/...)
    dir_name: str            # 트랙 디렉토리명 (01_extraction_to_json 등)
    title: str               # 사람이 읽는 트랙 이름
    endpoint_prefix: str     # endpoint 이름 prefix (gemma-extraction 등)
    max_seq_length: int
    use_qlora: bool
    # agentic 예시
    tool_name: str           # Strands @tool 함수명
    tool_doc: str            # tool docstring
    # 🔴 프롬프트는 영어로만 유지한다. 05 노트북의 LANG이 " Reply in {LANG}."을 덧붙여 응답 언어를
    #    바꾸므로 번역본을 따로 관리할 필요가 없다(언어를 늘려도 프롬프트는 그대로).
    agent_system: str        # 오케스트레이터 system prompt (영어)
    smoke_user: str          # 05 에이전트 스모크 사용자 입력 (영어)
    deploy_smoke_user: str   # 03 endpoint 스모크 사용자 입력
    # 🔴 02b 로컬 서빙 예시 입력. deploy_smoke_user보다 '학습 데이터와 같은 형태'로 준다
    #    (추출=tool 스키마 포함, 분류=라벨 후보 포함 등). 비우면 deploy_smoke_user를 쓴다.
    #    실측 교훈: system prompt와 스키마를 빼면 파인튜닝 모델이 일반 챗봇처럼 답해
    #    "학습이 안 된 것처럼" 보인다(같은 모델이 스키마를 주면 정확한 JSON을 냄).
    serve_example_user: str = ""
    # 🔴 서빙 컨텍스트 길이. 학습 길이(max_seq_length)와 **분리**해야 한다(실측 2026-07-30).
    #    학습은 "입력+정답"이 max_seq_length에 들어가도록 자르지만, 서빙은 "입력 + 새로 생성할 토큰"이
    #    모두 컨텍스트에 들어가야 한다. 둘을 같은 값으로 묶으면 긴 입력 트랙에서 평가가 죽는다:
    #    03_summarization 실측 — held-out 프롬프트 median 1370 / max 2006 토큰인데 학습값 2048을
    #    서빙 컨텍스트로 쓰면 (2006 + 256) > 2048 → vLLM이 400(context length exceeded)로 거부.
    #    0이면 max_seq_length * 2 를 쓴다(입력만큼 생성 여유를 둔다는 뜻).
    serve_max_model_len: int = 0
    # 🔴 평가·추론 시 생성할 최대 토큰. 정답이 이보다 길면 예측이 잘려 지표가 구조적으로 과소 측정된다.
    #    03_summarization 실측 — 정답 요약 median 209 / p90 475 / max 964 토큰이라 256으로는
    #    held-out 40%(20/50건)가 잘려 ROUGE-L이 실제보다 낮게 나온다.
    gen_max_tokens: int = 256
    eval_kind: str = "rouge_judge"  # extraction | classification | summarization | domain_qa
    dataset_blurb: str = ""  # 01 노트북에 표시할 시드 데이터셋 설명(이름·라이선스·원본 포맷·파싱)
    grpo_reward_kind: str = ""  # 비어있지 않으면 GRPO 노트북 생성(extraction|classification). reward가 명확한 트랙만.
    # 🔴 02b(로컬 서빙 검증) 노트북이 있는 트랙인지. 05_multimodal은 없으므로 99_cleanup에서
    #    '로컬 모델 정리' 섹션을 빼야 한다(없는 스크립트를 안내하면 안 됨).
    has_local_serve: bool = True
    # 🔴 00_setup 마지막에 안내할 다음 노트북 파일명. 05_multimodal은 합성 단계가 없어
    #    01_data_and_synthetic 대신 01_data_explore 로 이어지므로 트랙별로 달라야 한다
    #    (하드코딩하면 존재하지 않는 파일을 안내하게 된다). 비우면 _next_after_setup()이 정한다.
    next_after_setup: str = ""


def _next_after_setup(s: "TrackSpec") -> str:
    """00_setup 다음 노트북 파일명.

    🔴 하드코딩하면 안 되는 이유: 05_multimodal 트랙은 합성 단계가 없어 01 노트북 이름이
       01_data_explore.ipynb 이다. 01_data_and_synthetic.ipynb 로 고정 안내하면 존재하지 않는
       파일로 이어져 링크가 깨진다. spec에 값이 있으면 그것을, 없으면 트랙 키로 정한다.
    """
    if s.next_after_setup:
        return s.next_after_setup
    return "01_data_explore.ipynb" if s.key == "mm_extraction" else "01_data_and_synthetic.ipynb"


def _md_var(s: "TrackSpec") -> str:
    """이 트랙 전용 model_data %store 키(`endpoint_name`과 같은 이유 — _ep_var 독스트링 참고).

    🔴 model_data도 트랙마다 값이 다르다(트랙별로 다른 학습 잡의 산출물). 전역 키만 쓰면
       마지막에 학습한 트랙의 아티팩트가 다른 트랙 배포/평가에 쓰인다.
    """
    return f"md_{s.key}"


def _resume_cells(s: "TrackSpec") -> list[dict]:
    """세션이 끊긴 뒤 '호출 단계부터' 다시 시작하기 위한 독립 실행 셀.

    🔴 왜 필요한가: 03 노트북은 배포(§1)가 위에 있고 호출(§2~§3)이 아래에 있다. 커널이 끊기면
       아래 셀만 돌릴 수 없다 — import·config·endpoint_name이 전부 사라져 NameError가 난다.
       위 배포 셀을 다시 실행하면 되지만, 그 셀은 model_data/role 해석까지 하므로 불필요하게
       무겁고 (LMI 분기에서는) 엔드포인트를 또 만들 위험도 있다. 그래서 **호출에 필요한 것만**
       모은 셀을 둔다: path·import·config·endpoint_name 복구 + InService 확인.
    """
    return [
        md(
            "### ⏸️ 세션이 끊겼다면 — 여기서부터 이어서 실행\n"
            "커널을 재시작했거나 세션이 끊겼다면 위 배포 셀을 다시 돌릴 필요가 없습니다"
            "(**endpoint는 서버에 그대로 살아 있습니다**). 아래 셀 하나만 실행하면 호출에 필요한 것"
            "(경로·import·`endpoint_name`)이 모두 복구됩니다.\n"
            "> 이미 위에서부터 순서대로 실행했다면 이 셀은 건너뛰어도 되고, 실행해도 무해합니다."
        ),
        code(
            "# ── 세션 재개 전용 (이 셀만 실행하면 아래 호출 셀들이 바로 동작) ──\n"
            "import os, sys, importlib\n"
            "REPO = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
            "for p in (REPO, os.getcwd()):\n"
            "    if p not in sys.path:\n"
            "        sys.path.insert(0, p)\n"
            "from common import config, aws_utils; importlib.reload(config)\n"
            "\n"
            "# 트랙 전용 키 우선 — 전역 endpoint_name 은 다른 트랙이 덮어씁니다.\n"
            f"%store -r {_ep_var(s)}\n"
            "%store -r endpoint_name\n"
            f"endpoint_name = globals().get('{_ep_var(s)}') or globals().get('endpoint_name')\n"
            "assert endpoint_name, (\n"
            "    'endpoint_name 이 없습니다. §1에서 배포하거나, 아래처럼 직접 지정하세요:\\n'\n"
            f"    \"    endpoint_name = '{s.endpoint_prefix}-vllm-...'\")\n"
            "\n"
            "from sagemaker.core.resources import Endpoint\n"
            "ep = Endpoint.get(endpoint_name); ep.refresh()\n"
            "print('endpoint:', endpoint_name, '->', ep.endpoint_status)\n"
            "assert ep.endpoint_status == 'InService', (\n"
            "    f'{ep.endpoint_status} 상태입니다. Creating이면 잠시 뒤 다시 실행하세요.')"
        ),
    ]


def _ep_var(s: "TrackSpec") -> str:
    """이 트랙 전용 endpoint_name %store 키.

    🔴 왜 필요한가 (실측 2026-07-31): `%store`는 트랙·커널·리전을 넘어 공유되는 전역 저장소다.
       여러 트랙을 돌리면 마지막 트랙의 `endpoint_name`이 값을 덮어써, 다른 노트북이 엉뚱한
       엔드포인트를 호출한다. 증상이 배포 문제처럼 보여 진단이 어렵다 — 요약 노트북이 멀티모달
       엔드포인트(max_model_len=2048)를 불러 "maximum context length is 2048" 400 에러가 났다
       (요약 엔드포인트는 4096이라 정상인데도). 트랙별로 키를 분리하면 충돌이 구조적으로 불가능하다.
       (같은 이유로 train_path는 %store를 아예 쓰지 않고 트랙 로컬 파일을 쓴다.)
    """
    return f"ep_{s.key}"


def _stream_default(s: "TrackSpec") -> bool:
    """실시간 추론 셀에서 스트리밍을 기본으로 켤지.

    🔴 스트리밍이 이득인 건 '긴 자유서술'(요약·QA)이다. 추출(JSON)·분류(라벨)는 응답이
       완성돼야 파싱/사용이 가능하고 애초에 짧아서, 조각을 흘려도 체감 이득이 없다.
       기준은 gen_max_tokens — 트랙별 정답 길이 분포에서 정한 값이라 그대로 쓸 수 있다.
    """
    return s.eval_kind in ("summarization", "domain_qa", "rouge_judge")


def _serve_len(s: "TrackSpec") -> int:
    """서빙 컨텍스트 길이. 지정 없으면 학습 길이의 2배(입력만큼 생성 여유).

    🔴 학습 길이를 그대로 쓰면 안 되는 이유: 학습은 '입력+정답'을 자르지만 서빙은 '입력+생성'이
       컨텍스트에 함께 들어간다. 긴 입력 트랙(요약)에서 (프롬프트 + max_tokens) > 컨텍스트가 되어
       vLLM이 400(context length exceeded)으로 거부한다(실측).
    """
    return s.serve_max_model_len or s.max_seq_length * 2


# ---- 셀 헬퍼 (nbformat 없이 표준 .ipynb JSON) ----
def _src(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else [""]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_PATH = (
    "import os, sys\n"
    "# 리포 루트를 path에 추가해 common/ 와 트랙 로컬 모듈을 import\n"
    "REPO = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
    "sys.path.insert(0, REPO)\n"
    "sys.path.insert(0, os.getcwd())"
)


def _pip_install_code(pkgs: list[str], comment: str = "") -> str:
    """추가 패키지 설치 셀 소스 — uv 우선, pip 폴백 (00_setup과 동일 관용구).

    %pip 매직 대신 subprocess로 현재 커널 인터프리터(sys.executable)에 설치한다.
    uv가 있으면 `uv pip install --python <interp>`(빠름), 없으면 `pip install`.
    """
    quoted = ", ".join(repr(p) for p in pkgs)
    head = f"# {comment}\n" if comment else ""
    return (
        head +
        "import shutil, subprocess, sys\n"
        f"_pkgs = [{quoted}]\n"
        "if shutil.which('uv'):\n"
        "    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '-q', *_pkgs], check=True)\n"
        "else:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_pkgs], check=True)\n"
        "print('installed:', _pkgs)"
    )


def header(title: str, tldr: str, why: str, pain: str) -> dict:
    return md(
        f"# {title}\n\n"
        f"**TL;DR** — {tldr}\n\n"
        f"**Why** — {why}\n\n"
        f"**기존 Pain Point** — {pain}\n\n"
        "> 🔴 실제 실행 시 AWS 자격증명·GPU·엔드포인트 과금이 발생합니다. 먼저 `DRY_RUN=1`로 파이프라인을 검증하세요."
    )


# ---------------------------------------------------------------------------
def _c00(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"00 · 환경 설정 (Setup) — {s.title}",
            "AWS/HF 자격증명·리전·역할을 확인하고 의존성을 설치합니다.",
            "이후 모든 노트북이 여기서 만든 config/자격증명을 공유합니다.",
            "노트북마다 설정을 반복하면 값이 서로 어긋나는 config drift가 발생하므로, 모든 설정을 common/config.py 한 곳에서 관리합니다.",
        ),
        md(
            "## 1. 의존성 설치 (uv 권장 · pip 폴백)\n"
            "재현 가능한 환경을 위해 의존성 관리는 `uv`를 우선 사용하고, 설치되어 있지 않은 경우 `pip`으로 폴백합니다. "
            "`uv`가 없다면 다음 명령으로 먼저 설치하세요: `curl -LsSf https://astral.sh/uv/install.sh | sh`\n"
            "- 의존성을 최신 버전으로 올리려면 `uv lock --upgrade` 후 `uv sync`를 실행합니다. 특정 패키지만 갱신할 때는 `uv lock --upgrade-package transformers`처럼 지정합니다."
        ),
        code(
            "import shutil, subprocess, sys, os\n"
            "REPO = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
            "if shutil.which('uv'):\n"
            "    # uv가 현재 노트북 커널 인터프리터(sys.executable)에 pyproject.toml 의존성 설치. 최신은 uv lock --upgrade.\n"
            "    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '-r', 'pyproject.toml'], cwd=REPO, check=True)\n"
            "else:\n"
            "    print('uv not found -> pip fallback. (recommended: curl -LsSf https://astral.sh/uv/install.sh | sh)')\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',\n"
            "                    os.path.join(REPO, 'requirements.txt')], check=True)"
        ),
        md(
            "## 2. 경로 설정 — common/ import\n"
            "노트북은 트랙 하위 디렉토리에서 실행되므로, 리포 루트와 트랙 로컬 경로를 `sys.path`에 추가해야 "
            "`common/` 공용 모듈과 트랙별 `track_data` 모듈을 import할 수 있습니다."
        ),
        code(SETUP_PATH),
        md(
            "## 3. 환경변수 (플레이스홀더 — 시크릿 하드코딩 금지)\n"
            "리전·모델 ID 등 실행에 필요한 값을 환경변수로 설정합니다. HF 토큰이나 AWS 자격증명 같은 시크릿을 "
            "노트북에 직접 하드코딩하면 유출 위험이 있으므로, 아래에서는 플레이스홀더만 두고 실제 값은 env로 주입합니다. "
            "`setdefault`를 사용해 이미 설정된 값이 있으면 덮어쓰지 않습니다."
        ),
        code(
            "# 리전 — GPU 용량 부족(InsufficientInstanceCapacity)이면 이 값만 바꿔 재시도.\n"
            "os.environ.setdefault('AWS_REGION', 'us-west-2')\n"
            "# 🔴 boto3는 AWS_REGION 을 '기본 세션'에서 읽지 않습니다 → 둘을 같이 세팅해야\n"
            "#    SDK 내부가 ~/.aws/config 의 다른 리전으로 폴백하지 않습니다(실측).\n"
            "os.environ['AWS_DEFAULT_REGION'] = os.environ['AWS_REGION']\n"
            "\n"
            "# 모델 크기 — 'E4B'(기본, 단일 GPU) | '12B' | '26B-A4B'\n"
            "os.environ.setdefault('MODEL_SIZE', 'E4B')\n"
            "os.environ.setdefault('DRY_RUN', '1')   # 파이프라인 검증용. 실제 실행 시 '0'\n"
            "\n"
            "# (선택) 필요할 때만 주석 해제\n"
            "# os.environ['MODEL_ID'] = 'google/gemma-4-12B-it'      # 프리셋 대신 직접 지정\n"
            "# os.environ['MODEL_IS_GATED'] = '1'                    # gemma-3/2 등 gated 모델\n"
            "# os.environ['HF_HOME'] = os.path.expanduser('~/hf-cache')   # hf login 을 이 경로로 했다면\n"
            "# os.environ['BEDROCK_CLAUDE_MODEL_ID'] = 'us.anthropic.claude-...'"
        ),
        md(
            "> **위 (선택) 항목 참고**\n"
            "> - **HF 토큰**: gemma-4는 ungated라 필요 없습니다. gated 모델을 쓸 때만 `MODEL_IS_GATED=1` + 토큰. "
            "토큰은 env `HF_TOKEN` 또는 `hf auth login` 저장분을 config가 자동으로 찾습니다. "
            "단 `hf auth login`을 커스텀 경로로 했다면 `HF_HOME`도 같이 맞춰야 합니다.\n"
            "> - **Bedrock 모델 ID**: inference-profile 접두사(`us.`/`global.` 등)가 붙은 정확한 ID여야 합니다 — "
            "콘솔 모델 상세 페이지에서 확인하세요."
        ),
        md(
            "## 4. 로깅 설정 (1회) & config 로드\n"
            "로깅 핸들러는 애플리케이션 진입점에서 한 번만 구성하는 것이 원칙입니다. 라이브러리 코드(common/*)가 "
            "핸들러를 직접 설정하면 로그가 중복 출력되거나 사용자 설정을 덮어쓰기 때문에, common/* 모듈은 핸들러를 "
            "건드리지 않고 노트북에서 `setup_logging()`으로 1회 구성합니다."
        ),
        code(
            "from common.logging_utils import setup_logging, get_logger\n"
            "setup_logging()          # LOG_LEVEL env 존중(기본 INFO), 멱등\n"
            "log = get_logger('nb')   # gemma_e2e.nb\n"
            "log.info('Logging configured')"
        ),
        code(
            "import importlib\n"
            "from common import config; importlib.reload(config)\n"
            f"TRACK = config.TRACKS['{s.key}']\n"
            "print('TRACK          :', TRACK.name)\n"
            "print('seed_dataset   :', TRACK.seed_dataset)\n"
            "print('MODEL_ID       :', config.DEFAULT_MODEL_ID)\n"
            "print('AWS_REGION     :', config.AWS_REGION)\n"
            "print('DRY_RUN        :', config.is_dry_run())\n"
            "print('HF_TOKEN set   :', bool(config.get_hf_token()))"
        ),
        code(
            "import boto3\n"
            "try:\n"
            "    ident = boto3.client('sts', region_name=config.AWS_REGION).get_caller_identity()\n"
            "    print('AWS account:', ident['Account'])\n"
            "except Exception as e:\n"
            "    print('WARNING: AWS credential check failed - run aws configure or set a role:', e)\n"
            "\n"
            "# 🔴 리전 일치 확인: boto3 '기본 세션'(리전 미지정 클라이언트가 쓰는 것)도 같은 리전인지 본다.\n"
            "#    다르면 SDK 내부가 region_name 없이 만드는 클라이언트가 옛 리전에 붙을 수 있다.\n"
            "_default = boto3.Session().region_name\n"
            "print('boto3 default  :', _default)\n"
            "if _default != config.AWS_REGION:\n"
            "    print(f\"⚠️  boto3 기본 리전({_default}) != AWS_REGION({config.AWS_REGION}).\")\n"
            "    print('    위 셀의 AWS_DEFAULT_REGION 설정이 실행됐는지 확인하세요(커널 재시작 후 순서대로 실행).')\n"
            "    print(f'    ~/.aws/config 의 [default] region 이 {_default} 로 돼 있어도 이 env가 우선합니다.')\n"
            "else:\n"
            "    print('✅ 리전 일치 — SDK 내부 클라이언트도 같은 리전을 씁니다.')"
        ),
        md(
            "## 5. SageMaker 세션 & 역할\n"
            "SageMaker 학습 잡과 endpoint 배포에는 세션 객체와 실행 IAM role, 그리고 아티팩트를 저장할 S3 bucket이 "
            "필요합니다. role은 `config.resolve_sagemaker_role()`이 **env `SAGEMAKER_ROLE_ARN` → `get_execution_role()`("
            "Studio/NB) → IAM 자동 탐지(AmazonSageMaker-ExecutionRole-*)** 순으로 해석하므로, IAM user로 로컬 실행해도 "
            "계정의 실행 role을 자동으로 찾습니다. role ARN을 코드나 `.env`에 하드코딩하지 마세요 — 특정 계정에 종속되고 "
            "이식성이 떨어집니다(명시가 필요하면 셸에서 `export SAGEMAKER_ROLE_ARN=...`). 여기서 확인한 role/bucket은 "
            "`%store`로 저장해 이후 노트북에서 재사용합니다."
        ),
        code(
            "# 🔴 sagemaker SDK v3(3.x): Session/get_execution_role 는 sagemaker.core.helper.session_helper 로 이동.\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            "role = config.resolve_sagemaker_role(sess)   # env → get_execution_role → IAM 자동 탐지\n"
            "bucket = config.S3_BUCKET or sess.default_bucket()\n"
            "print('role  :', role); print('bucket:', bucket)\n"
            "%store role\n%store bucket"
        ),
        md(f"✅ 환경 설정이 끝났습니다. 다음은 **{_next_after_setup(s)}**로 이어집니다."),
    ]


def _c01(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"01 · 데이터 준비 & grounded 합성 — {s.title}",
            "공개 시드 데이터를 소량 샘플링한 뒤 Bedrock Converse로 grounded 합성 데이터를 생성하여 학습용 JSONL을 구성합니다.",
            "적은 수의 시드로 파이프라인을 빠르게 검증하고, 도메인에 맞춘 합성 데이터로 품질을 끌어올리는 production 패턴을 실제로 따라 해 봅니다.",
            "라벨 데이터가 부족하면 파인튜닝 자체가 어렵습니다. 그렇다고 근거 없이 합성하면 hallucination이 섞여 들어가므로, seed에 grounded된 데이터만 생성하고 critique 단계로 저품질 샘플을 걸러 냅니다.",
        ),
        code(SETUP_PATH),
        code(
            "import importlib\n"
            "from common import config, gemma_format; importlib.reload(config)\n"
            "from common.synth import bedrock_synth as bs\n"
            "import importlib, track_data as td; importlib.reload(td)\n"
            f"TRACK = config.TRACKS['{s.key}']   # 이 트랙의 설정(시드 데이터셋 등)\n"
            "NUM_SEED = 8 if config.is_dry_run() else config.NUM_SEED_SAMPLES\n"
            "NUM_SYNTH = 6 if config.is_dry_run() else config.NUM_SYNTHETIC\n"
            "print(f'track={TRACK.name}, seed_dataset={TRACK.seed_dataset}')\n"
            "print(f'seed={NUM_SEED}, synthetic={NUM_SYNTH}, dry_run={config.is_dry_run()}')"
        ),
        md(
            "## 1. 시드 로드 & 특성 확인\n"
            "합성 데이터의 품질은 시드 데이터의 특성을 얼마나 잘 반영하느냐에 달려 있습니다. 먼저 공개 시드를 "
            "로드해 입력·출력의 형태와 길이, 어투 등을 눈으로 확인하고, 이 특성이 이후 합성·학습 단계로 "
            "이어지도록 합니다.\n\n"
            + (s.dataset_blurb + "\n\n" if s.dataset_blurb else "")
            + "아래 셀은 원본 데이터셋을 이 트랙의 어댑터(`track_data.py`)로 파싱해 표준 "
            "`{\"input\", \"output\"}` 형태로 만든 뒤, 첫 샘플을 출력합니다. `load_seed_examples`가 "
            "원본 row를 어떻게 파싱하는지는 `track_data.py`를 참고하세요."
        ),
        code(
            "seeds = td.load_seed_examples(NUM_SEED, token=config.get_hf_token())\n"
            "print(f'Parsed seeds: {len(seeds)}  (dataset: {TRACK.seed_dataset})')\n"
            "print('--- sample input  ---\\n', seeds[0]['input'][:400])\n"
            "print('--- sample output ---\\n', seeds[0]['output'][:400])"
        ),
        md(
            "## 2. grounded 합성 (Bedrock Converse + critique/refine)\n"
            "시드 텍스트를 근거(grounding)로 제시하고 Bedrock Converse에 합성 데이터를 요청한 뒤, "
            "생성 결과를 다시 모델로 평가(critique)해 groundedness와 relevance 점수가 기준에 미치지 못하는 "
            "샘플을 걸러 내는 critique/refine 루프를 돕니다. 이렇게 하면 시드에서 벗어난 hallucination을 억제할 수 있습니다.\n"
            "생성 배치와 critique는 Bedrock 호출이 I/O 바운드이므로 `max_workers`만큼 **병렬**로 처리되고, "
            "**tqdm 진행바**로 채택된 예시 수가 실시간으로 표시됩니다(throttling이 나면 `max_workers`를 낮추세요).\n"
            "🔴 Bedrock 모델 ID는 env `BEDROCK_CLAUDE_MODEL_ID`로 지정하며, 호출량 기준으로 과금됩니다.\n\n"
            "> 대안: 활발히 유지보수되는 오픈 라이브러리(Kiln native Bedrock / Bespoke Curator via LiteLLM)는 `common/synth/README.md`를 참고하세요. distilabel은 유지보수가 정체되어 사용하지 않습니다."
        ),
        code(
            "assert config.BEDROCK_CLAUDE_MODEL_ID and 'claude' in config.BEDROCK_CLAUDE_MODEL_ID, \\\n"
            "    'BEDROCK_CLAUDE_MODEL_ID 를 inference-profile ID로 세팅하세요 (예: global.anthropic.claude-sonnet-5)'\n"
            "\n"
            "# 실시간 미리보기: 채택되는 예시의 처음 몇 개를 생성 중에 바로 출력\n"
            "PREVIEW_N = 3\n"
            "_shown = {'n': 0}\n"
            "def _preview(done, total):\n"
            "    if _shown['n'] < PREVIEW_N and synth_ref and len(synth_ref[0]) >= done:\n"
            "        ex = synth_ref[0][done - 1]\n"
            "        u = next((m['content'] for m in ex.messages if m['role'] == 'user'), '')\n"
            "        a = next((m['content'] for m in reversed(ex.messages) if m['role'] == 'assistant'), '')\n"
            "        print(f\"\\n[preview #{done}] g/r={ex.groundedness}/{ex.relevance}\\n  in : {u[:120]}\\n  out: {a[:120]}\")\n"
            "        _shown['n'] += 1\n"
            "synth_ref = []   # generate_grounded가 채우는 리스트 참조(progress 시점에 접근)\n"
            "synth = bs.generate_grounded(\n"
            "    task_instruction=td.TASK_INSTRUCTION,\n"
            "    seed_texts=td.seed_texts_for_synth(seeds),\n"
            "    n_total=NUM_SYNTH,\n"
            "    model_id=config.BEDROCK_CLAUDE_MODEL_ID,\n"
            "    region=config.BEDROCK_REGION,\n"
            "    to_messages=td.to_messages,\n"
            "    max_batches=3 if config.is_dry_run() else None,\n"
            "    max_workers=config.SYNTH_MAX_WORKERS,   # 동시 Bedrock 호출 수 (config/env, throttling 시 낮추기)\n"
            "    accepted_ref=synth_ref,                  # 실시간 미리보기용 참조\n"
            "    progress_cb=_preview,\n"
            ")\n"
            "print(f'\\nAccepted synthetic examples: {len(synth)}')"
        ),
        md(
            "## 3. seed vs 합성 EDA (분포·다양성·품질 점검)\n"
            "학습에 넣기 전에 합성 데이터를 정량 점검합니다. 여기서 30초 쓰면, 몇 시간짜리 학습을 버리는 일을 막을 수 있습니다.\n\n"
            "| 점검 | 무엇을 보나 | 문제면 무엇을 바꾸나 |\n"
            "|---|---|---|\n"
            "| 길이·중복 | 건수, 문자 길이 분포, 완전중복률 | 분포 이탈 → 생성 프롬프트 |\n"
            "| **토큰 길이** | 실제 토크나이저 기준 + 절단 위험 | **`max_seq_length` 결정** |\n"
            "| **근사중복** | 합성끼리 닮음 + **seed 표절** | temperature↑ / **평가 누출 차단** |\n"
            "| **어휘 다양성** | distinct-1/2, 시작 3-gram 편중 | 도입부 템플릿 고착 해소 |\n"
            "| 클래스 균형 | 라벨 분포·소수 클래스 소실 | 소수 클래스 추가 생성 |\n"
            + ("| 출력 유효성 | JSON 파싱률·필수키 | validator 강화 |\n" if s.key == "extraction" else "")
            # 🔴 절단 경고는 모든 트랙에 유효하지만, 예시는 트랙 출력 형태에 맞춰야 한다.
            #    (분류=단일 라벨, 요약=요약문, QA=자유형 답변에는 'JSON'이 등장하지 않는다.)
            + "\n🔴 **토큰 길이가 특히 중요합니다.** 학습이 자르는 단위는 문자가 아니라 토큰이고, "
            + ("한국어·JSON은 " if s.key == "extraction" else "한국어처럼 영어가 아닌 텍스트는 ")
            + "문자당 토큰 수가 영어의 몇 배입니다. 문자 길이로는 안전해 보여도 토큰으로는 `max_seq_length`를 넘어 "
            "**정답 뒷부분이 잘린 채 학습**될 수 있습니다 — 그러면 모델은 "
            + ("'끝나지 않는 JSON'" if s.key == "extraction" else "'끝나지 않는 출력'")
            + "을 정답으로 배웁니다.\n"
            "🔴 **seed 표절도 놓치기 쉽습니다.** 합성이 seed를 거의 그대로 베끼면 증강 효과가 없고, 그 seed를 "
            "held-out으로 쓰면 평가 점수가 부풀려집니다(누출).\n\n"
            "각 점검은 문제를 찾으면 ⚠️ 와 함께 **구체적 조치**를 함께 출력합니다."
        ),
        code(
            "from common.synth import eda\n"
            "from transformers import AutoTokenizer\n"
            "# 🔴 토큰 길이를 실제 학습 토크나이저로 재려면 tokenizer를 넘깁니다(권장 — max_seq_length 결정에 직결).\n"
            "_tok = AutoTokenizer.from_pretrained(config.DEFAULT_MODEL_ID, token=config.get_hf_token())\n"
            "stats = eda.quick_report(\n"
            "    seeds, synth,\n"
            "    tokenizer=_tok,\n"
            f"    max_seq_length={s.max_seq_length},   # 02 학습에서 쓰는 값과 동일하게\n"
            + (f"    expect_json=True,      # 추출→JSON 트랙: 출력 스키마 검증\n" if s.key == "extraction" else "")
            + "    plot=True,\n"
            ")\n"
            + ("eda.json_field_coverage(seeds, synth)   # 추출→JSON 트랙: 함수명/인자 키 커버리지\n"
               if s.key == "extraction" else "")
        ),
        md(
            "> 개별 점검만 다시 돌리고 싶다면: `eda.compare` · `eda.token_length_report` · "
            "`eda.near_duplicate_report` · `eda.lexical_diversity` · `eda.label_balance` · "
            "`eda.output_validity` (구현: `common/synth/eda.py`).\n"
            "> 근사중복은 O(n²) 비교라 기본 400건까지만 샘플링합니다 — 전량을 보려면 `sample=` 를 키우세요."
        ),
        md(
            "## 4. 시드 + 합성 병합 → 학습 JSONL (messages 포맷)\n"
            "원본 시드와 검증을 통과한 합성 데이터를 하나로 합쳐 학습셋을 구성합니다. 각 예시는 대화형 "
            "`messages` 포맷으로 저장하는데, 이는 이후 학습 단계에서 chat template이 자동 적용되도록 하기 위한 "
            "표준 형태입니다. 완성된 경로는 `%store`로 저장해 학습 노트북에서 그대로 사용합니다."
        ),
        code(
            "import os, json\n"
            "os.makedirs('data', exist_ok=True)\n"
            "all_msgs = [td.to_messages(s) for s in seeds] + [ex.messages for ex in synth]\n"
            "train_path = 'data/train.jsonl'\n"
            "with open(train_path, 'w', encoding='utf-8') as f:\n"
            "    for m in all_msgs:\n"
            "        f.write(json.dumps({'messages': m}, ensure_ascii=False) + '\\n')\n"
            "print(f'Training set: {len(all_msgs)} examples -> {train_path}')\n"
            "# (train_path는 %store 안 함 — 02는 이 트랙의 로컬 data/train.jsonl을 직접 사용해 트랙 오염 방지)"
        ),
        md(
            "## 5. 포맷 검증 — Gemma chat template 적용 미리보기\n"
            "학습에 넘기기 전에, 토크나이저의 `apply_chat_template`을 직접 적용해 보고 `<start_of_turn>` 같은 "
            "Gemma 대화 마커가 의도대로 조립되는지 확인합니다. 학습 시점에 자동으로 적용되는 것과 동일한 변환을 "
            "미리 눈으로 검증해 두면 포맷 불일치로 인한 학습 실패를 예방할 수 있습니다."
        ),
        code(
            "from transformers import AutoTokenizer\n"
            "tok = AutoTokenizer.from_pretrained(config.DEFAULT_MODEL_ID, token=config.get_hf_token())\n"
            "print(tok.apply_chat_template(all_msgs[0], tokenize=False)[:600])"
        ),
        md("✅ 데이터 준비가 끝났습니다. 다음은 **02_train_sft_sagemaker.ipynb**로 이어집니다. (합성 데이터를 대량으로 생성하면 Bedrock 호출 비용이 늘어나므로 주의하세요.)"),
    ]


def _c02(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"02 · SageMaker SFT 학습 (TRL/LoRA) — {s.title}",
            "`scripts/train.py`를 SageMaker ModelTrainer로 실행합니다(SFT). 로컬 dry-run에서 쓰는 스크립트와 완전히 동일합니다. (GRPO 등 다른 학습법은 별도 노트북.)",
            "학습 스크립트가 self-contained이므로, 로컬 GPU에서 dry-run으로 먼저 검증한 뒤 같은 스크립트를 그대로 클라우드 학습 잡으로 제출할 수 있습니다.",
            "학습 코드를 클라우드용과 로컬용으로 나눠 관리하면 두 버전이 서로 어긋나기 쉽습니다. 하나의 파일로 통일해 이러한 drift를 원천적으로 막습니다.",
        ),
        code(SETUP_PATH),
        code(
            "import importlib, boto3, os\n"
            "from common import config, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            "%store -r role\n%store -r bucket\n"
            "# 🔴 train_path는 %store(트랙 공유) 대신 이 트랙의 로컬 파일로 고정 — 트랙 간 값 오염 방지.\n"
            "train_path = 'data/train.jsonl'\n"
            "assert os.path.isfile(train_path), (\n"
            "    f'{train_path} 가 없습니다. 이 트랙의 01_data_and_synthetic.ipynb 를 먼저 실행해 '\n"
            "    '학습 데이터를 생성하세요. (%store는 트랙 간 공유되므로 다른 트랙 경로를 쓰면 안 됩니다.)')\n"
            "# %store 오염 방지: role이 없거나 옛 플레이스홀더면 다시 해석.\n"
            "if 'role' not in dir() or not role or ':role/' not in str(role):\n"
            "    role = config.resolve_sagemaker_role(sess)\n"
            "if 'bucket' not in dir() or not bucket:\n"
            "    bucket = config.S3_BUCKET or sess.default_bucket()\n"
            "print('train_path:', train_path, f'({sum(1 for _ in open(train_path))} lines)')\n"
            "print('role      :', role)"
        ),
        md(
            "## 1. `scripts/train.py` 학습 로직 이해하기\n"
            "이 트랙의 학습은 `scripts/train.py` 한 파일이 담당하며, **로컬 dry-run과 SageMaker 학습 잡에서 동일하게** 실행됩니다.\n"
            "SageMaker로 넘기기 전에 스크립트가 수행하는 핵심 단계를 짚어봅니다.\n\n"
            "1. **데이터 로드** — `messages` 컬럼(conversational)을 가진 JSONL을 `datasets`로 로드합니다.\n"
            "2. **chat template 자동 적용** — TRL `SFTTrainer`는 conversational 데이터셋을 받으면 토크나이저의\n"
            "   `apply_chat_template`을 자동 호출합니다. 즉 `<start_of_turn>` 마커를 직접 조립하지 않습니다.\n"
            "3. **LoRA 설정** — `target_modules='all-linear'` + `modules_to_save=['lm_head','embed_tokens']`.\n"
            "   Gemma의 특수 토큰 임베딩까지 학습하기 위해 두 모듈을 저장 대상에 포함합니다.\n"
            "4. **정밀도** — `bf16=True`. Gemma는 fp16에서 오버플로/NaN이 발생하므로 fp16을 쓰지 않습니다.\n"
            "5. **packing 안전장치** — `attn_implementation`이 flash-attention이 아니면(기본 `eager`) packing을\n"
            "   비활성화합니다. packing은 여러 샘플을 한 시퀀스로 합치는데, flash-attention이 아닐 경우 샘플 간\n"
            "   cross-contamination 위험이 있기 때문입니다.\n"
            "6. **QLoRA(선택)** — `--use_qlora True`이면 4-bit(nf4, double-quant)로 로드해 단일 GPU 메모리를 절약합니다.\n"
            "7. **어댑터 저장 & 머지(선택)** — 학습된 LoRA 어댑터를 저장하고, `--merge_adapter True`이면 base에 병합해\n"
            "   서빙용 단일 모델을 만듭니다.\n\n"
            "핵심 스니펫(설명용 발췌 — 실제 실행은 아래 dry-run 셀 및 `scripts/train.py` 전체):\n"
            "```python\n"
            "peft_config = LoraConfig(\n"
            "    r=16, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',\n"
            "    target_modules='all-linear',                  # Gemma 관용구\n"
            "    modules_to_save=['lm_head', 'embed_tokens'],  # 특수 토큰 임베딩까지 학습\n"
            ")\n"
            "sft_config = SFTConfig(\n"
            "    bf16=True,                                    # fp16 금지 (Gemma NaN)\n"
            "    packing=use_packing,                          # flash-attn 아니면 자동 off\n"
            "    gradient_checkpointing=True,\n"
            "    gradient_checkpointing_kwargs={'use_reentrant': False},\n"
            "    max_length=max_seq_length, optim='adamw_torch_fused',\n"
            ")\n"
            "# SFTTrainer가 conversational 'messages' 데이터에 chat template을 자동 적용\n"
            "trainer = SFTTrainer(model, args=sft_config, train_dataset=ds,\n"
            "                     peft_config=peft_config, processing_class=tokenizer)\n"
            "```"
        ),
        md(
            # 🔴 H3로 둔다(§1의 하위 절). H2로 올려 번호를 매기면 뒤 절이 밀려 같은 노트북 안의
            #    '§4' 상호참조 5곳(잡 이름 안내·재접속 셀·완료 대기 assert)이 어긋난다.
            #    내용도 '§1에서 읽은 그 스크립트를 로컬에서 그대로 돌려보기'라 하위 절이 자연스럽다.
            "### (선택) 로컬 GPU dry-run으로 먼저 검증 (권장)\n"
            "클라우드 학습 잡은 인스턴스 시간만큼 과금되므로, GPU 개발환경이 있다면 클라우드로 제출하기 전에\n"
            "로컬에서 스크립트를 짧게 실행해 파이프라인(데이터 로드 → 토크나이즈 → 몇 step 학습 → 저장)이\n"
            "정상 동작하는지 확인합니다. `--dry_run` 플래그가 epochs=1, 짧은 시퀀스, 최대 32행으로 축소하여\n"
            "수 분 내에 끝납니다."
        ),
        code(
            "!cd scripts && python train.py --dry_run \\\n"
            "    --model_id {config.DEFAULT_MODEL_ID} \\\n"
            "    --train_file ../{train_path} \\\n"
            f"    --max_seq_length {s.max_seq_length} \\\n"
            "    --output_dir ../out_dryrun"
        ),
        md(
            "## 2. 학습 데이터 S3 업로드\n"
            "SageMaker 학습 잡은 격리된 컨테이너에서 실행되며 로컬 파일 시스템에 접근할 수 없으므로, 학습 데이터를 "
            "먼저 S3에 업로드해야 합니다. 학습 시작 시 이 S3 경로가 학습 컨테이너의 입력 채널로 마운트됩니다."
        ),
        code(
            "import os\n"
            "# 내용 해시 비교로 조건부 업로드 — 여러 번 실행해도 바뀐 게 없으면 재업로드 안 함(빠름/무료).\n"
            f"key = f'{{config.S3_PREFIX}}/{s.key}/data/' + os.path.basename(train_path)\n"
            "train_s3 = aws_utils.upload_if_changed(train_path, bucket, key, region=config.AWS_REGION)"
        ),
        md(
            "## 3. ModelTrainer 구성 (🔴 JumpStart 아님 — DLC + 커스텀 TRL 스크립트)\n"
            "sagemaker SDK v3에서는 학습을 `ModelTrainer`로 정의합니다(v2의 `HuggingFace`/`Estimator`는 제거됨).\n"
            "우리의 `scripts/train.py`는 그대로 두고, `SourceCode`로 스크립트를, `Compute`로 인스턴스를, `training_image`로\n"
            "DLC 이미지를 지정합니다.\n"
            "- **DLC 이미지**는 `.env`의 `DLC_IMAGE_URI`(리전 포함 완전 URI)로 하드코딩해 뒀습니다 —\n"
            "  `common/dlc.resolve_training_image()`가 그 값을 그대로 씁니다. 무엇이 쓰이는지 한눈에 보이고,\n"
            "  SDK가 아는 버전 목록에 매이지 않습니다(그 목록은 최신 태그를 모를 수 있습니다).\n"
            "  `train.py`가 `scripts/requirements.txt`로 필요한 라이브러리를 직접 설치하므로, 베이스가 순수\n"
            "  PyTorch DLC여도 최신 transformers/trl/peft를 컨테이너 안에서 맞출 수 있어 유리합니다.\n"
            "- ⚠️ 리전을 옮길 땐 `AWS_REGION`과 `DLC_IMAGE_URI`의 리전을 함께 바꾸세요(이미지는 리전별 ECR에서만 pull).\n"
            "  현행 태그 확인: `aws ecr describe-images --registry-id 763104351884 --repository-name pytorch-training --region <region>`\n\n"
            "**핸즈온 기본값은 짧게 잡았습니다** — `MAX_TRAIN_SAMPLES=200`, `EPOCHS=2`. 실습에서 파이프라인이 끝까지 도는지 "
            "확인하는 것이 목적이므로, 전량·다epoch 학습은 `MAX_TRAIN_SAMPLES=None` / `EPOCHS=3~5`로 올려서 따로 돌리세요.\n"
            "- `stopping_condition`도 **반드시 명시**합니다 — 생략 시 SDK 기본값 1시간에 걸려 학습이 끝난 뒤 머지 단계에서 "
            "잡이 죽습니다(바로 아래 절에 실측)."
        ),
        code(
            "from common import dlc\n"
            "from sagemaker.train.model_trainer import ModelTrainer\n"
            "from sagemaker.core.training.configs import SourceCode, Compute, InputData, StoppingCondition\n"
            "\n"
            "# ── 실습 규모 (시간·비용 조절) ──\n"
            "MAX_TRAIN_SAMPLES = 200   # train.jsonl 앞 N건만 학습(파일은 그대로). 정식 학습은 None(전체).\n"
            "EPOCHS = 2                # 실습 2 / 정식 3~5\n"
            "MAX_RUNTIME_HOURS = 4     # 초과 시 강제 중단(아래 절 참고)\n"
            "\n"
            "hyperparameters = {\n"
            "    'model_id': config.DEFAULT_MODEL_ID,\n"
            "    'epochs': EPOCHS, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8,\n"
            "    'learning_rate': 2e-4,\n"
            f"    'max_seq_length': {s.max_seq_length},\n"
            "    'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05,\n"
            f"    'use_qlora': {s.use_qlora}, 'merge_adapter': True,\n"
            "}\n"
            "if MAX_TRAIN_SAMPLES:\n"
            "    hyperparameters['max_train_samples'] = MAX_TRAIN_SAMPLES\n"
            "\n"
            "# step = ceil(건수/8) x epochs. 실측 g6.2xlarge: seq2048≈17s, seq512≈7s/step\n"
            "_n = MAX_TRAIN_SAMPLES or sum(1 for _ in open(train_path))\n"
            "_steps = -(-_n // 8) * EPOCHS\n"
            f"_eta = _steps * {17 if s.max_seq_length >= 2048 else 7} / 60\n"
            "print(f'학습 {_n}건 x {EPOCHS}epoch = 약 {_steps} step '\n"
            "      f'-> 학습 ~{_eta:.0f}분 + 머지/업로드 ~5분 (한도 {MAX_RUNTIME_HOURS}시간)')\n"
            "assert _eta / 60 < MAX_RUNTIME_HOURS, (\n"
            "    f'예상 학습 시간({_eta:.0f}분)이 MAX_RUNTIME_HOURS({MAX_RUNTIME_HOURS}시간)에 육박합니다. '\n"
            "    'MAX_TRAIN_SAMPLES/EPOCHS를 줄이거나 MAX_RUNTIME_HOURS를 올리세요.')\n"
            "environment = {'HF_TOKEN': config.get_hf_token()} if config.get_hf_token() else {}\n"
            "\n"
            "# 학습용 DLC 이미지: .env의 DLC_IMAGE_URI(완전 URI)를 그대로 사용.\n"
            "image_uri = dlc.resolve_training_image(config.AWS_REGION)\n"
            "assert image_uri, (\n"
            "    '학습 이미지 해석 실패 — .env의 DLC_IMAGE_URI를 확인하세요(리전 포함 완전 URI). '\n"
            "    '태그 목록: ' + dlc.AVAILABLE_IMAGES_URL)\n"
            "print('DLC training image:', image_uri)\n"
            "\n"
            "trainer = ModelTrainer(\n"
            "    training_image=image_uri,\n"
            "    source_code=SourceCode(source_dir='scripts', entry_script='train.py',\n"
            "                           requirements='requirements.txt'),\n"
            "    compute=Compute(instance_type=config.TRAIN_INSTANCE_TYPE, instance_count=1),\n"
            "    hyperparameters=hyperparameters,\n"
            "    environment=environment,\n"
            "    role=role,\n"
            "    sagemaker_session=sess,\n"
            f"    base_job_name='{s.endpoint_prefix}-train',\n"
            "    # 반드시 명시 — 생략 시 SDK 기본 1시간(아래 절 참고)\n"
            "    stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_HOURS * 3600),\n"
            ")"
        ),
        md(
            "### 🔴 `MaxRuntimeExceeded` — 학습이 끝났는데 잡이 `Stopped`로 죽는 함정\n"
            "`stopping_condition`을 **생략하면 SDK가 `max_runtime_in_seconds=3600`(1시간)을 자동으로 넣습니다** "
            "(`sagemaker/train/defaults.py`). 이 값은 학습 코드 시간만이 아니라 **Pending(용량 대기) + "
            "Downloading(이미지 pull) + Training + 머지/업로드 전체**를 포함하므로, 실제 학습에 쓸 수 있는 시간은 1시간보다 짧습니다.\n\n"
            "실측(03_summarization, `gemma-summarization-train-20260731084146`, ml.g6.2xlarge):\n\n"
            "| 단계 | 시간 |\n|---|---|\n"
            "| Pending (용량 대기) | 6분 |\n| Downloading (이미지 pull) | 3분 |\n"
            "| Training — 189 step **전부 완료** | 55분 |\n| ⛔ 머지 도중 강제 종료 | 1시간 도달 |\n\n"
            "**학습은 100% 끝났는데도 결과물이 버려졌습니다.** LoRA 어댑터를 base에 머지하는 마지막 단계(실측 ~2분)에서 "
            "잘려, 아티팩트에 `adapter/`와 `checkpoint-*/`만 남고 **서빙용 머지 모델이 없어** 배포가 불가능했습니다. "
            "`MaxRuntimeExceeded`는 `FailureReason`도 비어 있어(상태만 `Stopped`) 원인을 찾기 어렵습니다.\n\n"
            "그래서 이 노트북은 `MAX_RUNTIME_HOURS`(기본 4시간)를 **명시**합니다. 넉넉히 잡아도 손해가 없습니다 — "
            "잡이 정상 종료되면 그 시점에 과금이 멈추므로, 이 값은 요금이 아니라 **폭주 방지 상한**입니다.\n"
            "> 여유가 필요하면 `MAX_RUNTIME_HOURS`만 올리세요(API 최대 28일). 반대로 실습 비용을 확실히 막고 싶으면 낮추되, "
            "머지·업로드용으로 **최소 15분은 남겨** 두세요."
        ),
        md(
            "## 4. 학습 시작 (.train, 비동기 제출) + CloudWatch 링크\n"
            "`train(wait=False)`로 학습 잡을 **비동기 제출**합니다. 이렇게 하면 셀이 잡 완료까지 블로킹하지 않고 "
            "바로 반환되므로, 아래에서 출력하는 CloudWatch/콘솔 링크로 진행 상황과 로그를 실시간 확인할 수 있습니다.\n"
            "(`wait=True`로 두면 잡이 끝날 때까지 'Waiting for TrainingJob...' 패널이 계속 갱신되어 링크를 그동안 볼 수 없습니다.)\n"
            "학습 데이터는 `InputData`로 `train` 채널에 연결되어 컨테이너의 `SM_CHANNEL_TRAIN` 경로로 마운트됩니다."
        ),
        code(
            "trainer.train(input_data_config=[InputData(channel_name='train', data_source=train_s3)],\n"
            "              wait=False, logs=False)   # 비동기 제출 — 블로킹 안 함\n"
            "from IPython.display import display\n"
            "job = trainer._latest_training_job\n"
            "print('training job:', job.training_job_name)\n"
            "display(aws_utils.cw_links(config.AWS_REGION, training_job=job.training_job_name))"
        ),
        md(
            "### 진행 상태 확인 (이 셀만 반복 실행)\n"
            "이 셀을 필요할 때마다 다시 실행해 잡의 진행 단계를 봅니다. `Starting → Pending(용량 대기) → "
            "Downloading(이미지 pull) → Training(코드 실행)` 순으로 진행되며, **Training 단계부터 CloudWatch 로그가 생깁니다**."
        ),
        code(
            "aws_utils.training_job_status(job.training_job_name, config.AWS_REGION)"
        ),
        md(
            "### 세션이 끊겼을 때 잡에 다시 붙기 (재접속)\n"
            "`train(wait=False)`로 제출한 학습 잡은 **SageMaker 서버에서 실행되므로, 노트북 커널이나 VS Code 세션이 "
            "끊겨도 잡은 계속 진행됩니다.** 다시 붙으려면 `trainer` 객체를 복구할 필요 없이 **잡 이름으로 조회**하면 됩니다 "
            "(v3에서는 `sagemaker.core.resources.TrainingJob.get(name)`). 아래 셀은 커널을 재시작한 뒤 이 노트북 위쪽 "
            "설정 셀들(§0~§1)만 실행한 상태에서 바로 쓸 수 있습니다.\n"
            "> 팁: 위 §4에서 출력된 잡 이름을 메모해 두면 `TrainingJob.get('<잡 이름>')`으로 어느 커널·머신에서도 정확히 그 잡에 붙습니다."
        ),
        code(
            "from sagemaker.core.resources import TrainingJob\n"
            "# 방법 A: 잡 이름을 알면 바로 붙기 (가장 확실 — 위 §4 출력에서 복사)\n"
            "# job = TrainingJob.get('<여기에 잡 이름>')\n"
            "# 방법 B: 이름을 잊었으면 base_job_name으로 최근 잡을 찾기 (get_all은 최신순)\n"
            f"jobs = list(TrainingJob.get_all(name_contains='{s.endpoint_prefix}-train'))\n"
            "assert jobs, '이 base_job_name으로 제출된 잡이 없습니다. §4를 먼저 실행하세요.'\n"
            "job = TrainingJob.get(jobs[0].get_name())\n"
            "job.refresh()\n"
            "print('reattached to:', job.training_job_name)\n"
            "print('status       :', job.training_job_status, '/', job.secondary_status)\n"
            "# 로그를 다시 스트리밍하며 대기하려면(잡이 InProgress일 때). Ctrl-C로 빠져나와도 잡은 계속 돕니다.\n"
            "# if job.training_job_status == 'InProgress':\n"
            "#     job.wait(logs=True)\n"
            "from IPython.display import display\n"
            "display(aws_utils.cw_links(config.AWS_REGION, training_job=job.training_job_name))"
        ),
        md(
            "## 5. (선택) 학습 완료 대기 → 모델 아티팩트\n"
            "잡이 끝나야 모델 아티팩트(S3)가 생깁니다. 아래 셀은 완료될 때까지 상태를 폴링하며 기다립니다. "
            "지금 기다리기 싫으면 이 셀은 건너뛰고, CloudWatch에서 `Completed`를 확인한 뒤 다시 실행해도 됩니다."
        ),
        code(
            "import time\n"
            "# `job`은 §4의 train 셀 또는 위 '세션이 끊겼을 때' 셀에서 정의됩니다(trainer 객체에 의존하지 않음).\n"
            "assert 'job' in dir() and job is not None, (\n"
            "    \"job이 없습니다 — §4의 train 셀이나 위 '세션이 끊겼을 때 잡에 다시 붙기' 셀을 먼저 실행하세요.\")\n"
            "while True:\n"
            "    job.refresh()\n"
            "    st = job.training_job_status\n"
            "    print('status:', st)\n"
            "    if st in ('Completed', 'Failed', 'Stopped'):\n"
            "        break\n"
            "    time.sleep(30)\n"
            "assert st == 'Completed', f'학습 잡이 {st} 상태입니다. CloudWatch 로그를 확인하세요.'\n"
            "# v3 모델 아티팩트 S3 URI = model_artifacts.s3_model_artifacts (v2의 estimator.model_data 대응)\n"
            "model_data = job.model_artifacts.s3_model_artifacts\n"
            "print('Training complete. Model artifact:', model_data)\n"
            f"{_md_var(s)} = model_data\n"
            "%store model_data\n"
            f"%store {_md_var(s)}"
        ),
        md("✅ 학습이 끝나고 `model_data`가 저장되면, (선택) **02b_local_serve.ipynb**로 배포 전 로컬 서빙 검증을 하거나 "
           "(로컬에서도 클라우드와 같은 vLLM으로 확인합니다) 바로 **03_deploy_endpoint.ipynb**로 넘어갑니다."
           + ("\n\n> 💡 이 트랙은 reward가 명확해 **(선택) 02a_train_grpo_sagemaker.ipynb**로 GRPO 추가 정련(SFT→GRPO)도 "
              "할 수 있습니다." if s.grpo_reward_kind else "")),
    ]


def _c02b(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"02b · (선택) 배포 전 로컬에서 미리 띄워보기 — {s.title}",
            "학습한 모델을 **내 GPU에서 먼저 서빙해 보고** 클라우드로 넘어갑니다. 클라우드와 같은 엔진(vLLM)을 쓰므로, "
            "여기서 응답이 나오면 배포도 거의 그대로 됩니다.",
            "SageMaker에 배포하면 GPU를 새로 띄우고 컨테이너를 내려받느라 **한 번에 5~15분**이 걸립니다. 그런데 서빙이 "
            "안 되는 이유는 대개 모델 파일 문제라, 로컬에서 30초면 같은 오류를 볼 수 있습니다.",
            "배포를 눌러 놓고 10분 기다렸다가 실패를 확인하고, 고쳐서 또 10분 기다리는 일이 반복됩니다. 로컬에서 미리 "
            "확인하면 이 왕복을 없앨 수 있습니다.",
        ),
        md(
            "## 이 노트북을 실행하려면\n"
            "- **로컬 GPU가 필요합니다.** GPU가 없는 환경이면 이 노트북을 건너뛰고 `03_deploy_endpoint`로 바로 가세요"
            "(선택 단계입니다).\n"
            "- **vLLM 설치**: `uv pip install 'vllm>=0.25'`. 클라우드에서 쓰는 버전과 맞추면 결과가 더 잘 재현됩니다"
            "(`.env`의 `VLLM_IMAGE_URI` 태그를 보세요 — 실측 0.25.1 / 0.26.0).\n"
            "- **모델 파일**: 아래 §1이 자동으로 준비합니다(로컬 학습 결과 또는 S3 학습 산출물).\n\n"
            "🔴 **서버는 터미널에서, 호출은 노트북에서** 합니다. vLLM 서버는 Ctrl-C까지 계속 실행되므로 "
            "노트북 셀에 넣으면 그 셀이 끝나지 않아 커널이 멈춘 것처럼 보입니다.\n\n"
            "> 이 킷의 클라우드 서빙 경로는 **vLLM(기본) / SGLang / DJL LMI** 셋인데 모두 vLLM 계열 엔진입니다. "
            "그래서 로컬 확인도 `vllm serve` 하나로 통일했습니다 — 여기서 뜨면 세 경로 모두 통과할 가능성이 높습니다."
        ),
        md(
            "## 1. 모델 파일 준비 (자동)\n"
            "서빙할 모델 폴더(`config.json` + 가중치)를 아래 셀이 알아서 준비합니다:\n"
            "- 로컬 학습 결과(`./out`)가 있으면 그걸 쓰고,\n"
            "- 없으면 `%store`에 저장된 `model_data`(SageMaker 학습 산출물)를 S3에서 내려받아 풉니다.\n\n"
            "SageMaker로 학습했다면 로컬 `out`이 없는 게 정상입니다 — 이때 S3에서 자동으로 받습니다"
            "(수 GB라 처음 한 번은 몇 분 걸립니다).\n"
            "🔴 **재학습한 뒤라면**: 이전에 받아 둔 파일이 있어도 산출물이 바뀌면 **자동으로 다시 내려받습니다**"
            "(안 그러면 옛 모델을 검증하게 됩니다 — 실제로 겪은 함정입니다).\n"
            "> SFT만 했든 SFT→GRPO까지 했든 확인 방법은 같습니다. 둘 다 같은 형식의 텍스트 모델로 저장되고, "
            "`model_data`가 가리키는 것이 곧 확인 대상입니다."
        ),
        code(
            SETUP_PATH
        ),
        code(
            "from common import config, aws_utils, model_inspect as mi\n"
            "\n"
            "# %store의 model_data(학습 산출물 S3 URI)를 읽어 옵니다.\n"
            "try:\n"
            "    get_ipython().run_line_magic('store', '-r model_data')\n"
            "except Exception:\n"
            "    pass\n"
            "\n"
            "# 🔴 리전 가드: %store 값은 리전을 바꿔도 남으므로 옛 리전 버킷을 가리킬 수 있습니다.\n"
            "model_data = aws_utils.ensure_model_data_in_region(\n"
            f"    locals().get('model_data'), config.AWS_REGION, job_prefix='{s.endpoint_prefix}-train')\n"
            "\n"
            "# 로컬 모델 디렉토리 확보: 'out'(로컬 dry-run)이 있으면 그것을, 없으면 S3 아티팩트를 해제.\n"
            "#    아티팩트가 바뀌면 자동으로 다시 내려받습니다(캐시 스탬프 비교) → 재학습 후에도 옛 모델을\n"
            "#    검증하는 실수를 막습니다. 강제로 다시 받으려면 force=True.\n"
            "MODEL_DIR = mi.prepare_local_model(model_data, config.AWS_REGION)\n"
            "\n"
            "# vLLM으로 뜰 수 있는지 '실제 체크포인트 텐서 키'로 판정합니다(config 값만으론 알 수 없음).\n"
            "#    구현과 배경 설명: common/model_inspect.py\n"
            "info = mi.inspect_servability(MODEL_DIR)\n"
            "ENGINE, is_text_only = info['engine'], info['is_text_only']"
        ),
        md(
            "## 2. vLLM 서버 띄우기 (터미널에서)\n"
            "`vllm serve`는 **OpenAI 호환 서버**를 :8000에 띄웁니다. 즉 OpenAI SDK나 `curl`로 그대로 호출할 수 있고, "
            "클라우드에 배포한 뒤에도 같은 형식으로 부르게 됩니다.\n\n"
            "아래 셀이 복사해 붙일 명령을 만들어 줍니다. **새 터미널**을 열어(VS Code: Terminal → New Terminal) "
            "이 트랙 폴더에서 실행하세요.\n"
            "- 일반(텍스트 모델): `bash scripts/serve_local_vllm.sh <MODEL_DIR>`\n"
            "- 멀티모달 모델을 **텍스트로만** 쓰려면: `TEXT_ONLY=1 bash scripts/serve_local_vllm.sh <MODEL_DIR>`\n"
            "- 멀티모달로(이미지 입력 허용): `MULTIMODAL=1 bash scripts/serve_local_vllm.sh <MODEL_DIR>`\n\n"
            "터미널에 `Uvicorn running on http://0.0.0.0:8000`이 보이면 준비된 것입니다"
            "(모델 로딩까지 수십 초~수 분). 🔴 **확인이 끝나면 Ctrl-C로 꼭 종료하세요** — GPU를 계속 점유합니다."
        ),
        code(
            "# 위 셀의 MODEL_DIR을 그대로 쓰는 명령을 출력합니다(복사해서 터미널에 붙여넣기).\n"
            "if ENGINE != 'vllm':\n"
            "    print('🔴 KV-shared 텐서가 누락된 체크포인트입니다 — 최신 train.py로 재-export 후 다시 시도하세요.')\n"
            "else:\n"
            "    flag = '' if is_text_only else 'TEXT_ONLY=1 '\n"
            "    print(f\"{flag}bash scripts/serve_local_vllm.sh {MODEL_DIR}\")"
        ),
        md(
            "## 3. 호출해 보기 — OpenAI 호환 API\n"
            "서버가 떴으면 실제로 불러 봅니다. **부르는 방법이 세 가지**인데 전부 같은 API를 씁니다 — "
            "익숙한 것을 고르세요.\n\n"
            "| 방법 | 언제 쓰나 |\n"
            "|---|---|\n"
            "| **`requests`** (아래 3-A) | 의존성 없이 바로. 응답 원본(JSON)을 그대로 보고 싶을 때 |\n"
            "| **OpenAI SDK** (3-B) | 기존 OpenAI 코드를 그대로 재사용. `base_url`만 바꿉니다 |\n"
            "| **`curl`** (3-C) | 터미널에서 빠르게 확인 |\n\n"
            "🔴 프롬프트는 `messages`로 보냅니다 — **chat template을 서버가 적용**하므로 우리가 조립할 필요가 없습니다. "
            "raw 문자열을 보내면 template이 빠져 반복·저품질 출력이 납니다(실측).\n\n"
            "🔴 **학습에 쓴 것과 같은 형태로 물어야 합니다.** `track_data.SYSTEM_PROMPT`를 함께 보내고, 입력도 "
            "학습 데이터와 같은 구조로 줍니다. 이걸 빼면 파인튜닝한 모델이 **일반 챗봇처럼 답해** 학습이 안 된 것처럼 "
            "보입니다 — 아래 §3-D에서 같은 모델의 응답이 어떻게 달라지는지 직접 비교합니다."
        ),
        md("### 3-A. `requests` (기본)"),
        code(
            "assert ENGINE == 'vllm', ('KV-shared 텐서가 누락된 체크포인트라 vLLM으로 뜨지 않습니다. '\n"
            "                          '최신 train.py로 재-export 후 다시 시도하세요(자동 복원).')\n"
            "import requests, json\n"
            "import importlib, track_data as td; importlib.reload(td)   # 학습에 쓴 SYSTEM_PROMPT 재사용\n"
            "BASE = 'http://localhost:8000/v1'\n"
            "try:\n"
            "    served = requests.get(f'{BASE}/models', timeout=5).json()['data'][0]['id']\n"
            "    print('server up. served model:', served)\n"
            "except Exception as e:\n"
            "    raise SystemExit(f'vLLM 서버에 연결 실패({e}). 위 §2 명령으로 서버를 먼저 띄우세요.')\n"
            "\n"
            f"user = {json.dumps(s.serve_example_user or s.deploy_smoke_user, ensure_ascii=False)}\n"
            "messages = [{'role': 'system', 'content': td.SYSTEM_PROMPT},\n"
            "            {'role': 'user', 'content': user}]\n"
            "resp = requests.post(f'{BASE}/chat/completions', timeout=120, json={\n"
            "    'model': served, 'messages': messages,\n"
            "    'max_tokens': 256, 'temperature': 0.2,\n"
            "}).json()\n"
            "print('--- system ---\\n', td.SYSTEM_PROMPT)\n"
            "print('--- user ---\\n', user)\n"
            "print('--- SLM output ---\\n', resp['choices'][0]['message']['content'])\n"
            "print('--- usage ---\\n', resp.get('usage'))   # 토큰 사용량(프롬프트/생성)"
        ),
        md(
            "### 3-B. OpenAI SDK — `base_url`만 바꿔서 그대로\n"
            "OpenAI 코드를 이미 쓰고 있다면 **`base_url`과 `api_key`만 바꾸면** 우리 모델을 부를 수 있습니다"
            "(vLLM은 인증을 요구하지 않으므로 `api_key`는 아무 값이나 넣습니다). 스트리밍도 그대로 됩니다."
        ),
        code(
            _pip_install_code(["openai>=1.100.0"], "OpenAI SDK (없으면 설치)") + "\n"
            "from openai import OpenAI\n"
            "client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')  # 로컬 vLLM은 인증 없음\n"
            "\n"
            "# §3-A에서 만든 messages를 그대로 씁니다(system prompt + 학습 형태 입력).\n"
            "#    여기서 system을 빼고 맨 질문만 보내면 일반 챗봇처럼 답합니다 — §3-D에서 비교합니다.\n"
            "out = client.chat.completions.create(\n"
            "    model=served,\n"
            "    messages=messages,\n"
            f"    max_tokens={s.gen_max_tokens}, temperature=0.2,\n"
            ")\n"
            "print('--- SLM output ---\\n', out.choices[0].message.content)\n"
            "\n"
            "# 스트리밍: 토큰이 생성되는 대로 받아 첫 응답 체감을 줄입니다(vLLM은 native 지원).\n"
            "print('\\n--- streaming ---')\n"
            "for chunk in client.chat.completions.create(\n"
            "        model=served, messages=messages,\n"
            f"        max_tokens={s.gen_max_tokens}, temperature=0.2, stream=True):\n"
            "    delta = chunk.choices[0].delta.content\n"
            "    if delta:\n"
            "        print(delta, end='', flush=True)\n"
            "print()"
        ),
        md(
            "### 3-C. `curl` — 터미널에서 한 줄로\n"
            "🔴 payload를 **파일로 저장한 뒤 `-d @파일`** 로 넘깁니다. 프롬프트에 `'`(아포스트로피)나 한글이 들어가면 "
            "shell 인용이 깨지기 때문입니다(`What's...` 같은 문장에서 바로 발생)."
        ),
        code(
            "# payload를 파일로 저장(따옴표 문제 회피) → 출력된 curl을 복사해 터미널에 붙여넣으세요.\n"
            "import json as _json\n"
            "# §3-A의 messages를 그대로 씁니다(system prompt 포함).\n"
            "_payload = {'model': served, 'messages': messages, 'max_tokens': 256, 'temperature': 0.2}\n"
            "with open('req.json', 'w', encoding='utf-8') as f:\n"
            "    _json.dump(_payload, f, ensure_ascii=False)\n"
            "print('saved: req.json')\n"
            "print()\n"
            "print('curl -s http://localhost:8000/v1/chat/completions \\\\')\n"
            "print(\"  -H 'Content-Type: application/json' \\\\\")\n"
            "print(\"  -d @req.json | jq -r '.choices[0].message.content'\")\n"
            "print()\n"
            "print('# jq가 없으면: | python3 -c \\'import sys,json; print(json.load(sys.stdin)[\\\"choices\\\"][0][\\\"message\\\"][\\\"content\\\"])\\'')"
        ),
        md(
            "### 3-D. 🔴 프롬프트를 어떻게 주느냐가 결과를 바꿉니다\n"
            "**같은 모델**에 같은 질문을 세 가지 방식으로 물어 응답을 비교합니다. 파인튜닝이 잘 됐는지 판단할 때 "
            "가장 먼저 확인해야 할 부분입니다 — 학습과 다른 형태로 물으면 학습 효과가 안 보입니다.\n\n"
            "실측 예(추출 트랙):\n"
            "```\n"
            "A) system 없음, 스키마 없음 → \"I do not have real-time access to weather...\"   (일반 챗봇)\n"
            "B) system 있음, 스키마 없음 → {\"name\": \"get_current_weather\", ...}            (함수명 추측)\n"
            "C) system + 스키마         → {\"name\": \"get_weather\", \"arguments\": {...}}      ✅ 정확\n"
            "```\n"
            "C가 학습 데이터와 같은 형태입니다. **배포 후 호출부(04 평가·05 agentic)도 모두 C 방식**으로 보냅니다."
        ),
        code(
            "# 같은 모델 · 같은 질문 · 다른 프롬프트 구성 → 응답이 어떻게 달라지는지 확인합니다.\n"
            "def _ask(msgs):\n"
            "    r = requests.post(f'{BASE}/chat/completions', timeout=120, json={\n"
            "        'model': served, 'messages': msgs, 'max_tokens': 200, 'temperature': 0.2}).json()\n"
            "    return r['choices'][0]['message']['content'].strip()\n"
            "\n"
            f"bare = {json.dumps(s.deploy_smoke_user, ensure_ascii=False)}   # 학습 형태가 아닌 '맨' 질문\n"
            "\n"
            "print('=' * 70)\n"
            "print('A) system 없음 + 학습 형태 아님  ← 이렇게 물으면 학습 효과가 안 보입니다')\n"
            "print('-' * 70)\n"
            "print(_ask([{'role': 'user', 'content': bare}])[:300])\n"
            "print()\n"
            "print('=' * 70)\n"
            "print('B) system prompt만 추가')\n"
            "print('-' * 70)\n"
            "print(_ask([{'role': 'system', 'content': td.SYSTEM_PROMPT},\n"
            "            {'role': 'user', 'content': bare}])[:300])\n"
            "print()\n"
            "print('=' * 70)\n"
            "print('C) system + 학습 데이터와 같은 형태  ← 배포 후에도 이 방식으로 호출합니다')\n"
            "print('-' * 70)\n"
            "print(_ask(messages)[:300])   # §3-A에서 만든 messages(= serve_example_user)\n"
            "print('=' * 70)"
        ),
        md(
            "## 4. (선택) 성능 측정 — `vllm bench`\n"
            "vLLM에는 **벤치마크 CLI가 내장**돼 있습니다(`vllm bench --help`로 확인). 배포 전에 여기서 재 두면 "
            "**어떤 인스턴스가 필요한지, 동시 요청을 몇 개까지 받을 수 있는지**를 근거를 갖고 정할 수 있습니다.\n\n"
            "| 서브커맨드 | 무엇을 재나 | 서버 필요? |\n"
            "|---|---|---|\n"
            "| `bench serve` | **온라인 처리량 + 지연**(TTFT·TPOT·P99) — 실제 서빙에 가장 가까움 | ✅ 필요 |\n"
            "| `bench latency` | 배치 1개의 순수 추론 지연 | ❌ 직접 로드 |\n"
            "| `bench throughput` | 오프라인 일괄 처리량(배치 추론) | ❌ 직접 로드 |\n"
            "| `bench startup` | 모델 로딩 시간 — **endpoint 콜드스타트** 예측에 유용 | ❌ |\n\n"
            "**핵심 지표 두 개**\n"
            "- **TTFT**(Time To First Token): 첫 글자가 나오기까지. 체감 반응 속도를 좌우합니다.\n"
            "- **TPOT**(Time Per Output Token): 토큰당 생성 시간. 긴 답변의 총 소요를 좌우합니다.\n\n"
            "로컬 GPU와 클라우드 인스턴스가 다르면 **절대값은 다릅니다.** 그래도 "
            "\"동시 요청을 늘리면 어디서 무너지는지\", \"입력이 길어지면 얼마나 느려지는지\" 같은 **경향**은 그대로 쓸 만합니다.\n\n"
            "> 아래 셀은 명령만 출력합니다. §2에서 띄운 서버를 그대로 쓰되, **또 다른 터미널**에서 실행하세요."
        ),
        code(
            "# 실행할 명령을 출력합니다(복사해서 **새 터미널**에 붙여넣기 — §2의 서버는 그대로 두세요).\n"
            "#    스크립트가 서버 생존 확인·모델 이름 조회·플래그 조립을 대신 처리합니다.\n"
            f"print(f'cd {{os.getcwd()}}')\n"
            "print()\n"
            "print('# (1) 온라인 서빙 — 처음엔 이것만 봐도 충분합니다')\n"
            "print(f'bash scripts/bench_local_vllm.sh {MODEL_DIR}')\n"
            "print()\n"
            "print('# (2) 동시성 한계 찾기 — 1→4→8→16 스윕')\n"
            "print(f'MODE=sweep bash scripts/bench_local_vllm.sh {MODEL_DIR}')\n"
            "print()\n"
            "print('# (3) 콜드스타트 — endpoint가 InService까지 걸리는 시간을 가늠 (서버 불필요)')\n"
            "print(f'MODE=startup bash scripts/bench_local_vllm.sh {MODEL_DIR}')\n"
            "print()\n"
            "print('# (4) 오프라인 배치 처리량 — Batch Transform 규모 산정용 (서버 불필요)')\n"
            "print(f'MODE=throughput bash scripts/bench_local_vllm.sh {MODEL_DIR}')\n"
            "print()\n"
            "print('옵션(env): NUM_PROMPTS=50 CONCURRENCY=8 INPUT_LEN=1024 OUTPUT_LEN=256 SAVE=1')\n"
            "print('  SAVE=1 을 주면 결과 JSON을 ./bench 에 남깁니다.')"
        ),
        md(
            "> **측정값을 배포에 어떻게 쓰나**\n"
            "> - TTFT가 목표보다 크다 → 더 큰 GPU(또는 `--max-model-len`을 줄여 KV 캐시 여유 확보)\n"
            "> - 동시 8에서 이미 P99가 무너진다 → `03`에서 인스턴스를 키우거나 오토스케일링 최소 인스턴스를 늘립니다\n"
            "> - 콜드스타트가 길다 → endpoint를 내리지 않고 유지하거나, provisioned 상태를 유지하는 편이 낫습니다"
        ),
        md("✅ 응답이 확인됐다면 서빙 준비가 끝났습니다 — **03_deploy_endpoint.ipynb**로 배포하세요"
           "(1-A: vLLM/SGLang DLC, 1-B: DJL LMI).\n"
           "\n🔴 **로컬 vLLM 서버를 Ctrl-C로 종료하세요** — GPU를 계속 점유합니다.\n"
           "\n> §1 판정에서 KV-shared 텐서 누락이 나왔다면, 최신 `train.py`로 다시 학습/재-export하면 자동 복원됩니다"
           "(`_revive_kv_shared_from_base`가 base에서 그 54개 텐서를 되살려 저장)."),
    ]


def _c02_grpo(s: TrackSpec) -> list[dict]:
    """02 (대안) · GRPO 학습 — reward가 프로그램적으로 명확한 트랙(추출·분류)에만."""
    return [
        header(
            f"02-GRPO · SFT 다음 GRPO 정련 (RLHF) — {s.title}",
            "SFT로 학습한 모델을 **base로 이어받아** GRPO(Group Relative Policy Optimization)로 추가 정련합니다. "
            "정답 모방(SFT) 위에, prompt당 여러 응답을 생성해 **reward 함수**로 좋은 응답을 강화합니다.",
            "정석 RLHF 파이프라인은 **SFT → 정책최적화(GRPO/PPO)** 순서입니다. SFT로 형식·기본 능력을 갖춘 뒤 GRPO로 "
            "태스크 지표(추출=JSON 정확도, 분류=라벨 일치)를 직접 끌어올립니다. 이 트랙은 reward를 프로그램적으로 채점할 수 있어 GRPO에 적합합니다.",
            "base에서 바로 GRPO를 돌리면 형식조차 안 잡혀 rollout이 불안정합니다. SFT를 먼저 하면 GRPO가 안정적으로 수렴합니다. "
            "(요약·자유서술은 reward가 애매해 이 킷은 추출·분류 트랙에만 GRPO를 제공합니다.)",
        ),
        code(SETUP_PATH),
        md(
            "## GRPO vs SFT — 무엇이 다른가\n"
            "| | SFT (`02_train_sft_sagemaker`) | **GRPO (이 노트북)** |\n"
            "|---|---|---|\n"
            "| 학습 신호 | 정답 completion 모방 | reward 함수로 생성물 강화 |\n"
            "| 필요한 것 | `{messages}` (정답 포함) | prompt + **reward 함수**(정답은 채점용) |\n"
            "| 연산량 | 1x | **크다** — prompt당 `num_generations`개 생성(rollout) |\n"
            "| 적합 | 대부분의 태스크 | reward가 명확한 태스크(추출·분류) |\n\n"
            f"이 트랙의 reward: **`{s.grpo_reward_kind}`** — "
            + ("생성된 JSON의 유효성 + 함수명/인자 정확도(F1)를 0~1로 채점합니다.\n"
               if s.grpo_reward_kind == "extraction"
               else "예측 라벨이 정답 라벨과 일치하는지 채점합니다(정확 1.0 / 부분 0.3 / 오답 0.0).\n")
            + "reward 로직은 `scripts/train_grpo.py`의 `reward_" + s.grpo_reward_kind + "` 함수를 참고하세요.\n"
            "GRPO는 rollout 때문에 SFT보다 학습이 오래 걸립니다 — 실습은 `MAX_TRAIN_SAMPLES`/`num_generations`를 낮춰 시작하세요."
        ),
        code(
            "import importlib, boto3\n"
            "from common import config, dlc, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "from sagemaker.train.model_trainer import ModelTrainer\n"
            "from sagemaker.core.training.configs import SourceCode, Compute, InputData, StoppingCondition\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            "%store -r role\n"
            f"%store -r {_md_var(s)}\n"
            "%store -r model_data   # SFT(02_train_sft_sagemaker)가 저장한 산출물 = GRPO의 base\n"
            f"model_data = globals().get('{_md_var(s)}') or globals().get('model_data')\n"
            "if 'role' not in dir() or not role or ':role/' not in str(role):\n"
            "    role = config.resolve_sagemaker_role(sess)\n"
            "# 🔴 리전 가드: %store 값은 리전을 바꿔도 남으므로 옛 리전 버킷을 가리킬 수 있습니다.\n"
            "#    학습 잡도 같은 리전 S3만 읽으므로 여기서 현재 리전 최신 산출물로 맞춥니다.\n"
            "from common import aws_utils\n"
            "model_data = aws_utils.ensure_model_data_in_region(\n"
            f"    locals().get('model_data'), config.AWS_REGION, job_prefix='{s.endpoint_prefix}-train')\n"
            "print('role      :', role)\n"
            "print('SFT model :', model_data)   # 이 모델을 base로 GRPO를 이어서 학습합니다"
        ),
        md(
            "## 1. GRPO용 prompt 준비 — 🔴 **SFT 데이터를 그대로 쓰면 안 됩니다**\n"
            "SFT와 RL은 필요한 데이터가 다릅니다. GRPO는 `prompt`만 받아 스스로 생성하고 정답은 **reward 계산에만** "
            "씁니다(`train_grpo.py`의 `_to_grpo`가 `{messages}`를 `{prompt, reference}`로 분해 — reference는 모델에 "
            "보여주지 않습니다).\n\n"
            "**같은 데이터를 쓰면 학습이 아예 안 됩니다.** GRPO는 prompt당 rollout을 여러 개 생성해 "
            "**그룹 안에서 상대 비교**로 학습하는데, SFT가 이미 잘 맞히는 prompt는 rollout이 전부 만점이 되어 "
            "편차가 사라집니다 → **advantage ≈ 0 → gradient가 흐르지 않습니다.** GPU 시간만 쓰고 배우는 게 없습니다.\n"
            "🔴 그래서 **같은 분포에서 슬라이스만 나눠도 부족합니다** — 누출은 막지만 advantage 문제는 남습니다.\n\n"
            "| `GRPO_PROMPT_SOURCE` | 무엇 | 비용·선행조건 | advantage |\n"
            "|---|---|---|---|\n"
            "| `synth` (기본) | Bedrock으로 **prompt만** 생성 + 난이도 제약 | Bedrock 과금(소액) | ✅ 확보 |\n"
            "| `failures` | `04_evaluate`에서 **틀린 건만** | 03·04 선행 필요 | **가장 강함** |\n"
            "| `holdout` | SFT가 쓰지 않은 구간 | 무료·즉시 | ⚠️ 약함(같은 분포) |\n\n"
            "🔴 **기본값이 `synth`인 이유**: `holdout`은 무료지만 같은 분포라 advantage가 잘 안 생깁니다. "
            "`synth`는 생성 프롬프트에 **난이도 제약**을 걸어 어려운 예시를 만듭니다 — 실측(추출 트랙): 제약 없이 "
            "합성하면 8건 전부 인자 0개(seed 분포가 인자 없는 함수 94%)였는데, 제약을 걸면 **인자 없음 0건 / "
            "평균 인자 2.1개**가 되고 값을 간접 표현(\"the day after tomorrow\")하는 입력이 나옵니다.\n"
            "  (제약은 **생성 프롬프트에만** 넣습니다. critique에도 넣으면 seed와 다르다며 전부 기각합니다 — 실측 8/8 기각.)\n"
            "`04_evaluate`를 이미 돌렸다면 **`failures`가 가장 효과적**입니다. 실전에서는 여기에 "
            "**실제 트래픽 로그**가 가장 좋은 소스입니다.\n"
            "> 상세 근거: [`docs/03_finetuning.md` §5.5](../../docs/03_finetuning.md), 구현: `common/grpo_data.py`"
        ),
        code(
            "import os, importlib\n"
            "# 🔴 common/* 를 고친 뒤 커널을 재시작하지 않으면 파이썬이 **옛 모듈을 캐시**해 계속 씁니다\n"
            "#    (실측: 고친 aws_utils 가 반영되지 않아 Bedrock 응답 파싱이 계속 실패 → 합성 0건).\n"
            "#    reload로 이 함정을 막습니다. 그래도 이상하면 Kernel → Restart 후 처음부터 실행하세요.\n"
            "from common import aws_utils, grpo_data as gd\n"
            "from common.synth import bedrock_synth as _bs\n"
            "for _m in (aws_utils, _bs, gd):\n"
            "    importlib.reload(_m)\n"
            "import importlib, track_data as td; importlib.reload(td)\n"
            "\n"
            "# prompt 소스: 'synth'(기본) | 'failures' | 'holdout'\n"
            "GRPO_PROMPT_SOURCE = 'synth'\n"
            "N_GRPO = 100          # GRPO는 prompt당 rollout N개라 느립니다 — 작게 시작하세요.\n"
            "\n"
            "if GRPO_PROMPT_SOURCE == 'holdout':\n"
            "    # SFT가 쓴 앞 NUM_SEED_SAMPLES건 '이후' 구간 → 누출 방지(단 같은 분포)\n"
            "    rows = gd.from_holdout('data/train.jsonl', N_GRPO, sft_used=config.NUM_SEED_SAMPLES)\n"
            "elif GRPO_PROMPT_SOURCE == 'synth':\n"
            "    # ⚠️ SFT 합성과 같은 시드를 주면 분포가 또 겹칩니다 → SFT 미사용 시드 구간을 넘깁니다.\n"
            "    pool = td.load_seed_examples(config.NUM_SEED_SAMPLES + N_GRPO, token=config.get_hf_token())\n"
            "    fresh = pool[config.NUM_SEED_SAMPLES:]\n"
            "    rows = gd.from_synth(task_instruction=td.TASK_INSTRUCTION,\n"
            "                        seed_texts=td.seed_texts_for_synth(fresh),\n"
            "                        n=N_GRPO, model_id=config.BEDROCK_CLAUDE_MODEL_ID,\n"
            "                        region=config.BEDROCK_REGION, to_messages=td.to_messages,\n"
            f"                        kind='{s.grpo_reward_kind or s.key}')   # 난이도 제약 적용\n"
            "else:  # 'failures' — 04_evaluate 를 먼저 실행해 heldout/preds 가 커널에 있어야 합니다.\n"
            "    assert 'preds' in dir() and 'heldout' in dir(), (\n"
            "        \"failures 소스는 04_evaluate 의 (heldout, preds)가 필요합니다.\\n\"\n"
            "        '  → 04_evaluate 를 먼저 실행한 뒤 같은 커널에서 이 셀을 돌리거나, 그 결과를 저장해 불러오세요.')\n"
            f"    rows = gd.from_failures(heldout, preds, kind='{s.grpo_reward_kind or s.key}',\n"
            "                            to_messages=td.to_messages, max_n=N_GRPO)\n"
            "\n"
            "gd.describe(rows, source=GRPO_PROMPT_SOURCE)\n"
            "train_path = gd.write_grpo_jsonl(rows, 'data/grpo_train.jsonl')\n"
            "bucket = config.S3_BUCKET or sess.default_bucket()\n"
            "key = f'{config.S3_PREFIX}/" + s.key + "/grpo/train.jsonl'\n"
            "train_s3 = aws_utils.upload_if_changed(train_path, bucket, key, config.AWS_REGION)\n"
            "print('train_s3:', train_s3)"
        ),
        md(
            "## 2. GRPO ModelTrainer 구성\n"
            "`entry_script`가 `train_grpo.py`이고, `reward_kind`로 이 트랙의 채점 방식을 지정합니다. "
            "`num_generations`는 prompt당 생성 수(그룹 크기)로, 클수록 학습 신호가 좋아지지만 연산량이 늘어납니다."
        ),
        code(
            "# 학습 건수는 §1의 N_GRPO 가 이미 결정합니다(prompt 소스에서 그만큼만 뽑음).\n"
            "#    여기서 더 줄이려면 MAX_TRAIN_SAMPLES 를 정수로 두세요(None이면 §1이 만든 전량).\n"
            "MAX_TRAIN_SAMPLES = None\n"
            "MAX_RUNTIME_HOURS = 6   # GRPO는 rollout 때문에 SFT보다 오래 걸립니다\n"
            "hyperparameters = {\n"
            "    # model_id는 멀티모달 감지 fallback용. 실제 base는 아래 'model' 채널(SFT 산출물)에서 로드.\n"
            "    'model_id': config.DEFAULT_MODEL_ID,\n"
            f"    'reward_kind': '{s.grpo_reward_kind}',\n"
            "    'epochs': 1, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8,\n"
            "    'learning_rate': 1e-5,\n"
            "    'num_generations': 8, 'max_completion_length': 256,\n"
            f"    'max_seq_length': {s.max_seq_length},\n"
            "    'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05,\n"
            f"    'use_qlora': {s.use_qlora}, 'merge_adapter': True,\n"
            "}\n"
            "if MAX_TRAIN_SAMPLES:\n"
            "    hyperparameters['max_train_samples'] = MAX_TRAIN_SAMPLES\n"
            "environment = {'HF_TOKEN': config.get_hf_token()} if config.get_hf_token() else {}\n"
            "image_uri = dlc.resolve_training_image(config.AWS_REGION)\n"
            "assert image_uri, 'DLC 이미지 해석 실패 — DLC_IMAGE_URI env로 지정: ' + dlc.AVAILABLE_IMAGES_URL\n"
            "trainer = ModelTrainer(\n"
            "    training_image=image_uri,\n"
            "    source_code=SourceCode(source_dir='scripts', entry_script='train_grpo.py',\n"
            "                           requirements='requirements.txt'),\n"
            "    compute=Compute(instance_type=config.TRAIN_INSTANCE_TYPE, instance_count=1),\n"
            "    hyperparameters=hyperparameters,\n"
            "    environment=environment,\n"
            "    role=role,\n"
            "    sagemaker_session=sess,\n"
            f"    base_job_name='{s.endpoint_prefix}-grpo',\n"
            "    stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_HOURS * 3600),\n"
            ")"
        ),
        md(
            "## 3. 학습 시작 (비동기 제출) — SFT 산출물을 base로 마운트\n"
            "🔴 **`model` 채널로 SFT 산출물(`model_data`)을 마운트**합니다. 컨테이너 안 `SM_CHANNEL_MODEL`(=`/opt/ml/input/data/model`)에 "
            "풀리고, `train_grpo.py`가 `--base_model_dir`로 이를 base로 받아 GRPO를 이어서 학습합니다(정석 SFT→GRPO). "
            "상태 확인·세션 재접속은 SFT 노트북 §4~§5와 동일합니다(`TrainingJob.get(name)`로 재조회)."
        ),
        code(
            "trainer.train(input_data_config=[\n"
            "        InputData(channel_name='train', data_source=train_s3),\n"
            "        InputData(channel_name='model', data_source=model_data),  # SFT 산출물 = GRPO base\n"
            "    ], wait=False, logs=False)\n"
            "from IPython.display import display\n"
            "job = trainer._latest_training_job\n"
            "print('GRPO training job:', job.training_job_name)\n"
            "display(aws_utils.cw_links(config.AWS_REGION, training_job=job.training_job_name))"
        ),
        code(
            "aws_utils.training_job_status(job.training_job_name, config.AWS_REGION)"
        ),
        md(
            "## 4. 완료 대기 → 모델 아티팩트\n"
            "완료되면 SFT와 동일하게 `model_data`(S3)가 나오고, 멀티모달 base면 `train_grpo.py`가 텍스트 전용으로 "
            "재-export해 저장합니다. 이후 **03_deploy_endpoint**로 배포합니다(SFT와 동일)."
        ),
        code(
            "import time\n"
            "while True:\n"
            "    job.refresh()\n"
            "    st = job.training_job_status\n"
            "    print('status:', st)\n"
            "    if st in ('Completed', 'Failed', 'Stopped'):\n"
            "        break\n"
            "    time.sleep(30)\n"
            "assert st == 'Completed', f'GRPO 잡이 {st} 상태입니다. CloudWatch 로그를 확인하세요.'\n"
            "grpo_model_data = job.model_artifacts.s3_model_artifacts\n"
            "print('GRPO complete. Model artifact:', grpo_model_data)\n"
            "# 이후 02b/03은 model_data를 서빙합니다. GRPO 결과를 배포하려면 model_data를 GRPO 산출물로 갱신:\n"
            "model_data = grpo_model_data\n"
            f"{_md_var(s)} = grpo_model_data   # 이 트랙 전용 키도 갱신\n"
            "%store model_data\n"
            f"%store {_md_var(s)}\n"
            "%store grpo_model_data   # SFT와 비교하려면 각각의 URI를 따로 보관\n"
            "print('model_data -> GRPO 산출물로 설정됨 (02b/03이 이걸 서빙)')"
        ),
        md("✅ GRPO 학습이 끝났습니다. `model_data`가 GRPO 산출물을 가리키도록 갱신됐으니 "
           "**03_deploy_endpoint.ipynb**(또는 02b 로컬 검증)로 그대로 배포하면 됩니다.\n"
           "> SFT vs GRPO 성능 비교: `grpo_model_data`와 SFT의 URI를 각각 배포/평가(04_evaluate)해 지표를 비교하세요. "
           "SFT만 배포하려면 02_train_sft의 `model_data`를 다시 `%store` 하면 됩니다."),
    ]


def _c03(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"03 · Endpoint 배포 & invoke 스모크 — {s.title}",
            "파인튜닝한 Gemma SLM을 SageMaker real-time endpoint로 배포하고, 실제 호출을 통해 정상 동작을 검증합니다.",
            "real-time endpoint는 GPU를 상시 확보한 채 서빙하므로 지연이 낮고, 이후 agentic 루프에서 tool로 바로 호출할 수 있습니다.",
            "serverless inference는 GPU를 제공하지 않아 SLM/LLM 서빙에는 적합하지 않으므로, 여기서는 real-time endpoint를 사용합니다.",
        ),
        code(SETUP_PATH),
        code(
            "import importlib, boto3\n"
            "from common import config, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            f"%store -r {_md_var(s)}\n"
            "%store -r model_data\n%store -r role\n"
            f"model_data = globals().get('{_md_var(s)}') or globals().get('model_data')\n"
            "# model_data = 이 트랙의 최신 학습 산출물(SFT 또는 GRPO — 서빙 형식 동일).\n"
            "# %store 오염 방지: role이 없거나 옛 플레이스홀더면 다시 해석.\n"
            "if 'role' not in dir() or not role or ':role/' not in str(role):\n"
            "    role = config.resolve_sagemaker_role(sess)\n"
            "\n"
            "# 리전 가드: %store 값이 옛 리전을 가리키면 자동 교체(ensure_model_data_in_region 독스트링 참고).\n"
            "model_data = aws_utils.ensure_model_data_in_region(\n"
            f"    locals().get('model_data'), config.AWS_REGION, job_prefix='{s.endpoint_prefix}-train')\n"
            f"{_md_var(s)} = model_data\n"
            "%store model_data\n"
            f"%store {_md_var(s)}\n"
            "print('model_data:', model_data, '  ← 이 산출물을 배포합니다')\n"
            "print('role      :', role)"
        ),
        md(
            "## 배포 모드 3계층 (SDK v3 `Mode`) — 로컬에서 먼저 검증 후 endpoint\n"
            "SageMaker SDK v3 `ModelBuilder`는 **같은 코드로 3가지 배포 대상**을 고를 수 있습니다(`mode=` 인자). "
            "클라우드 endpoint를 띄우기 전에 로컬에서 먼저 검증하면 시간·비용을 아낍니다.\n\n"
            "| 모드 | 실행 위치 | 용도 | 요구 |\n"
            "|---|---|---|---|\n"
            "| `Mode.IN_PROCESS` | 현재 파이썬 프로세스 | 가장 빠른 로직 검증(초경량) | 별도 인프라 없음(백엔드 제약 있음) |\n"
            "| `Mode.LOCAL_CONTAINER` | 로컬 Docker 컨테이너 | endpoint와 **동일 컨테이너**를 로컬에서 재현 | 로컬 Docker + GPU |\n"
            "| `Mode.SAGEMAKER_ENDPOINT` | SageMaker (클라우드) | 실제 서빙(기본) | AWS 과금 |\n\n"
            "`Mode`는 `from sagemaker.serve.mode.function_pointers import Mode`로 가져오며, 지정하지 않으면 "
            "`SAGEMAKER_ENDPOINT`가 기본입니다. 아래 1-A/1-B는 모두 이 클라우드 배포 경로에 해당합니다.\n"
        ),
        md(
            "이 킷은 gemma-4를 배포하기 전 검증할 때 SDK 로컬 모드(`IN_PROCESS`/`LOCAL_CONTAINER`)를 기본으로 쓰지 않습니다. "
            "`IN_PROCESS`는 내부적으로 `transformers.pipeline`이나 `SentenceTransformer`로만 모델을 올리기 때문에 생성형 "
            "LLM인 gemma-4는 로드되지 않고, `LOCAL_CONTAINER`도 vLLM DLC에는 해당 분기가 없어 실행되지 않습니다.\n\n"
            "그래서 로컬 검증은 앞서 실행한 **`02b_local_serve`**(로컬 GPU에 `vllm serve`)로 하고, 클라우드 배포는 "
            "아래 **1-A(vLLM/SGLang DLC)** 또는 **1-B(DJL LMI)** 로 진행합니다. "
            "각 모드를 실제로 돌려 확인한 근거와 예외 사례가 궁금하다면 "
            "[`docs/05_serving_containers.md`의 §4.5](../../docs/05_serving_containers.md)를 참고하세요."
        ),
        md(
            "## 서빙 엔진 선택 — vLLM(기본) · SGLang · DJL LMI\n"
            "**전 사이즈(E2B/E4B/12B/26B/31B)가 이 세 경로로 서빙됩니다.** 셋 다 **연속 배칭 + OpenAI 호환(`messages`)** "
            "이라 호출 코드가 완전히 같습니다 — 엔진을 바꿔도 04·05 노트북은 그대로 돕니다.\n\n"
            "| SERVING_ENGINE | 컨테이너 | 특징 | 실행 절 |\n"
            "|---|---|---|---|\n"
            "| `vllm` (기본) | vLLM DLC | 최신 vLLM, 가장 널리 검증됨 | **1-A** |\n"
            "| `sglang` | SGLang DLC | RadixAttention(프리픽스 캐시 재사용에 강함) | **1-A** (같은 셀) |\n"
            "| `lmi` | DJL LMI | AWS 관리형 추상화, `OPTION_*` env | **1-B** |\n\n"
            "`.env`의 `SERVING_ENGINE`으로 고르고, 이미지는 `*_IMAGE_URI`로 하드코딩해 뒀습니다.\n\n"
            "> 🔴 **E2B/E4B에 있었던 함정 (알아두면 유용)**: E계열은 `num_kv_shared_layers>0`인데, transformers가 "
            "KV-shared 레이어에 `k_norm`/`k_proj`/`v_proj` 모듈을 만들지 않아 `save_pretrained` 시 그 텐서가 소실됩니다"
            "(E4B 실측 54개). vLLM은 전 레이어에 `k_norm`을 등록하므로 `weights not initialized` ValueError로 죽습니다"
            "([vLLM #44788](https://github.com/vllm-project/vllm/issues/44788)). "
            "**이 킷의 `train.py`가 저장 직전에 그 텐서를 복원**하므로(연산에 쓰이지 않는 dead weight라 정확도 무해) "
            "지금은 E4B도 vLLM으로 정상 서빙됩니다. 즉 #44788은 \"E계열은 vLLM 불가\"가 아니라 "
            "\"transformers가 저장한 체크포인트가 vLLM 불가\"입니다. 상세: "
            "[`docs/05_serving_containers.md` §4.7](../../docs/05_serving_containers.md)"
        ),
        code(
            "from common import config, dlc\n"
            "# 기본 'vllm'. 바꾸려면 .env의 SERVING_ENGINE=sglang|lmi (또는 여기서 ENGINE=... 직접 지정).\n"
            "ENGINE = config.SERVING_ENGINE\n"
            "print('SERVING_ENGINE:', ENGINE, '| model:', config.DEFAULT_MODEL_ID)\n"
            "print()\n"
            "# 엔진별 이미지 URI — .env에 완전 URI로 하드코딩돼 있습니다(*_IMAGE_URI).\n"
            "#    리전을 옮길 땐 AWS_REGION과 .env의 URI 리전을 함께 바꾸세요.\n"
            "for name, uri in dlc.serving_image_table(config.AWS_REGION).items():\n"
            "    print(('→ ' if name == ENGINE else '   ') + f'{name:8s} {uri}')\n"
            "print()\n"
            "print({'vllm': '아래 1-A 실행(vLLM DLC)', 'sglang': '아래 1-A 실행(SGLang DLC — 같은 셀)',\n"
            "       'lmi': '아래 1-B 실행(DJL LMI)'}[ENGINE])\n"
            "# ⚠️ 태그는 자주 갱신됩니다. 실패하면 현행 태그 확인:\n"
            "#   aws ecr describe-images --registry-id 763104351884 --repository-name vllm --region <region> \\\n"
            "#     --query 'reverse(sort_by(imageDetails,&imagePushedAt))[:5].imageTags'"
        ),
        md(
            "### 엔진과 서빙 컨테이너는 레이어가 다릅니다\n"
            "흔한 오해가 \"LMI를 쓰면 vLLM을 못 쓴다\"인데, 사실 **DJL LMI는 내부에서 vLLM 엔진을 감싸는 AWS 관리형 "
            "컨테이너**입니다(`OPTION_ROLLING_BATCH=vllm`). vLLM DLC는 vLLM을 직접 담은 AWS 컨테이너이고요. "
            "즉 1-A와 1-B는 **같은 엔진을 다른 포장으로** 쓰는 선택입니다.\n\n"
            "| 구분 | **1-A. vLLM / SGLang DLC** (기본) | **1-B. DJL LMI** (옵션) |\n"
            "|---|---|---|\n"
            "| 설정 방식 | `SM_VLLM_*` / `SM_SGLANG_*` env → CLI 플래그 | `OPTION_*` env (예 `OPTION_ROLLING_BATCH=vllm`) |\n"
            "| 버전 최신성 | 최신(실측 vLLM 0.25.1/0.26.0) | 번들 vLLM 버전에 종속 |\n"
            "| 언제 | 최신 모델·최신 엔진 기능 | 관리형 추상화·기존 LMI 자산 재사용 |\n\n"
            "🔴 **gemma-4는 vLLM ≥ 0.19 필요** → 기본값이 이를 충족합니다. LMI를 쓸 땐 번들 vLLM이 이 조건을 넘는 "
            "최신 태그인지 확인하세요. 태그는 배포 직전 "
            "[available_images](https://aws.github.io/deep-learning-containers/reference/available_images/)에서 재확인하고, "
            "배경은 [`docs/05_serving_containers.md`](../../docs/05_serving_containers.md)를 참고하세요.\n"
            "🔴 **endpoint는 삭제 전까지 시간당 과금됩니다** → 실습 후 `99_cleanup` 필수. **1-A/1-B 중 하나만 실행**하세요.\n\n"
            "> **텍스트 vs 멀티모달 서빙**: gemma-4/gemma-3-4b+ 는 멀티모달 base입니다. 학습에서 "
            "**텍스트 전용으로 재-export**(config `model_type=*_text`)했다면 그냥 텍스트로 서빙됩니다. 재-export 안 한 "
            "멀티모달 아티팩트를 **텍스트로만** 쓰려면 `--limit-mm-per-prompt`로 이미지/오디오를 0으로 두세요. "
            "이미지→텍스트 등 멀티모달 태스크는 그대로 멀티모달 서빙합니다."
        ),
        md(
            "## 1-A. vLLM / SGLang DLC로 배포 (기본, 권장)\n"
            "`SERVING_ENGINE`이 `vllm`(기본) 또는 `sglang`일 때 실행합니다 — **둘 다 이 셀이 처리합니다**(OpenAI 호환 "
            "서버라 호출 스키마가 같습니다). AWS 독립 컨테이너(`vllm:...` / `sglang:...`)에 학습 모델을 실어 배포합니다. sagemaker SDK v3는\n"
            "배포를 `ModelBuilder`로 정의합니다(v2의 `Model`/`HuggingFaceModel`은 제거됨). `image_uri`(vLLM DLC) +\n"
            "`s3_model_data_url`(학습 아티팩트) + `env_vars`(`SM_VLLM_*`)만 주면 됩니다 — passthrough 경로라\n"
            "`schema_builder`가 필요 없습니다. 모델 가중치는 `SM_VLLM_MODEL=/opt/ml/model`(아티팩트 마운트 경로,\n"
            "머지 모델이 루트에 있음)로 가리킵니다.\n"
            "🔴 gemma-4 서빙엔 vLLM ≥ 0.19 필요 → vLLM DLC(실측 0.25.1)가 충족. 태그는 실행 전 available_images에서 재확인.\n"
            "> **텍스트 전용 재-export 모델**(config `model_type=*_text`)은 그대로 텍스트 서빙됩니다. 재-export 안 한\n"
            ">  멀티모달 아티팩트를 텍스트로만 쓰려면 아래 `SM_VLLM_LIMIT_MM_PER_PROMPT` 주석을 해제하세요.\n\n"
            "🔴 **`MAX_NUM_SEQS`/`GPU_MEM_UTIL`을 낮춰 둔 이유** — 24GB GPU(L4)에서 vLLM 기본값은 여유가 거의 없습니다. "
            "실측(이 킷 endpoint, vLLM 0.26.0, E4B bf16 14.23 GiB): KV 캐시를 배정한 뒤 남은 여유가 **0.47 GiB**뿐이었습니다. "
            "멀티모달 트랙(05)은 vision tower 때문에 가중치가 1 GiB 더 커서 **같은 설정으로 CUDA OOM이 나 배포가 `Failed`**했습니다.\n"
            "- `max_num_seqs`(기본 **256**)는 샘플러 logits 버퍼를 `256 × vocab 262,144 × 4B = 256 MiB`로 잡습니다. "
            "실습은 동시 요청이 1~2건이므로 **32**로 낮춰도 손실이 없고, 버퍼는 32 MiB로 줄어듭니다.\n"
            "- 증상이 `did not pass the ping health check`로만 보여 원인을 찾기 어렵습니다 — 실제 `torch.OutOfMemoryError`는 "
            "CloudWatch endpoint 로그에만 남습니다.\n"
            "> 동시 처리량이 필요하면 `MAX_NUM_SEQS`를 올리되, 그때는 `ml.g6e.2xlarge`(L40S 45GB)처럼 큰 GPU를 쓰세요."
        ),
        code(
            "from common import config, dlc\n"
            "from sagemaker.serve import ModelBuilder\n"
            "from sagemaker.serve.mode.function_pointers import Mode\n"
            "import json, time\n"
            "# 배포 모드: 로컬에서 먼저 검증하려면 Mode.LOCAL_CONTAINER(로컬 Docker+GPU 필요)로 바꿔 실행,\n"
            "#    검증되면 Mode.SAGEMAKER_ENDPOINT(기본)로 클라우드 배포. (같은 mb 코드, mode만 교체)\n"
            "DEPLOY_MODE = Mode.SAGEMAKER_ENDPOINT   # 또는 Mode.LOCAL_CONTAINER (로컬 검증)\n"
            "# 엔진/이미지는 env가 결정합니다(SERVING_ENGINE, *_IMAGE_URI 또는 *_DLC_VERSION).\n"
            "#    이 셀은 vllm | sglang 둘 다 처리합니다(둘 다 OpenAI 호환 서버 → 호출 스키마 동일).\n"
            "assert ENGINE in ('vllm', 'sglang'), (\n"
            "    f\"ENGINE={ENGINE!r} — 이 셀은 vllm/sglang 전용입니다. \"\n"
            "    \"'lmi'면 아래 1-B를 실행하세요.\")\n"
            f"endpoint_name = f'{s.endpoint_prefix}-{{ENGINE}}-{{int(time.time())}}'\n"
            "serve_image = dlc.resolve_serving_image(config.AWS_REGION, ENGINE)\n"
            "assert serve_image, f'{ENGINE} 이미지 해석 실패 — env로 지정하세요: ' + dlc.AVAILABLE_IMAGES_URL\n"
            "print(f'{ENGINE} DLC image:', serve_image, '| mode:', DEPLOY_MODE)\n"
            "# 모델 경로 '/opt/ml/model' — train.py가 머지 모델을 아티팩트 루트에 저장합니다.\n"
            "# 엔진별 env 키는 dlc.serving_env()가 관리. max_num_seqs/mem_util은 24GB GPU OOM 방지(docs/05 §4.9).\n"
            "serve_env = dlc.serving_env(\n"
            "    ENGINE,\n"
            f"    max_model_len={_serve_len(s)},\n"
            "    max_num_seqs=32,\n"
            "    gpu_memory_utilization='0.90',\n"
            "    hf_token=config.get_serving_hf_token(),\n"
            "    # 멀티모달 base를 '텍스트로만' 서빙할 때(재-export 안 한 경우) 이미지/오디오 차단:\n"
            "    # mm_limit=json.dumps({'image': 0, 'audio': 0}),\n"
            ")\n"
            "print('serve_env:', serve_env)\n"
            "mb = ModelBuilder(\n"
            "    image_uri=serve_image,\n"
            "    s3_model_data_url=model_data,          # 학습 산출 S3 아티팩트 (v3: model_path는 로컬 경로이므로 사용 금지)\n"
            "    env_vars=serve_env,\n"
            "    role_arn=role,\n"
            "    sagemaker_session=sess,\n"
            "    instance_type=config.INFER_INSTANCE_TYPE,\n"
            "    mode=DEPLOY_MODE,                      # SAGEMAKER_ENDPOINT(기본) | LOCAL_CONTAINER(로컬 검증)\n"
            ")\n"
            "mb.build()\n"
            "# wait=False로 비동기 배포 — 셀이 'InService'까지 블로킹하지 않고 바로 반환됩니다.\n"
            "# endpoint 생성은 GPU 프로비저닝 + 컨테이너 pull + 모델 로드로 수 분~십수 분 걸립니다.\n"
            "# (LOCAL_CONTAINER 모드면 로컬 Docker에 뜨며, wait 등 일부 인자는 무시될 수 있습니다.)\n"
            "endpoint = mb.deploy(endpoint_name=endpoint_name, initial_instance_count=1,\n"
            "                     instance_type=config.INFER_INSTANCE_TYPE, wait=False)\n"
            "# 트랙 전용 키로도 저장 — 전역 키는 다른 트랙이 덮어씁니다(docs/05 §4.13).\n"
            f"{_ep_var(s)} = endpoint_name\n"
            "%store endpoint_name\n"
            f"%store {_ep_var(s)}\n"
            "from IPython.display import display\n"
            "print('deploying endpoint:', endpoint_name)\n"
            "display(aws_utils.cw_links(config.AWS_REGION, endpoint_name=endpoint_name))"
        ),
        md(
            "### 배포 상태 확인 · 세션이 끊겼을 때 다시 붙기 (재접속)\n"
            "endpoint 생성도 학습 잡처럼 **SageMaker 서버에서 진행되므로, 커널이나 세션이 끊겨도 계속됩니다.** "
            "위 `deploy(wait=False)`가 바로 반환되니, 아래 셀을 반복 실행해 `Creating → InService` 진행을 확인하세요. "
            "세션이 끊긴 뒤에는 `endpoint` 객체 없이 **endpoint 이름으로 재조회**합니다"
            "(v3: `sagemaker.core.resources.Endpoint.get(name)`).\n"
            "> 이 셀은 커널 재시작 후 위 설정 셀(임포트·`%store -r endpoint_name`)만 실행한 상태에서 바로 쓸 수 있습니다."
        ),
        code(
            "from sagemaker.core.resources import Endpoint\n"
            "# 트랙 전용 키 우선 — 전역 endpoint_name 은 다른 트랙이 덮어씁니다.\n"
            f"%store -r {_ep_var(s)}\n"
            "%store -r endpoint_name\n"
            f"endpoint_name = globals().get('{_ep_var(s)}') or globals().get('endpoint_name')\n"
            "assert endpoint_name, 'endpoint_name 이 없습니다 — 03의 배포 셀을 먼저 실행하세요.'\n"
            "print('사용할 endpoint:', endpoint_name)\n"
            "ep = Endpoint.get(endpoint_name)\n"
            "ep.refresh()\n"
            "print('endpoint:', endpoint_name, '->', ep.endpoint_status)  # Creating / InService / Failed\n"
            "if ep.endpoint_status == 'Failed':\n"
            "    print('FailureReason:', getattr(ep, 'failure_reason', None))\n"
            "# InService까지 폴링 대기하려면 아래 주석을 해제(끊겨도 서버 배포는 계속됨):\n"
            "# ep.wait_for_status(target_status='InService')\n"
            "from IPython.display import display\n"
            "display(aws_utils.cw_links(config.AWS_REGION, endpoint_name=endpoint_name))"
        ),
        md(
            "## 1-B. (옵션) DJL LMI 컨테이너로 배포 — `SERVING_ENGINE=lmi`\n"
            "AWS 관리형 서빙 추상화(자동 배칭·라우팅 등)를 원하거나 기존 LMI 자산을 재사용한다면 DJL LMI를 씁니다.\n"
            "LMI는 내부에서 vLLM 등 백엔드를 감싸며, `OPTION_*` env로 설정합니다(`OPTION_ROLLING_BATCH=vllm`).\n"
            "🔴 **버전 주의**: gemma-4 서빙엔 vLLM ≥ 0.19가 필요합니다. 번들 vLLM이 낮은 LMI 태그는 gemma-4를\n"
            "로드하지 못하므로 **최신 태그**를 쓰세요(ECR 실조회 2026-07-30: `0.36.0-lmi27.0.0-cu130-v1.1`이 최신). "
            "`LMI_VERSION` 또는 `LMI_IMAGE_URI` env로 지정합니다.\n"
            "확실한 쪽을 원하면 기본 경로인 1-A(vLLM DLC)를 쓰세요. 1-A를 이미 배포했다면 이 셀은 건너뜁니다"
            "(중복 endpoint = 중복 과금)."
        ),
        code(
            "RUN_LMI = (ENGINE == 'lmi')   # SERVING_ENGINE=lmi 이면 자동 실행. 강제하려면 True로.\n"
            "if RUN_LMI:\n"
            "    from common import dlc\n"
            "    from sagemaker.serve import ModelBuilder\n"
            "    import time\n"
            f"    endpoint_name = f'{s.endpoint_prefix}-lmi-{{int(time.time())}}'\n"
            "    lmi_image = dlc.resolve_serving_image(config.AWS_REGION, 'lmi')   # env LMI_IMAGE_URI/LMI_VERSION 존중\n"
            "    print('LMI image:', lmi_image)\n"
            "    # 1-A와 같은 함수 — 엔진만 'lmi'로 주면 OPTION_* 키로 변환됩니다.\n"
            "    lmi_env = dlc.serving_env(\n"
            "        'lmi',\n"
            f"        max_model_len={_serve_len(s)},\n"
            "        max_num_seqs=32, gpu_memory_utilization='0.90',\n"
            "        hf_token=config.get_serving_hf_token(),\n"
            "    )\n"
            "    print('lmi_env:', lmi_env)\n"
            "    mb = ModelBuilder(image_uri=lmi_image, s3_model_data_url=model_data,\n"
            "                      env_vars=lmi_env, role_arn=role, sagemaker_session=sess,\n"
            "                      instance_type=config.INFER_INSTANCE_TYPE)\n"
            "    mb.build()\n"
            "    endpoint = mb.deploy(endpoint_name=endpoint_name, initial_instance_count=1,\n"
            "                         instance_type=config.INFER_INSTANCE_TYPE, wait=False)  # 비동기\n"
            "    %store endpoint_name\n"
            "    from IPython.display import display\n"
            "    print('deploying endpoint:', endpoint_name)\n"
            "    display(aws_utils.cw_links(config.AWS_REGION, endpoint_name=endpoint_name))\n"
            "    # 상태 확인/재접속은 1-A 뒤의 재접속 셀과 동일: Endpoint.get(endpoint_name).refresh()\n"
            "else:\n"
            "    print('DJL LMI(B) skipped. Using vLLM DLC endpoint from step 1-A.')"
        ),
        *_resume_cells(s),
        md(
            "## 2. invoke 스모크 (sagemaker-runtime — Bedrock 아님)\n"
            "배포한 endpoint가 실제로 응답하는지 최소 호출로 확인합니다. 여기서 호출하는 대상은 우리가 배포한 "
            "SageMaker endpoint이므로 `sagemaker-runtime` API를 사용합니다(Bedrock 호출이 아닙니다).\n"
            "위에서 `wait=False`로 배포했으므로, **먼저 endpoint가 `InService`인지 확인**합니다. 아직 `Creating`이면 "
            "`InService`가 될 때까지 기다립니다(끊겨도 배포는 서버에서 계속됨).\n"
            "vLLM · SGLang · LMI(vLLM 백엔드) **셋 다 OpenAI 호환 chat 스키마(`messages`)** 를 받으므로, "
            "`invoke_sagemaker_chat`으로 messages를 그대로 보냅니다 — 🔴 **서버가 chat template을 적용**하므로 "
            "우리가 렌더할 필요가 없습니다(raw 텍스트를 보내면 template이 빠져 반복·저품질 출력이 납니다. 실측 확인). "
            "응답 파서는 `{\"choices\":[...]}`와 `{\"generated_text\"}` 양쪽을 모두 처리합니다."
        ),
        code(
            "# invoke 전 InService 보장 (wait=False 배포이므로)\n"
            "from sagemaker.core.resources import Endpoint\n"
            "ep = Endpoint.get(endpoint_name); ep.refresh()\n"
            "if ep.endpoint_status != 'InService':\n"
            "    print('waiting for InService (current:', ep.endpoint_status, ')...')\n"
            "    ep.wait_for_status(target_status='InService')\n"
            "print('endpoint InService:', endpoint_name)"
        ),
        code(
            "import importlib, track_data as td; importlib.reload(td)\n"
            f"user = {json.dumps(s.deploy_smoke_user, ensure_ascii=False)}\n"
            "messages = [{'role': 'user', 'content': f'{td.SYSTEM_PROMPT}\\n\\n{user}'}]\n"
            "# vllm/sglang/lmi 모두 messages 스키마 → 서버가 chat template을 적용합니다.\n"
            "out = aws_utils.invoke_sagemaker_chat(endpoint_name, messages, region=config.AWS_REGION,\n"
            f"                                     max_tokens={s.gen_max_tokens}, temperature=0.1)\n"
            "from common.display_utils import show_inference\n"
            "show_inference(user, out, title='배포 스모크')"
        ),
        md(
            "## 3. 실시간 추론 (실제 서비스와 동일한 호출)\n"
            "실제 배포 환경에는 정답(reference)이 없습니다 — 새 입력이 들어오면 그대로 추론해 응답할 뿐입니다. "
            "여기서는 **학습에 쓰지 않은 입력 1~2건**으로 실시간 추론을 보여 줍니다. (학습은 데이터셋 앞부분만 "
            "사용하므로, 그 뒤 슬라이스에서 입력을 뽑아 '새 입력' 상황을 재현합니다.)\n"
            "> 정답과 비교해 성능을 수치로 재는 held-out 평가는 뒤의 **evaluate 노트북**에서 다룹니다.\n\n"
            "**스트리밍(`STREAM`)** — vLLM/SGLang/LMI는 OpenAI 호환 SSE를 지원하므로 "
            "`invoke_endpoint_with_response_stream`으로 **토큰이 생성되는 대로** 받아볼 수 있습니다. "
            "실측(요약 트랙, vLLM 0.26.0): **첫 응답 0.42초 vs 완성 대기 16.16초 → 체감 38배**.\n"
            + ("이 트랙은 응답이 길어 기본으로 켭니다.\n"
               if _stream_default(s) else
               "이 트랙은 응답이 JSON/라벨이라 완성돼야 쓸 수 있으므로 기본은 끕니다(켜도 동작합니다).\n")
            + "> 🔴 스트리밍은 **첫 토큰 체감만** 줄입니다 — 전체 생성 시간이나 동시 처리량(throughput)은 "
            "그대로입니다. 위 실측에서도 완료 시각은 15.9s vs 16.2s로 사실상 같습니다."
        ),
        code(
            "import importlib, track_data as td; importlib.reload(td)\n"
            "# 학습에 안 쓴 입력(앞부분 NUM_SEED_SAMPLES건 이후)에서 2건만 로드 — 입력만 사용.\n"
            "holdout = td.load_seed_examples(config.NUM_SEED_SAMPLES + 2, token=config.get_hf_token())[-2:]\n"
            "\n"
            "from common.display_utils import show_inference, stream_inference\n"
            "\n"
            "# STREAM=True: 토큰을 생성되는 대로 표시(docs/05 §4.6).\n"
            f"STREAM = {_stream_default(s)}\n"
            "\n"
            "def msgs_for(user_input: str) -> list:\n"
            "    \"\"\"vllm/sglang/lmi 공통 chat 스키마 — 서버가 chat template을 적용합니다.\"\"\"\n"
            "    return [{'role': 'user', 'content': f'{td.SYSTEM_PROMPT}\\n\\n{user_input}'}]\n"
            "\n"
            "# 마크다운 렌더 — 긴 입력은 접고 JSON은 들여쓰기(print는 절단됨).\n"
            "for i, ex in enumerate(holdout, 1):\n"
            "    if STREAM:\n"
            "        pieces = aws_utils.stream_sagemaker_chat(\n"
            "            endpoint_name, msgs_for(ex['input']), region=config.AWS_REGION,\n"
            f"            max_tokens={s.gen_max_tokens}, temperature=0.2)\n"
            "        stream_inference(ex['input'], pieces, index=i)\n"
            "    else:\n"
            "        out = aws_utils.invoke_sagemaker_chat(\n"
            "            endpoint_name, msgs_for(ex['input']), region=config.AWS_REGION,\n"
            f"            max_tokens={s.gen_max_tokens}, temperature=0.2)\n"
            "        show_inference(ex['input'], out, index=i)"
        ),
        md(
            "## 4. (선택) LiteLLM 게이트웨이로도 호출\n"
            "여러 프로바이더를 하나의 인터페이스로 다루고 싶다면 LiteLLM 게이트웨이를 통해서도 endpoint를 호출할 수 있습니다. "
            "다만 LiteLLM은 sagemaker 패키지와 importlib-metadata 의존성이 충돌해 코어 의존성에 포함하지 않았으므로, "
            "필요할 때만 `pip install 'litellm>=1.93.0'`으로 별도 설치합니다.\n"
            "⚠️ 핵심 경로는 위 2번(sagemaker-runtime)만으로 완결되며, 이 셀은 통합 인터페이스를 보여 주기 위한 선택적 데모입니다."
        ),
        code(
            "try:\n"
            "    from common import llm_gateway as gw\n"
            "    resp = gw.endpoint_chat(user, endpoint_name, region=config.AWS_REGION,\n"
            f"                            chat_route=False, hf_model_name=config.DEFAULT_MODEL_ID, max_tokens={s.gen_max_tokens})\n"
            "    print('via LiteLLM:\\n', resp)\n"
            "except ImportError:\n"
            "    print('litellm not installed - optional. To use: pip install \\'litellm>=1.93.0\\' (separate env recommended)')\n"
            "except Exception as e:\n"
            "    print('LiteLLM path needs tuning to the endpoint serving schema:', e)"
        ),
        md("✅ endpoint 동작을 확인했습니다. 다음은 **04_evaluate.ipynb**로 held-out 성능을 수치로 확인합니다. (⚠️ 실습이 끝나면 endpoint를 반드시 삭제하세요.)"),
    ]


def _c04(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"05 · Agentic loop (Strands: Bedrock Claude + SLM tool) — {s.title}",
            "Strands Agent를 구성해 reasoning은 Bedrock Claude가 맡고, 도메인 특화 작업은 파인튜닝한 SLM endpoint를 tool로 호출해 처리합니다.",
            "파인튜닝한 SLM은 특정 작업을 빠르게 처리하는 전문가이고, Claude는 범용 추론과 오케스트레이션을 담당하는 역할로, 각자의 강점에 맞게 업무를 분담합니다.",
            "범용 LLM 하나로 모든 작업을 처리하면 비용이 크고 지연도 큽니다. 특화 SLM을 tool로 연결해 반복 작업을 넘김으로써 비용과 지연을 함께 낮춥니다.",
        ),
        md(
            "## 0. 응답 언어 선택\n"
            "아래 셀의 `LANG`으로 **에이전트가 사용자에게 답하는 언어**를 고릅니다. 프롬프트 자체는 영어로 두고 "
            "`' Reply in {LANG}.'` 한 문장만 덧붙이는 방식이라, **번역본을 따로 관리할 필요가 없습니다** — "
            "`'Japanese'` 처럼 아무 언어나 넣어도 동작합니다.\n\n"
            "🔴 **SLM tool의 출력은 바뀌지 않습니다.** SLM은 학습된 대로 동작하므로(이 킷의 시드는 영어 기반) "
            "tool이 돌려주는 값은 그대로이고, 달라지는 것은 **Claude가 그 결과를 설명·요약하는 언어**입니다. "
            "즉 `LANG='Korean'`은 \"작업은 영어로 처리하고 설명만 한국어로 해 줘\"에 해당합니다."
        ),
        code(
            "# Agent response language. Add any language you want — the prompts stay in English and\n"
            "# only this instruction is appended, so no translated copies to maintain.\n"
            "LANG = 'Korean'          # e.g. 'Korean' | 'English' | 'Japanese'\n"
            "REPLY_IN = f' Reply in {LANG}.'\n"
            "print('agent replies in:', LANG)"
        ),
        code(_pip_install_code(["strands-agents>=1.48.0", "strands-agents-tools"],
                               "Strands 설치 (uv 우선, pip 폴백 — 00_setup과 동일 관용구)")),
        code(SETUP_PATH),
        code(
            "import importlib\n"
            "from common import config; importlib.reload(config)\n"
            "# 트랙 전용 키 우선 — 전역 endpoint_name 은 다른 트랙이 덮어씁니다.\n"
            f"%store -r {_ep_var(s)}\n"
            "%store -r endpoint_name\n"
            f"endpoint_name = globals().get('{_ep_var(s)}') or globals().get('endpoint_name')\n"
            "assert endpoint_name, 'endpoint_name 이 없습니다 — 03의 배포 셀을 먼저 실행하세요.'\n"
            "print('사용할 endpoint:', endpoint_name)\n"
            "print('endpoint:', endpoint_name)\n"
            "assert config.BEDROCK_CLAUDE_MODEL_ID, 'BEDROCK_CLAUDE_MODEL_ID env 필요'"
        ),
        md(
            "## 1. SLM endpoint를 tool로 래핑 (🔴 sagemaker-runtime)\n"
            "에이전트가 SLM을 호출할 수 있으려면 endpoint 호출 로직을 Strands `@tool` 함수로 감싸야 합니다. "
            "tool의 docstring은 에이전트가 이 tool을 언제 사용할지 판단하는 근거가 되므로 역할을 명확히 기술합니다.\n"
            "🔴 **`messages`로 보냅니다** — vLLM/SGLang/LMI는 OpenAI 호환 서버라 **chat template을 서버가 적용**합니다. "
            "로컬에서 토크나이저로 렌더한 raw 문자열(`{inputs: ...}`)을 보내면 다음 에러가 납니다(실측):\n"
            "```\n"
            "Could not find a handler for the request. Expected one of:\n"
            "  ['ChatCompletionRequest', 'CompletionRequest']\n"
            "```\n"
            "그래서 이 tool은 `invoke_sagemaker_chat`을 씁니다 — 토크나이저·transformers 의존도 필요 없습니다."
        ),
        code(
            "from strands import Agent, tool\n"
            "from common import aws_utils, gemma_format as gf\n"
            "import importlib, track_data as td; importlib.reload(td)\n"
            "\n"
            f"@tool\n"
            f"def {s.tool_name}(text: str) -> str:\n"
            f'    """{s.tool_doc}"""\n'
            "    # 🔴 messages 그대로 전송 → 서버(vLLM/SGLang/LMI)가 chat template을 적용합니다.\n"
            "    msgs = gf.build_inference_messages(text, system_content=td.SYSTEM_PROMPT)\n"
            "    return aws_utils.invoke_sagemaker_chat(endpoint_name, msgs,\n"
            "                                          region=config.AWS_REGION,\n"
            f"                                          max_tokens={s.gen_max_tokens}, temperature=0.1)"
        ),
        md(
            "## 2. Bedrock Claude를 reasoning 모델로 Agent 구성 (모델 ID는 env)\n"
            "Strands는 Bedrock을 기본 프로바이더로 지원하므로, `BedrockModel`에 Claude 모델 ID를 지정해 reasoning "
            "엔진으로 삼습니다. 앞에서 래핑한 SLM tool을 함께 등록하면, 에이전트는 system prompt에 따라 스스로 "
            "추론하다가 필요한 시점에 SLM tool을 호출합니다. 모델 ID 같은 값은 코드에 박지 않고 env에서 읽어 옵니다."
        ),
        code(
            "from strands.models import BedrockModel\n"
            "bedrock_model = BedrockModel(model_id=config.BEDROCK_CLAUDE_MODEL_ID,\n"
            "                             region_name=config.BEDROCK_REGION)\n"
            "\n"
            "# The prompt stays in English; REPLY_IN (from the language cell) appends the language\n"
            "# instruction. Tool output is unaffected — only the agent's own wording changes.\n"
            f"SYSTEM_PROMPT = {json.dumps(s.agent_system, ensure_ascii=False)} + REPLY_IN\n"
            "agent = Agent(model=bedrock_model,\n"
            f"              tools=[{s.tool_name}],\n"
            "              system_prompt=SYSTEM_PROMPT)\n"
            "print('system_prompt:', SYSTEM_PROMPT)"
        ),
        md(
            "### (대안) LiteLLM 프로바이더로 모델 통일\n"
            "여러 프로바이더의 모델을 동일한 방식으로 다루고 싶다면, Strands의 LiteLLM 프로바이더를 통해 Bedrock 모델을 "
            "지정할 수도 있습니다. 아래는 참고용 예시이므로 주석 처리해 두었습니다."
        ),
        code(
            "# %pip install -q 'strands-agents[litellm]'\n"
            "# from strands.models.litellm import LiteLLMModel\n"
            "# lm = LiteLLMModel(model_id=f'bedrock/{config.BEDROCK_CLAUDE_MODEL_ID}', params={'max_tokens': 1024})\n"
            "# agent = Agent(model=lm, tools=[" + s.tool_name + "])"
        ),
        md(
            "## 3. 에이전트 스모크 (최소 호출 — endpoint + Bedrock 이중 과금 주의)\n"
            "구성한 에이전트에 예시 입력을 하나 넣어 reasoning → tool 호출 → 응답으로 이어지는 루프가 정상 동작하는지 "
            "확인합니다. 이 한 번의 호출에도 Bedrock reasoning과 SLM endpoint가 함께 과금되므로, 검증에는 최소한의 "
            "호출만 사용합니다."
        ),
        code(
            f"SMOKE_USER = {json.dumps(s.smoke_user, ensure_ascii=False)} + REPLY_IN\n"
            "print('--- user ---\\n', SMOKE_USER, '\\n')\n"
            "result = agent(SMOKE_USER)\n"
            "print(result)"
        ),
        md("✅ 로컬 agentic 루프가 동작합니다. 프로덕션 배포는 **06_agentcore_deploy.ipynb**로 이어지며, 실습을 마치려면 99_cleanup으로 정리하세요."),
    ]


def _c05(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"06 · AgentCore Runtime 배포 (프로덕션) — {s.title}",
            "앞서 로컬에서 검증한 Strands 에이전트를 Amazon Bedrock AgentCore Runtime에 배포합니다. 세션 격리와 서버리스 실행을 기본으로 제공합니다.",
            "AgentCore는 프레임워크와 모델에 종속되지 않는 서버리스 호스팅으로, 인프라를 직접 관리하지 않고도 에이전트를 운영할 수 있습니다.",
            "에이전트를 직접 서버로 띄우면 스케일링·세션 격리·관측(observability)을 모두 직접 구축해야 합니다. AgentCore가 이 부담을 대신 맡아 줍니다.",
        ),
        md(
            "> ⚠️ **빠르게 바뀌는 영역** — 이 영역은 변화가 잦으므로, 배포하기 전에 리전 가용성·GA 여부·CLI/SDK 스키마를 반드시 다시 확인하세요 (`# TODO verify`).\n"
            "> 검증(2026-07): 현행 권장 배포 = **`@aws/agentcore` npm CLI**(`agentcore create/dev/deploy/invoke`).\n"
            "> 구 `bedrock-agentcore-starter-toolkit`(agentcore configure/launch)는 더 이상 권장되지 않습니다.\n"
            "> 호스팅 SDK = `bedrock-agentcore`(`BedrockAgentCoreApp`/`@app.entrypoint`), ARM64 `/invocations`+`/ping` :8080.\n\n"
            "이 트랙에서 만든 tool과 endpoint에 맞춰 `agentcore/app.py`의 `SLM_ENDPOINT_NAME`과 tool 정의를 조정하세요. "
            "함께 제공되는 스캐폴드는 정보추출 트랙을 기준으로 작성되어 있습니다."
        ),
        md(
            "## 0. 사전 준비 — 🔴 VS Code **새 터미널**에서 진행하세요\n"
            "AgentCore는 설치·생성·로컬서버·배포가 모두 CLI 작업이고, 대화형 프롬프트·장시간 dev 서버·PATH 연속성 때문에 "
            "**노트북 셀이 아니라 터미널에서** 하는 게 맞습니다. (셀의 `!명령`은 매번 새 셸이라 nvm PATH가 안 이어져 "
            "`agentcore: command not found`가 납니다.)\n\n"
            "**VS Code에서 새 터미널을 열고**(Terminal → New Terminal), 리포 루트(`sagemaker-finetune-serve-e2e/`)에서 아래를 순서대로 실행하세요:\n"
            "```bash\n"
            "# 1) Node ≥ 20 + @aws/agentcore 설치 (Node가 이미 20+면 자동 스킵, sudo 불필요)\n"
            "bash agentcore/setup_agentcore_cli.sh\n"
            "source $HOME/.nvm/nvm.sh && nvm use 20      # 이 터미널 세션에 Node 20 적용\n"
            "\n"
            "# 2) 에이전트 프로젝트 생성 (대화형 마법사 대신 flag 한 방)\n"
            "bash agentcore/create_agent.sh\n"
            "```\n"
            "> `@aws/agentcore`는 **Node.js 20 이상** 필요(18 이하면 `EBADENGINE` + 런타임 오류, `/usr/local` 전역설치는 "
            "`EACCES` 권한오류). 위 스크립트가 nvm으로 홈에 Node 20을 깔아 두 문제를 모두 피합니다.\n"
            "> `create_agent.sh`가 끝나면 SLM tool 이식 결과와 다음 두 단계(로컬 검증 → 배포)를 출력합니다"
            "(아래 1~3절과 같은 내용입니다)."
        ),
        md(
            "## 1. 프로젝트 생성 (non-interactive — 마법사 대신 스크립트 한 방)\n"
            "`agentcore create`는 대화형으로 하나씩 물어 번거롭습니다. 모든 항목을 flag로 주는 스크립트로 한 번에 생성합니다"
            "(실측 2026-07, CLI v0.24.2). 🔴 **터미널에서** 실행하세요(대화형·PATH 갱신).\n"
            "```bash\n"
            "bash agentcore/create_agent.sh\n"
            "```\n"
            "> 이 스크립트가 스캐폴딩 생성 + **SLM tool 자동 이식**(데모 tool → `extract_structured_json`, "
            "`templates/main.py`) + 모델 ID env화(`templates/load.py`) + `uv sync`까지 한 번에 합니다. "
            "즉 손수 코드 편집 없이 바로 로컬 검증으로 넘어갑니다."
        ),
        md(
            "## 2. 로컬 검증 (배포 전 실제 추론 — 스크립트 한 방)\n"
            "🔴 배포 전에 **로컬에서 먼저** 에이전트가 도는지 확인합니다(AWS 과금 없음). reasoning은 Bedrock Claude·"
            "추출은 SLM endpoint tool로 처리합니다. 아래 스크립트가 dev 서버 기동 → 추론 → 종료를 자동으로 합니다:\n"
            "```bash\n"
            "bash agentcore/verify_local.sh <SLM_ENDPOINT_NAME> [AWS_REGION]\n"
            # 예시 endpoint 이름은 03이 실제로 만드는 형식({prefix}-{engine}-{timestamp})으로 트랙에 맞춰 넣는다.
            f"# 예: bash agentcore/verify_local.sh {s.endpoint_prefix}-vllm-1784XXXXXX us-west-2\n"
            "```\n"
            "> `<SLM_ENDPOINT_NAME>`은 03에서 배포한 endpoint 이름입니다(`%store -r endpoint_name`으로 확인).\n"
            "> 🔴 tool은 endpoint에 **`messages` 형식**으로 보냅니다 — 그래야 핸들러(`inference.py`)가 서버측에서 "
            "chat template을 적용합니다. raw 텍스트를 직송하면 template 미적용으로 빈/degenerate 응답이 납니다(실측).\n"
            "> 🔴 스크립트 내부 실측 노하우: dev 서버는 `setsid ... </dev/null &`로 띄우고(stdin 분리), 종료는 `kill <pid>`로 "
            "(`pkill -f 'agentcore dev'`는 실행 셸까지 죽임). 직접 돌릴 일이 있으면 참고하세요.\n"
            "```bash\n"
            "# (참고) verify_local.sh가 내부적으로 하는 일:\n"
            "# export SLM_ENDPOINT_NAME=<endpoint> AWS_REGION=<리전> BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5\n"
            "# cd agentcore/gemmaextraction\n"
            "# setsid agentcore dev --skip-deploy --logs </dev/null >/tmp/agentcore_dev.log 2>&1 &\n"
            "# sleep 20 && curl -s http://localhost:8080/ping\n"
            "# agentcore dev --stream \"Extract a tool call as JSON from: ...\" </dev/null\n"
            "# for p in $(pgrep -f 'agentcore dev'); do kill $p; done\n"
            "```"
        ),
        md(
            "## 3. 배포 (AWS Runtime endpoint)\n"
            "로컬 검증이 끝나면 클라우드에 배포합니다. 리전·GA 여부·CLI 스키마를 다시 확인한 뒤 터미널에서 실행하세요."
        ),
        code(
            "# 터미널에서 (🔴 Node ≥ 20 — 위 0단계 스크립트로 설치, create_agent.sh로 생성·로컬검증 후):\n"
            "#   export SLM_ENDPOINT_NAME=<endpoint> AWS_REGION=<리전> BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5\n"
            "#   cd agentcore/gemmaextraction\n"
            "#   agentcore deploy                       # ARM64 → ECR → Runtime endpoint (CDK)\n"
            "#   agentcore invoke --prompt '...'        # 배포된 endpoint 호출\n"
            "print('배포는 위 주석 명령을 터미널에서 — 리전/GA/CLI 스키마 재확인 후 실행하세요')"
        ),
        md(
            "## 🔴 정리\n"
            "AgentCore Runtime과 ECR 이미지, 그리고 여기에 연결된 SageMaker endpoint는 모두 과금 대상 리소스입니다. "
            "실습을 마치면 99_cleanup으로 endpoint를 정리하고, AgentCore Runtime도 별도로 삭제하세요."
        ),
    ]


def _c99(s: TrackSpec) -> list[dict]:
    return [
        header(
            f"99 · 정리 (Cleanup) — 🔴 반드시 실행 — {s.title}",
            "endpoint와 모델, 그리고 배포했다면 AgentCore Runtime까지 삭제해 과금을 멈춥니다.",
            "real-time endpoint는 삭제하기 전까지 사용 여부와 무관하게 시간당 계속 과금되기 때문입니다.",
            "실습 후 리소스를 정리하지 않으면 GPU 인스턴스 요금이 계속 청구됩니다.",
        ),
        code(SETUP_PATH),
        code(
            "import importlib, boto3\n"
            "from common import config, aws_utils; importlib.reload(config)\n"
            "aws_utils.print_cost_warning()   # 🔴 config가 아니라 aws_utils에 있습니다\n"
            "print('region:', config.AWS_REGION)"
        ),
        md(
            "## 1. SageMaker endpoint + endpoint-config + model 삭제\n"
            "endpoint를 배포하면 **endpoint · endpoint-config · model** 세 리소스가 함께 만들어집니다. "
            "시간당 과금은 endpoint에서만 발생하지만, config와 model이 남으면 콘솔이 지저분해지고 "
            "계정당 개수 제한에도 걸립니다.\n\n"
            "🔴 **model 이름은 `endpoint_name`과 다릅니다** — `ModelBuilder`가 `model-42c30d1e` 같은 임의 이름을 "
            "자동 생성하기 때문입니다. 그래서 `endpoint_name`으로 지우려 하면 model만 조용히 남습니다(실측). "
            "아래 셀은 **endpoint-config에서 실제 ModelName을 조회**해 지웁니다.\n"
            "🔴 삭제 순서는 **endpoint → config → model**입니다(config/model이 사용 중이면 삭제가 거부됨)."
        ),
        code(
            "# 트랙 전용 키 우선 — 전역 endpoint_name 은 다른 트랙이 덮어씁니다.\n"
            f"%store -r {_ep_var(s)}\n"
            "%store -r endpoint_name\n"
            f"endpoint_name = globals().get('{_ep_var(s)}') or globals().get('endpoint_name')\n"
            "assert endpoint_name, 'endpoint_name 이 없습니다 — 03의 배포 셀을 먼저 실행하세요.'\n"
            "print('사용할 endpoint:', endpoint_name)\n"
            "sm = boto3.client('sagemaker', region_name=config.AWS_REGION)\n"
            "\n"
            "# 1) endpoint-config에서 실제 model 이름을 먼저 알아낸다(삭제하면 조회 불가 → 순서 중요).\n"
            "model_names = []\n"
            "try:\n"
            "    cfg = sm.describe_endpoint_config(EndpointConfigName=endpoint_name)\n"
            "    model_names = [v['ModelName'] for v in cfg.get('ProductionVariants', []) if v.get('ModelName')]\n"
            "    print('endpoint-config가 참조하는 model:', model_names or '(없음)')\n"
            "except Exception as e:\n"
            "    print('endpoint-config 조회 생략:', str(e)[:120])\n"
            "\n"
            "# 2) endpoint → endpoint-config → model 순서로 삭제(각각 독립적으로 감싸 하나가 실패해도 계속).\n"
            "for fn, arg, name in ([(sm.delete_endpoint, 'EndpointName', endpoint_name),\n"
            "                       (sm.delete_endpoint_config, 'EndpointConfigName', endpoint_name)]\n"
            "                      + [(sm.delete_model, 'ModelName', m) for m in model_names]):\n"
            "    try:\n"
            "        fn(**{arg: name}); print(f'deleted: {arg}={name}')\n"
            "    except Exception as e:\n"
            "        print(f'skipped {arg}={name}: {str(e)[:110]}')"
        ),
        md(
            "### 이 트랙의 남은 리소스 일괄 정리 (여러 번 배포했다면)\n"
            "실습 중 endpoint를 여러 번 띄웠다면 `%store`의 `endpoint_name`은 **마지막 것만** 가리킵니다. "
            "아래 셀로 이 트랙 prefix에 해당하는 잔여 리소스를 모두 찾아 정리하세요 — 이름 없는 model까지 포함합니다."
        ),
        code(
            "# 🔴 이 트랙이 만든 리소스를 prefix로 훑어 남은 것을 모두 삭제합니다(다른 트랙/작업은 건드리지 않음).\n"
            f"PREFIX = '{s.endpoint_prefix}'\n"
            "DRY = True   # 먼저 True로 목록만 확인 → 맞으면 False로 바꿔 실제 삭제\n"
            "\n"
            "eps = [e['EndpointName'] for e in sm.list_endpoints(NameContains=PREFIX)['Endpoints']]\n"
            "cfgs = [c['EndpointConfigName'] for c in sm.list_endpoint_configs(NameContains=PREFIX)['EndpointConfigs']]\n"
            "# model은 ModelBuilder가 임의 이름(model-xxxx)을 붙여 prefix로 못 찾는다 → config가 참조하는 것을 모은다.\n"
            "models = set()\n"
            "for c in cfgs:\n"
            "    try:\n"
            "        for v in sm.describe_endpoint_config(EndpointConfigName=c).get('ProductionVariants', []):\n"
            "            if v.get('ModelName'):\n"
            "                models.add(v['ModelName'])\n"
            "    except Exception:\n"
            "        pass\n"
            "print('endpoints       :', eps or '(없음)')\n"
            "print('endpoint-configs:', cfgs or '(없음)')\n"
            "print('models          :', sorted(models) or '(없음)')\n"
            "\n"
            "if DRY:\n"
            "    print('\\nDRY=True — 목록만 표시했습니다. 위 목록이 맞으면 DRY=False로 바꿔 다시 실행하세요.')\n"
            "else:\n"
            "    for n in eps:\n"
            "        try: sm.delete_endpoint(EndpointName=n); print('deleted endpoint:', n)\n"
            "        except Exception as e: print('skip endpoint', n, str(e)[:80])\n"
            "    for n in cfgs:\n"
            "        try: sm.delete_endpoint_config(EndpointConfigName=n); print('deleted config:', n)\n"
            "        except Exception as e: print('skip config', n, str(e)[:80])\n"
            "    for n in sorted(models):\n"
            "        try: sm.delete_model(ModelName=n); print('deleted model:', n)\n"
            "        except Exception as e: print('skip model', n, str(e)[:80])"
        ),
        # 🔴 02b가 있는 트랙만 이 섹션을 넣는다(05_multimodal은 02b가 없어 스크립트도 없음).
        *([
        md(
            f"## {2 if s.has_local_serve else 0}. (02b를 실행했다면) 로컬 리소스 정리 — 모델 파일·vLLM 프로세스\n"
            "`02b_local_serve`로 로컬 검증을 했다면 **내 머신에** 다음이 남아 있습니다. 과금은 없지만 "
            "디스크와 GPU를 계속 차지합니다:\n\n"
            "| 남는 것 | 크기 | 왜 지워야 하나 |\n"
            "|---|---|---|\n"
            "| `local_model/` | **약 15GB**(E4B) | 모델 압축 해제본 |\n"
            "| vLLM 서버 프로세스 | GPU 전체 | 살아 있으면 **다음 학습/서빙이 OOM** |\n"
            "| `bench/`, `req.json` | 작음 | 벤치 결과·curl payload |\n\n"
            "터미널에서 한 줄로 정리합니다(**목록만 보여주는 게 기본** — `--yes`를 줘야 실제 삭제):\n"
            "```bash\n"
            "bash scripts/cleanup_local.sh              # 무엇이 지워질지 먼저 확인\n"
            "bash scripts/cleanup_local.sh --yes        # 실제 삭제\n"
            "KEEP_MODEL=1 bash scripts/cleanup_local.sh --yes   # 모델은 남기고 나머지만(재검증 예정)\n"
            "```\n"
            "🔴 vLLM 종료는 `kill <pid>`로 정밀하게 합니다 — `pkill -f vllm`은 실행 중인 셸/노트북까지 죽일 수 있습니다."
        ),
        code(
            "# (참고) 지금 로컬에 무엇이 남아 있는지 노트북에서 확인 — 삭제는 위 터미널 명령으로.\n"
            "import os, shutil, subprocess\n"
            "for p in ('local_model', 'bench', 'req.json', 'model.tar.gz'):\n"
            "    if os.path.exists(p):\n"
            "        sz = subprocess.run(['du', '-sh', p], capture_output=True, text=True).stdout.split()[0]\n"
            "        print(f'  {sz:>8}  {p}')\n"
            "    else:\n"
            "        print(f'  {\"-\":>8}  {p} (없음)')\n"
            "print()\n"
            "# GPU를 아직 물고 있는 프로세스가 있는지\n"
            "try:\n"
            "    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',\n"
            "                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()\n"
            "    print('GPU 점유 프로세스:', out or '(없음 ✅)')\n"
            "except FileNotFoundError:\n"
            "    print('(nvidia-smi 없음 — GPU 없는 환경)')\n"
            "print('\\n정리:  bash scripts/cleanup_local.sh --yes')"
        ),
        ] if s.has_local_serve else []),
        md(
            f"## {3 if s.has_local_serve else 2}. (AgentCore 사용 시) 로컬 dev·프로젝트 + AWS Runtime 정리\n"
            "06번에서 AgentCore를 썼다면, endpoint와는 별도로 로컬 dev 서버·프로젝트 폴더와(배포했다면) Runtime·ECR도 "
            "정리해야 합니다. **터미널에서** 아래 cleanup 스크립트 한 줄로 처리합니다:\n"
            "```bash\n"
            "bash agentcore/cleanup_agent.sh          # 로컬 정리(dev 프로세스 종료 + 프로젝트 폴더 삭제)\n"
            "bash agentcore/cleanup_agent.sh --aws    # 배포까지 했다면: 로컬 + AWS Runtime/ECR(agentcore destroy)\n"
            "```\n"
            "> 🔴 로컬 dev만 돌렸으면(배포 안 함) 첫 줄로 충분합니다 — AWS엔 과금 리소스가 없습니다. "
            "`agentcore deploy`로 클라우드에 올렸을 때만 `--aws`가 필요합니다."
        ),
        code(
            "# (참고) 노트북에서 로컬 정리를 실행하려면 — 리포 루트 기준 절대경로로:\n"
            "import os, subprocess\n"
            "def _find_repo(d=None):\n"
            "    d = os.path.abspath(d or os.getcwd())\n"
            "    for _ in range(6):\n"
            "        if os.path.isfile(os.path.join(d, 'agentcore', 'cleanup_agent.sh')):\n"
            "            return d\n"
            "        d = os.path.dirname(d)\n"
            "    return None\n"
            "_repo = _find_repo()\n"
            "if _repo:\n"
            "    print('cleanup:', os.path.join(_repo, 'agentcore', 'cleanup_agent.sh'))\n"
            "    print('터미널 실행 권장:  bash agentcore/cleanup_agent.sh [--aws]')\n"
            "    # subprocess.run(['bash', os.path.join(_repo,'agentcore','cleanup_agent.sh')], check=False)\n"
            "else:\n"
            "    print('agentcore/cleanup_agent.sh를 못 찾음 — 리포 루트에서 터미널로 실행하세요')"
        ),
        md(
            f"## {4 if s.has_local_serve else 3}. 확인 — 과금 리소스가 모두 사라졌는지\n"
            "endpoint뿐 아니라 endpoint-config·model까지 함께 확인합니다. **시간당 과금은 endpoint에서만** "
            "발생하므로 그 목록이 비어 있으면 요금은 멈춘 것입니다(config/model은 개수 제한과 콘솔 정리 문제).\n"
            "🔴 **이 트랙 것과 다른 트랙 것을 구분해서** 보여 줍니다 — 계정 전체 목록만 보면 다른 트랙의 "
            "endpoint를 보고 '이 트랙이 안 지워졌다'고 오해하게 됩니다."
        ),
        code(
            "eps = [e['EndpointName'] for e in sm.list_endpoints()['Endpoints']]\n"
            "cfgs = [c['EndpointConfigName'] for c in sm.list_endpoint_configs()['EndpointConfigs']]\n"
            "mdls = [m['ModelName'] for m in sm.list_models()['Models']]\n"
            f"mine = [n for n in eps if n.startswith('{s.endpoint_prefix}')]\n"
            "others = [n for n in eps if n not in mine]\n"
            "\n"
            f"print('이 트랙({s.endpoint_prefix}) endpoint :', mine or 'none ✅  ← 이 트랙 과금 멈춤')\n"
            "print('다른 트랙/작업 endpoint      :', others or 'none')\n"
            "print('endpoint-configs(계정 전체) :', cfgs or 'none ✅')\n"
            "print('models(계정 전체)           :', mdls or 'none ✅')\n"
            "print()\n"
            "print('region:', config.AWS_REGION, ' ← 다른 리전에도 띄운 적이 있으면 그 리전도 확인하세요')\n"
            "if mine:\n"
            "    print('\\n🔴 이 트랙 endpoint가 남아 있습니다 — 위 §1의 일괄 정리 셀을 DRY=False로 실행하세요.')\n"
            "elif others:\n"
            "    print('\\n⚠️ 이 트랙은 정리됐습니다. 다른 트랙 endpoint가 과금 중이니 그 트랙의 99_cleanup도 실행하세요:')\n"
            "    for n in others:\n"
            "        print('   -', n)\n"
            "else:\n"
            "    print('\\n✅ 이 리전에 남은 endpoint가 없습니다 — 과금 멈춤.')"
        ),
        md("✅ 정리가 끝났습니다. Bedrock Converse는 상시 유지되는 리소스가 없고 호출 단위로만 과금되므로 별도의 teardown이 필요하지 않습니다."),
    ]


def _c06(s: TrackSpec) -> list[dict]:
    """04 · 평가 (held-out). 트랙별 메트릭. eval-notebook-design 워크플로우 근거."""
    # 트랙별 메트릭 셀
    if s.eval_kind == "extraction":
        metric_md = (
            "## 3. 메트릭 (추출→JSON): valid_json_rate · name_accuracy · **arg_f1**(primary)\n"
            "구조화 추출 과제이므로, 출력이 유효한 JSON인지(valid_json_rate)와 함수/필드 이름이 맞는지(name_accuracy)를 "
            "먼저 보고, 최종 성능 지표로는 인자 단위 F1(**arg_f1**)을 사용합니다.\n"
            "🔴 held-out은 반드시 합성 증강 '이전'의 시드에서 결정론적으로 분리한 것이어야 하며, 합성 데이터로 평가해서는 안 됩니다."
        )
        metric_code = (
            "from common import eval_utils\n"
            "# gold = 파싱된 {'name','arguments'} dict\n"
            "pairs = [(pred, __import__('json').loads(ex['output'])) for pred, ex in zip(preds, heldout)]\n"
            "scores = eval_utils.eval_extraction(pairs)\n"
            "print(scores)"
        )
    elif s.eval_kind == "classification":
        metric_md = (
            "## 3. 메트릭 (분류): accuracy · **macro-F1**(primary)\n"
            "분류 과제이므로 accuracy와 함께, 클래스 불균형에 덜 민감한 **macro-F1**을 주 지표로 사용합니다. "
            "모델은 자유 텍스트로 답하므로, 예측을 닫힌 라벨셋에 매핑하기 위해 exact match → substring → fuzzy 순으로 "
            "정규화합니다. 평가 데이터는 §1에서 분리한 held-out(학습 구간 뒤 슬라이스)입니다."
        )
        metric_code = (
            "from common import eval_utils\n"
            "# 🔴 라벨 이름 77개는 track_data 헬퍼로 가져옵니다. load_dataset('PolyAI/banking77')를\n"
            "#    직접 부르면 스크립트 기반 데이터셋이라 최신 datasets에서 실패합니다.\n"
            "label_names = td.load_label_names(token=config.get_hf_token())\n"
            "pairs = [(pred, ex['output']) for pred, ex in zip(preds, heldout)]\n"
            "scores = eval_utils.eval_classification(pairs, label_names)\n"
            "print(scores)"
        )
    elif s.eval_kind == "summarization":
        metric_md = (
            "## 3. 메트릭 (요약): **ROUGE-L**(primary, 자동) + Bedrock LLM-judge(groundedness/coverage)\n"
            "요약 과제에서는 자동 지표인 **ROUGE-L**을 주 지표로 쓰되, ROUGE는 참조 요약과의 표면적 겹침(overlap)만 "
            "측정하므로 원문에 충실한지(faithfulness)는 잡아내지 못합니다. 이를 보완하기 위해 Bedrock LLM-judge로 "
            "groundedness와 coverage를 함께 평가합니다. 평가 데이터는 §1에서 분리한 held-out(학습 구간 뒤 슬라이스)입니다."
        )
        metric_code = (
            "from common import eval_utils, config\n"
            "rouge = eval_utils.eval_rouge([(pred, ex['output']) for pred, ex in zip(preds, heldout)])\n"
            "print('ROUGE:', rouge)\n"
            "# LLM-judge (Bedrock 호출 — 비용 발생). dry 평가면 앞 N건만.\n"
            "judged = []\n"
            "for pred, ex in list(zip(preds, heldout))[:20]:\n"
            "    judged.append(eval_utils.llm_judge(\n"
            "        model_id=config.BEDROCK_CLAUDE_MODEL_ID, region=config.BEDROCK_REGION,\n"
            "        source=ex['input'], prediction=pred, reference=ex['output'],\n"
            "        rubric='Rate the summary vs the source document.',\n"
            "        axes=['groundedness', 'coverage']))\n"
            "print('LLM-judge:', eval_utils.aggregate_judge(judged, ['groundedness','coverage']))"
        )
    else:  # domain_qa
        metric_md = (
            "## 3. 메트릭 (도메인QA): **Bedrock LLM-judge**(primary, correctness/helpfulness/groundedness 1-5) + ROUGE-L proxy\n"
            "도메인 QA는 정답이 자유형 문장이라 exact-match로는 제대로 평가할 수 없으므로, correctness·helpfulness·"
            "groundedness를 1~5점으로 채점하는 **Bedrock LLM-judge**를 주 지표로 사용하고, ROUGE-L은 보조 proxy로만 "
            "함께 봅니다. dolly 데이터셋은 train 스플릿만 제공하므로, 결정론적으로 분리한 슬라이스를 held-out으로 사용합니다."
        )
        metric_code = (
            "from common import eval_utils, config\n"
            "rouge = eval_utils.eval_rouge([(pred, ex['output']) for pred, ex in zip(preds, heldout)])\n"
            "print('ROUGE-L proxy:', rouge)\n"
            "judged = []\n"
            "for pred, ex in list(zip(preds, heldout))[:20]:\n"
            "    judged.append(eval_utils.llm_judge(\n"
            "        model_id=config.BEDROCK_CLAUDE_MODEL_ID, region=config.BEDROCK_REGION,\n"
            "        source=ex['input'], prediction=pred, reference=ex['output'],\n"
            "        rubric='Rate the answer for correctness, helpfulness, and (if context present) groundedness.',\n"
            "        axes=['correctness', 'helpfulness', 'groundedness']))\n"
            "print('LLM-judge:', eval_utils.aggregate_judge(judged, ['correctness','helpfulness','groundedness']))"
        )

    return [
        header(
            f"04 · 평가 (held-out) — {s.title}",
            "파인튜닝한 endpoint를 held-out 세트로 평가해 성공 기준을 수치로 확인합니다.",
            "학습이 실제로 효과가 있었는지는 학습에 쓰지 않은 held-out 지표로만 판단할 수 있습니다. 합성 데이터로 평가하면 teacher 모델을 얼마나 모방했는지를 재는 데 그칩니다.",
            "합성 데이터나 학습셋으로 평가하면 성능이 과대평가됩니다. 반드시 시드의 test 스플릿, 또는 증강 이전에 분리해 둔 슬라이스만 사용해야 합니다.",
        ),
        code(_pip_install_code(["scikit-learn>=1.5.0", "rouge-score>=0.1.2", "rapidfuzz>=3.9.0"],
                               "평가 메트릭 의존성 (pyproject 코어에도 있음; 미설치 환경 대비 안전망). uv 우선, pip 폴백")),
        code(SETUP_PATH),
        code(
            "import os, importlib\n"
            "from common import config, aws_utils, gemma_format as gf; importlib.reload(config)\n"
            "import importlib, track_data as td; importlib.reload(td)\n"
            "# 트랙 전용 키 우선 — 전역 endpoint_name 은 다른 트랙이 덮어씁니다.\n"
            f"%store -r {_ep_var(s)}\n"
            "%store -r endpoint_name\n"
            f"endpoint_name = globals().get('{_ep_var(s)}') or globals().get('endpoint_name')\n"
            "assert endpoint_name, 'endpoint_name 이 없습니다 — 03의 배포 셀을 먼저 실행하세요.'\n"
            "print('사용할 endpoint:', endpoint_name)\n"
            "from transformers import AutoTokenizer\n"
            "tok = AutoTokenizer.from_pretrained(config.DEFAULT_MODEL_ID, token=config.get_hf_token())\n"
            "# held-out 개수: 튜토리얼 기본 50(단건 서빙 기준 ~1~2분). 정식 벤치는 N_EVAL을 키우세요(env로도 조정 가능).\n"
            "N_EVAL = 20 if config.is_dry_run() else int(os.environ.get('N_EVAL', '50'))\n"
            "ENGINE = config.SERVING_ENGINE   # 로그 표시용(vllm/sglang/lmi 모두 messages 스키마라 호출 코드는 동일)"
        ),
        md(
            "## 1. held-out 세트 로드 (🔴 합성/학습셋 아님)\n"
            "평가에는 **학습에 쓰지 않은** 예시만 사용합니다. `01_data_and_synthetic`은 시드의 **앞** "
            "`config.NUM_SEED_SAMPLES`건(기본 300)을 학습 JSONL에 넣으므로, 여기서는 그 **뒤** 인덱스에서 "
            "`N_EVAL`건을 잘라 씁니다.\n\n"
            "🔴 **넉넉히 로드해서 뒤쪽 N건을 쓰는 방식(`pool[-N_EVAL:]`)은 위험합니다.** 예를 들어 `N_EVAL=50`이면 "
            "50×3=150건만 로드되어 held-out이 학습 구간(0~299) **안쪽**에 통째로 들어가고, 결과적으로 학습 데이터로 "
            "평가해 점수가 부풀려집니다. 그래서 아래 셀은 학습 구간 건수를 명시적으로 건너뜁니다.\n\n"
            "`load_seed_examples`는 같은 인덱스를 항상 같은 순서로 돌려주므로(분류 트랙은 고정 시드 42로 셔플) "
            "이 분리는 재현 가능합니다."
        ),
        code(
            "# 🔴 학습 구간과 겹치지 않게 분리: 01이 학습에 쓴 '앞 N_TRAIN_USED건'을 건너뛰고 그 뒤 N_EVAL건을 쓴다.\n"
            "N_TRAIN_USED = config.NUM_SEED_SAMPLES   # 01_data_and_synthetic 이 학습에 사용한 앞부분 건수\n"
            "pool = td.load_seed_examples(N_TRAIN_USED + N_EVAL, token=config.get_hf_token())\n"
            "heldout = pool[N_TRAIN_USED:N_TRAIN_USED + N_EVAL]\n"
            "assert heldout, (\n"
            "    f'시드가 {len(pool)}건뿐이라 학습 구간({N_TRAIN_USED}건) 뒤에 남는 예시가 없습니다 — '\n"
            "    'NUM_SEED_SAMPLES를 줄이거나 더 큰 시드 데이터셋을 쓰세요.')\n"
            "print(f'held-out: {len(heldout)}건  (시드 인덱스 {N_TRAIN_USED}~{N_TRAIN_USED + len(heldout) - 1}'\n"
            "      f' — 학습 구간 0~{N_TRAIN_USED - 1} 제외)')"
        ),
        md(
            "## 2. endpoint로 예측 생성 (🔴 sagemaker-runtime, 소규모 병렬)\n"
            "held-out 각 입력을 endpoint로 호출해 예측을 모읍니다. 재현성을 위해 `temperature=0.0`(결정론적)으로 디코딩합니다.\n"
            "- **호출 스키마**: vLLM/SGLang/LMI 모두 `{messages}`를 받고 **서버가 chat template을 적용**하므로 "
            "우리가 렌더할 필요가 없습니다(엔진을 바꿔도 이 셀은 그대로 돕니다).\n"
            "- **소규모 병렬**(`ThreadPoolExecutor`, 순서 보존)로 왕복 지연을 겹칩니다. 세 엔진 모두 연속 배칭이 있어 "
            "동시성 8로 둡니다 — endpoint 인스턴스가 작아 429/타임아웃이 나면 낮추세요."
        ),
        code(
            "from concurrent.futures import ThreadPoolExecutor\n"
            "\n"
            "def _predict(ex):\n"
            "    # 🔴 vllm/sglang/lmi 공통: messages 그대로 전송 → 서버가 chat template 적용.\n"
            "    msgs = gf.build_inference_messages(ex['input'], system_content=td.SYSTEM_PROMPT)\n"
            "    return aws_utils.invoke_sagemaker_chat(\n"
            f"        endpoint_name, msgs, region=config.AWS_REGION, max_tokens={s.gen_max_tokens}, temperature=0.0)\n"
            "\n"
            "MAX_WORKERS = int(os.environ.get('EVAL_WORKERS', '8'))   # 연속 배칭 엔진이라 8부터 시작\n"
            "with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:\n"
            "    preds = list(pool.map(_predict, heldout))   # map은 입력 순서 보존 → heldout[i]와 preds[i] 매칭 유지\n"
            "print('predictions generated:', len(preds), f'(engine={ENGINE}, workers={MAX_WORKERS})')"
        ),
        md(metric_md),
        code(metric_code),
        md("✅ 평가가 끝났습니다. 이 지표를 파인튜닝 전 baseline gemma와 비교하면 학습으로 얻은 개선폭을 정량적으로 확인할 수 있습니다. 다음은 **05_agentic_strands.ipynb**로 endpoint를 tool 삼아 agentic 루프를 구성합니다.\n\n"
           "> 💡 참고: SageMaker SDK v3의 관리형 evaluator(`BenchMarkEvaluator`/`LLMAsJudgeEvaluator`/`CustomScorerEvaluator`)는 "
           "**SageMaker Public Hub에 평가 레시피가 등록된 모델(Amazon Nova·일부 JumpStart)** 전용입니다. gemma-4 커스텀 "
           "파인튜닝 산출물(S3 체크포인트)은 Hub 레시피가 없어 지원되지 않으므로(실측: `DescribeHubContent ... does not exist`), "
           "이 킷은 위의 **로컬 메트릭 평가**를 gemma-4의 평가 경로로 사용합니다."),
    ]


_BUILDERS = {
    "00_setup.ipynb": _c00,
    "01_data_and_synthetic.ipynb": _c01,
    "02_train_sft_sagemaker.ipynb": _c02,
    "02b_local_serve.ipynb": _c02b,
    "03_deploy_endpoint.ipynb": _c03,
    "04_evaluate.ipynb": _c06,
    "05_agentic_strands.ipynb": _c04,
    "06_agentcore_deploy.ipynb": _c05,
    "99_cleanup.ipynb": _c99,
}


def build_track(spec: TrackSpec, tracks_root: str | None = None) -> None:
    """spec에 따라 트랙 디렉토리에 노트북 생성. grpo_reward_kind가 있으면 GRPO 노트북도 추가."""
    root = tracks_root or os.path.dirname(os.path.abspath(__file__))
    track_dir = os.path.join(root, spec.dir_name)
    os.makedirs(os.path.join(track_dir, "scripts"), exist_ok=True)
    builders = dict(_BUILDERS)
    if spec.grpo_reward_kind:  # reward가 명확한 트랙(추출·분류)만 GRPO 대안 노트북 제공
        builders["02a_train_grpo_sagemaker.ipynb"] = _c02_grpo
    for name, builder in builders.items():
        path = os.path.join(track_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_notebook(builder(spec)), f, ensure_ascii=False, indent=1)
    print(f"✅ {spec.dir_name}: {len(builders)} notebooks")
