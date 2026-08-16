"""정보 추출 트랙의 노트북 생성기."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 공유 빌더를 불러올 수 있도록 tracks/를 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(HERE))

from _shared_build import _clean_notebook_text  # noqa: E402


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)}


def _src(lines) -> list[str]:
    text = _clean_notebook_text("\n".join(lines))
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[dict]) -> None:
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, ensure_ascii=False, indent=1)
    print(f"생성: {name}")


# 공통 헤더
def header(title: str, tldr: str, why: str, pain: str) -> dict:
    return md(
        f"# {title}",
        "",
        f"**요약**: {tldr}",
        "",
        f"**목적**: {why}",
        "",
        f"**배경**: {pain}",
        "",
        "> 실제 실행에는 AWS 자격증명과 비용이 필요합니다. 먼저 `DRY_RUN=1`로 파이프라인을 검증하세요.",
    )


SETUP_PATH = (
    "import os, sys\n"
    "# 리포 루트를 path에 추가해 common/ 를 import\n"
    "REPO = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
    "sys.path.insert(0, REPO)\n"
    "sys.path.insert(0, os.path.join(REPO, 'tracks', '01_extraction_to_json'))"
)

# ---------------------------------------------------------------------------
# 00_setup
# ---------------------------------------------------------------------------
def _flagship_spec():
    from _shared_build import TrackSpec
    return TrackSpec(
        key="extraction", dir_name="01_extraction_to_json", title="정보추출→JSON",
        endpoint_prefix="gemma-extraction", max_seq_length=2048, use_qlora=True,
        eval_kind="extraction", grpo_reward_kind="extraction",
        tool_name="extract_structured_json",
        # 공유 에이전트 노트북에서 사용하는 트랙별 값입니다.
        tool_doc="Extract structured JSON (function call / key-values) from text using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. When the user gives text needing structured extraction, call "
                      "extract_structured_json, then explain/validate the JSON. Keep the JSON itself verbatim."),
        smoke_user=('Please extract a structured tool call from: "Book a table for 4 at 7pm at Nonna '
                    'restaurant". Available tool: book_table(restaurant, party_size, time).'),
        deploy_smoke_user="What's the weather in Busan tomorrow?",
        # 학습 데이터와 같은 도구 스키마 및 요청 형식을 사용합니다.
        serve_example_user=(
            'Available tools: [{"name": "get_weather", "description": "Get the weather forecast for a location", '
            '"parameters": {"type": "object", "properties": {"location": {"type": "string"}, '
            '"date": {"type": "string"}}, "required": ["location"]}}]\n\n'
            "What's the weather in Busan tomorrow?"
        ),
        dataset_blurb=(
            "**시드 데이터셋**: [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) "
            "(apache-2.0, ungated) — LLM function-calling 대화 셋.\n"
            "- **원본 포맷**: `system`(사용 가능한 함수 JSON 스키마) + `chat`(멀티턴 텍스트, assistant가 `<functioncall> {json}` 방출).\n"
            "- **이 트랙의 파싱**: 첫 USER 발화 + 함수 스키마 → `input`, assistant의 함수호출 JSON → `output`. "
            "즉 **자연어 요청 + 툴 스키마 → 구조화 JSON 호출** 쌍으로 변환합니다.\n"
            "- **성공 기준**: JSON 유효성 + 함수명/인자 정확도(arg F1).\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "system: SYSTEM: You are a helpful assistant with access to the following functions... \n"
            '        {\"name\": \"get_exchange_rate\", \"description\": \"Get the exchange rate between two currencies\", ...}\n'
            "chat:   USER: Can you get the exchange rate from USD to KRW?\n"
            '        ASSISTANT: <functioncall> {\"name\": \"get_exchange_rate\", \"arguments\": {\"from\": \"USD\", \"to\": \"KRW\"}}\n'
            "```\n"
            "→ 파싱 후: `input`=USER 발화+함수 스키마, `output`=`{\"name\":\"get_exchange_rate\",\"arguments\":{...}}`"
        ),
    )


def build_00():
    # 환경설정(session/role, sagemaker v3)은 공유 빌더(_shared_build._c00) 재사용 (중복 방지)
    from _shared_build import _c00
    write("00_setup.ipynb", _c00(_flagship_spec()))


# ---------------------------------------------------------------------------
# 01_data_and_synthetic
# ---------------------------------------------------------------------------
def build_01():
    # 데이터+합성(병렬/진행바)은 공유 빌더(_shared_build._c01) 재사용 (중복 방지)
    from _shared_build import _c01
    write("01_data_and_synthetic.ipynb", _c01(_flagship_spec()))


# ---------------------------------------------------------------------------
# 02_train_sft_sagemaker
# ---------------------------------------------------------------------------
def build_02():
    # 학습(ModelTrainer, sagemaker v3)은 공유 빌더(_shared_build._c02) 재사용 (중복 방지)
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from _shared_build import _c02
    write("02_train_sft_sagemaker.ipynb", _c02(_flagship_spec()))


# ---------------------------------------------------------------------------
# 02b_local_serve: 선택적 로컬 vLLM 검증
# ---------------------------------------------------------------------------
def build_02b():
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from _shared_build import _c02b
    write("02b_local_serve.ipynb", _c02b(_flagship_spec()))


# ---------------------------------------------------------------------------
# 02a_train_grpo_sagemaker: SFT 다음에 실행하는 선택적 GRPO 학습
# ---------------------------------------------------------------------------
def build_02_grpo():
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from _shared_build import _c02_grpo
    write("02a_train_grpo_sagemaker.ipynb", _c02_grpo(_flagship_spec()))


# ---------------------------------------------------------------------------
# 03_deploy_endpoint
# ---------------------------------------------------------------------------
def build_03():
    # 배포(DJL LMI 기본 + vLLM/EAGLE3 옵션)는 공유 빌더(_shared_build._c03) 재사용 (중복 방지)
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from _shared_build import TrackSpec, _c03
    spec = TrackSpec(
        key="extraction", dir_name="01_extraction_to_json", title="정보추출→JSON",
        endpoint_prefix="gemma-extraction", max_seq_length=2048, use_qlora=True,
        eval_kind="extraction",
        tool_name="extract_structured_json", tool_doc="", agent_system="",
        smoke_user="", deploy_smoke_user="What's the weather in Busan tomorrow?",
        # 학습 데이터와 같은 스키마와 요청 형식의 로컬 서빙 예시입니다.
        #    스키마를 빼면 모델이 함수명을 추측하고, system prompt까지 빼면 일반 챗봇처럼 답한다(실측).
        serve_example_user=(
            'Available tools: [{"name": "get_weather", "description": "Get the weather forecast for a location", '
            '"parameters": {"type": "object", "properties": {"location": {"type": "string"}, '
            '"date": {"type": "string"}}, "required": ["location"]}}]\n\n'
            "What's the weather in Busan tomorrow?"
        ),
    )
    write("03_deploy_endpoint.ipynb", _c03(spec))


# ---------------------------------------------------------------------------
# 05_agentic_strands
# ---------------------------------------------------------------------------
def build_04():
    """공유 빌더로 에이전트 노트북을 생성합니다."""
    from _shared_build import _c04
    write("05_agentic_strands.ipynb", _c04(_flagship_spec()))
# ---------------------------------------------------------------------------
def build_05():
    """공통 본문에 정보 추출 트랙 전용 참고 절을 추가합니다."""
    from _shared_build import _c05

    cells = _c05(_flagship_spec())
    extra = [
        md(
            "## 4. (참고) 에이전트 엔트리포인트 — `agentcore/app.py`",
            "위 `create_agent.sh`가 생성한 프로젝트와 별개로, 리포의 `agentcore/app.py`에 엔트리포인트 스캐폴드가 들어 있습니다. "
            "AgentCore Runtime이 요구하는 HTTP 계약(`/invocations`, `/ping`)과 앞 노트북에서 만든 agent 로직을 어떻게 연결하는지 이 파일로 확인하세요.",
            "> 이 스캐폴드는 **정보추출 트랙 전용**입니다(tool = `extract_structured_json`, system prompt도 추출용). 다른 트랙에 쓸 때는 tool 함수와 프롬프트를 그 트랙 것으로 바꿔야 합니다.",
        ),
        code(
            "import os\n"
            "print(open(os.path.join(os.path.abspath(os.path.join(os.getcwd(),'..','..')),\n"
            "                        'agentcore', 'app.py')).read())",
        ),
        md(
            "## 5. (대안) boto3로 직접 Runtime 생성",
            "CLI를 쓰지 않고 직접 배포하려면, ARM64 컨테이너를 빌드해 ECR에 푸시한 뒤 `bedrock-agentcore-control`의 `create_agent_runtime`을 호출합니다.",
            "⚠️ `create_agent_runtime`의 파라미터 스키마는 변경될 수 있으므로, 배포 전에 최신 boto3 레퍼런스에서 반드시 다시 확인하세요.",
        ),
        code(
            "# import boto3\n"
            "# ctl = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)\n"
            "# ctl.create_agent_runtime(\n"
            "#     agentRuntimeArtifact={'containerConfiguration': {'containerUri': '<ECR_IMAGE_ARM64>'}},\n"
            "#     networkConfiguration={'networkMode': 'PUBLIC'},\n"
            "#     roleArn='<AGENTCORE_ROLE_ARN>')   # TODO verify 스키마\n"
            "# 호출: boto3.client('bedrock-agentcore').invoke_agent_runtime(agentRuntimeArn=..., runtimeSessionId=<33+chars>, payload=..., qualifier='DEFAULT')\n"
            "print('Uncomment after re-checking the schema')",
        ),
    ]
    # 공유 본문의 마지막 '정리' 절 바로 앞에 삽입한다(정리는 항상 노트북 끝에 와야 함).
    at = next(
        (i for i, c in enumerate(cells) if "".join(c["source"]).startswith("## 정리")),
        len(cells),
    )
    write("06_agentcore_deploy.ipynb", cells[:at] + extra + cells[at:])


# ---------------------------------------------------------------------------
# 99_cleanup
# ---------------------------------------------------------------------------
def build_99():
    """공유 빌더로 정리 노트북을 생성합니다."""
    from _shared_build import _c99
    write("99_cleanup.ipynb", _c99(_flagship_spec()))


# ---------------------------------------------------------------------------
# 04_evaluate는 공유 빌더를 재사용합니다.
# ---------------------------------------------------------------------------
def build_06():
    import sys
    sys.path.insert(0, os.path.dirname(HERE))  # tracks/ 를 path에
    from _shared_build import _c06
    write("04_evaluate.ipynb", _c06(_flagship_spec()))   # evaluator 매핑 포함된 flagship spec 재사용


if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
    build_02b()      # 02b_local_serve (선택, 로컬 vLLM 프리플라이트)
    build_02_grpo()  # 02a_train_grpo_sagemaker (추출 트랙 — reward 명확)
    build_03()
    build_06()   # 04_evaluate (deploy 직후 성능 확인, 로컬 메트릭)
    build_04()   # 05_agentic_strands
    build_05()   # 06_agentcore_deploy
    build_99()
    print("done (00,01,02,02b,02a_grpo,03,04,05,06,99)")
