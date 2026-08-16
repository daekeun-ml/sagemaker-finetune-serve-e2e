"""리포 루트의 `mlflow_setup.ipynb`를 생성합니다.

노트북은 직접 수정하지 않고 이 파일을 고친 뒤 다시 생성합니다. MLflow App 하나를 모든 코스가
공유하므로 노트북도 `tracks/` 밖에 둡니다.

실행: python tools/build_mlflow_notebook.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tracks"))

from _shared_build import _notebook, code, md  # noqa: E402

OUT = os.path.join(REPO, "mlflow_setup.ipynb")

SETUP_PATH = (
    "import os, sys\n"
    "# 리포 루트를 import 경로에 추가합니다.\n"
    "REPO = os.path.abspath(os.getcwd())\n"
    "sys.path.insert(0, REPO)"
)


def cells() -> list[dict]:
    return [
        md(
            "# SageMaker Managed MLflow 설정\n\n"
            "이 노트북은 `pipelines/` 실행 결과를 MLflow에 기록하기 위한 환경을 준비합니다. 먼저 "
            "로컬 SQLite에서 기록 기능을 확인하고, 여러 사람이 실험을 공유하거나 SageMaker 학습 "
            "지표를 함께 보려는 경우에만 MLflow App을 만듭니다.\n\n"
            "한 번 실행하고 결과만 확인한다면 MLflow가 필요하지 않습니다. 이 경우 `.env`의 "
            "`USE_MLFLOW=0`을 유지하십시오. 사용 시점과 구성 방식은 "
            "[MLflow로 파인튜닝 실험 비교하기](docs/mlflow.md)에서 확인할 수 있습니다."
        ),
        md(
            "## 0. 진행 순서\n\n"
            "| 단계 | 작업 |\n"
            "|---|---|\n"
            "| 1 | 현재 MLflow 설정과 실제 기록 대상 확인 |\n"
            "| 2 | 로컬 SQLite에 테스트 실행 기록 저장 |\n"
            "| 3 | 필요한 경우 MLflow App 생성과 권한 확인 |\n"
            "| 4 | 파이프라인 실행 후 MLflow 기록 확인 |\n"
            "| 5 | 더 이상 필요하지 않은 App과 S3 아티팩트 정리 |\n\n"
            "1단계와 2단계는 AWS 리소스를 만들지 않습니다. 3단계의 App 생성 셀부터 AWS 리소스와 "
            "S3 저장 공간을 사용합니다."
        ),
        code(SETUP_PATH),
        code(
            "from common.logging_utils import setup_logging\n"
            "setup_logging()\n"
            "\n"
            "from common import config, mlflow_utils\n"
            "\n"
            "REGION = config.AWS_REGION\n"
            "print(f'리전       : {REGION}')\n"
            "print(f'mlflow     : {mlflow_utils.client_version()}')\n"
            "print(f'USE_MLFLOW : {config.USE_MLFLOW or \"(미설정)\"}')"
        ),
        md(
            "## 1. 현재 설정 확인\n\n"
            "MLflow 설정은 환경변수로 관리하며, 셸에서 지정한 값이 `.env`보다 우선합니다.\n\n"
            "| 설정 | 의미 |\n"
            "|---|---|\n"
            "| `USE_MLFLOW=0` | MLflow 기록을 모두 끕니다 |\n"
            "| `USE_MLFLOW=1` | MLflow 기록을 켭니다 |\n"
            "| `MLFLOW_APP_NAME` | 자동으로 찾을 App 이름입니다. 기본값은 `gemma-e2e`입니다 |\n"
            "| `MLFLOW_TRACKING_URI` | `local`, 표준 MLflow URI 또는 SageMaker MLflow ARN을 직접 지정합니다 |\n\n"
            "`USE_MLFLOW=1`이고 Tracking URI가 비어 있으면 같은 리전에서 App을 찾습니다. App을 "
            "찾지 못하면 `.mlflow/mlflow.db`를 사용하는 로컬 모드로 전환됩니다."
        ),
        code(
            "# 환경변수를 반영한 실제 기록 대상을 확인합니다.\n"
            "target = mlflow_utils.target_from_env()\n"
            "print(f'실제 기록 대상: {target.describe()}')"
        ),
        md(
            "## 2. 로컬 SQLite에서 기록 확인\n\n"
            "관리형 환경을 만들기 전에 MLflow 클라이언트가 정상적으로 기록하고 조회하는지 확인합니다. "
            "이 단계는 현재 `USE_MLFLOW` 값과 관계없이 테스트 실행 기록을 `.mlflow/mlflow.db`에 저장합니다."
        ),
        code(
            "# 로컬 SQLite에 테스트 run을 기록합니다.\n"
            "local = mlflow_utils.resolve_target('local')\n"
            "with mlflow_utils.run(local, 'mlflow-setup-check', run_name='connectivity') as handle:\n"
            "    handle.log_params({'check': 'local-sqlite'})\n"
            "    handle.log_metrics({'ok': 1})\n"
            "    print('run_id:', handle.run_id)"
        ),
        code(
            "# 기록된 run을 조회합니다.\n"
            "import mlflow\n"
            "\n"
            "mlflow.set_tracking_uri(local.uri)\n"
            "client = mlflow.MlflowClient()\n"
            "for experiment in client.search_experiments():\n"
            "    for run in client.search_runs([experiment.experiment_id]):\n"
            "        print(experiment.name, '|', "
            "run.data.tags.get('mlflow.runName'), '|', run.data.metrics)"
        ),
        md(
            "### 로컬 UI 열기\n\n"
            "저장소 루트의 터미널에서 다음 명령을 실행합니다.\n\n"
            "```bash\n"
            "mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --port 5000\n"
            "```\n\n"
            "EC2에서 실행 중이라면 VS Code Remote SSH, SSH 포트 포워딩 또는 SSM 포트 포워딩을 "
            "사용합니다. 다음 셀은 현재 인스턴스에 맞는 접속 명령을 출력합니다. MLflow UI에는 자체 "
            "인증이 없으므로 `--host 0.0.0.0`으로 외부에 직접 공개하지 마십시오."
        ),
        code(
            "# 실행하지 않고 포트포워딩 안내만 출력합니다.\n"
            "print(mlflow_utils.port_forward_hint(port=5000))"
        ),
        md(
            "## 3. MLflow App 준비\n\n"
            "여러 사람이 같은 실험을 조회하거나 SageMaker 학습 컨테이너의 단계별 지표까지 기록하려면 "
            "관리형 MLflow를 사용합니다. 이 저장소는 새 환경을 MLflow App으로 구성하며, 기존 "
            "Tracking Server가 있다면 새 App을 만들지 않고 해당 ARN을 `MLFLOW_TRACKING_URI`에 "
            "지정할 수 있습니다.\n\n"
            "먼저 현재 리전의 App 목록을 확인한 뒤 생성 셀을 실행하십시오. 같은 이름의 App이 있으면 "
            "새로 만들지 않고 기존 App을 재사용합니다."
        ),
        code(
            "# 현재 리전의 App 목록을 조회합니다.\n"
            "for app in mlflow_utils.list_apps(region=REGION) or [{'name': '(없음)'}]:\n"
            "    print(app)"
        ),
        code(
            "# 이 셀은 AWS 리소스를 만듭니다. 목록을 확인한 뒤 실행하세요.\n"
            "APP_NAME = os.environ.get('MLFLOW_APP_NAME') or mlflow_utils.DEFAULT_APP_NAME\n"
            "info = mlflow_utils.ensure_app(APP_NAME, region=REGION, wait=False)\n"
            "info"
        ),
        md(
            "### 생성 완료 대기\n\n"
            "App 생성에는 시간이 걸릴 수 있습니다. 앞 셀은 생성 요청만 제출하므로 다음 셀에서 준비 "
            "상태가 될 때까지 기다립니다.\n\n"
            "App ARN의 마지막 값은 App 이름이 아니라 AWS가 생성한 ID입니다. ARN을 직접 조립하지 "
            "말고 생성 결과나 `find_app()` 조회 결과를 사용하십시오."
        ),
        code(
            "ready = mlflow_utils.wait_for_app(info['arn'], region=REGION)\n"
            "print(ready['status'], ready['mlflow_version'])\n"
            "TRACKING_URI = ready['arn']"
        ),
        md(
            "### 관리형 UI 열기\n\n"
            "다음 셀은 인증 정보가 포함된 일회용 사전 서명 URL을 만듭니다. URL을 발급한 뒤 바로 열고, "
            "다른 사람과 공유하거나 저장소에 남기지 마십시오. 브라우저 세션을 만든 뒤에는 파이프라인 "
            "로그에 출력되는 실행 기록 링크를 사용할 수 있습니다."
        ),
        code(
            "# 인증이 포함된 URL을 출력합니다. 커밋 전에 출력을 지우세요.\n"
            "print(mlflow_utils.app_ui_url(APP_NAME, region=REGION))"
        ),
        md(
            "### IAM 확인\n\n"
            "권한은 노트북을 실행하는 사용자 또는 역할과 SageMaker 학습 실행 역할에 각각 필요합니다. "
            "노트북 실행자는 App 관리와 UI 접근 권한이 필요하고, 학습 실행 역할은 학습 지표를 기록할 "
            "수 있어야 합니다.\n\n"
            "다음 셀은 두 주체의 대표 권한을 확인합니다. 권한 확인 API 자체가 거부되면 "
            "`iam/mlflow-full-policy.json`과 `iam/mlflow-training-role-policy.json`을 기준으로 "
            "연결된 정책을 직접 확인하십시오."
        ),
        code(
            "perms = mlflow_utils.check_mlflow_permissions()\n"
            "perms"
        ),
        md(
            "### 학습 실행 역할에 권한 추가\n\n"
            "학습 실행 역할의 권한이 부족하면 다음 셀이 최소 권한 정책을 만들고 역할에 연결합니다. "
            "공유 실행 역할의 IAM 정책을 변경하므로 대상 역할과 출력 내용을 확인한 뒤 실행하십시오."
        ),
        code(
            "role_status = perms.get('training execution role', {})\n"
            "if role_status.get('ok') is False:\n"
            "    print(mlflow_utils.grant_mlflow_to_role())\n"
            "else:\n"
            "    print('권한 변경이 필요하지 않습니다.')"
        ),
        md(
            "## 4. 파이프라인 실행\n\n"
            "`.env`에서 `USE_MLFLOW=1`과 `MLFLOW_APP_NAME`을 설정하면 파이프라인이 같은 리전의 App을 "
            "이름으로 찾습니다. App ARN을 별도 설정 파일에 저장할 필요는 없습니다.\n\n"
            "```bash\n"
            "USE_MLFLOW=1 python pipelines/run_extraction.py --stages all\n"
            "```\n\n"
            "파이프라인 전체는 상위 run으로 기록되고 SageMaker 학습 작업은 연결된 하위 run으로 "
            "기록됩니다. 모델 파일은 MLflow에 중복 업로드하지 않고 SageMaker가 생성한 S3 URI만 "
            "파라미터로 기록합니다.\n\n"
            "다음 셀은 App의 MLflow 버전과 현재 클라이언트 버전을 비교합니다."
        ),
        code(
            "# client와 server의 MLflow version을 비교합니다.\n"
            "managed = mlflow_utils.resolve_target(info['arn'])\n"
            "detail = mlflow_utils.describe_managed(managed, region=REGION)\n"
            "print(detail)\n"
            "mlflow_utils.warn_version_gap(detail)"
        ),
        md(
            "## 5. 정리\n\n"
            "실험 기록과 App을 더 이상 보존할 필요가 없을 때만 삭제 셀의 주석을 해제하십시오. App을 "
            "삭제해도 S3 아티팩트는 자동으로 지워지지 않으므로 출력된 S3 경로를 확인한 뒤 별도로 "
            "정리해야 합니다."
        ),
        code(
            "# 필요할 때만 주석을 해제하세요.\n"
            "# mlflow_utils.delete_app(APP_NAME, region=REGION)\n"
            "# print(info['artifact_store_uri'])\n"
            "# !aws s3 rm {info['artifact_store_uri']} --recursive"
        ),
        md(
            "## 더 읽기\n\n"
            "- `docs/mlflow.md`: MLflow가 필요한 경우, 구성 방식과 파이프라인 기록 범위\n"
            "- `common/mlflow_utils.py`: 기록 대상 선택, App 관리와 권한 처리 구현\n"
            "- [AWS Managed MLflow 공식 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html): "
            "지원 리전, 버전과 서비스 제한"
        ),
    ]


def main() -> None:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_notebook(cells()), f, ensure_ascii=False, indent=1)
    print(f"generated {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
