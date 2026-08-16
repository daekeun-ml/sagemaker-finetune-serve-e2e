"""도메인 QA 코스를 실행합니다.

코스별 설정과 데이터 변환은 `tracks/04_domain_qa`에 있고,
스테이지 구현은 `pipelines._common`에 있습니다. 이 코스는 GRPO를 지원하지 않습니다.
"""
from __future__ import annotations

import os
import sys

# 파일을 직접 실행해도 pipelines 패키지를 찾도록 저장소 루트를 추가합니다.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="domain_qa"))
