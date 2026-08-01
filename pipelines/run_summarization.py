"""
pipelines/run_summarization.py — 문서 요약 코스를 평범한 파이썬으로 E2E 실행.

    python pipelines/run_summarization.py --dry-run                 # 과금 리소스 0으로 전 경로 검증
    python pipelines/run_summarization.py --stages data,train       # model_data 를 상태 파일에 기록
    python pipelines/run_summarization.py --stages deploy,eval      # 그 값을 읽어 배포·평가
    python pipelines/run_summarization.py --stages cleanup          # 🔴 endpoint 삭제(시간당 과금 정지)

같은 로직의 노트북: tracks/03_summarization/*.ipynb (에이전틱 단계는 노트북 전용).
설정은 <repo>/config.yaml, 시크릿은 env — 스테이지 구현은 pipelines/_common.py 에 있다.
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_summarization.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines` 가
# 안 된다 → 리포 루트를 넣어 준다(python -m pipelines.run_summarization 로 실행할 때는 이미 들어 있다).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="summarization"))
