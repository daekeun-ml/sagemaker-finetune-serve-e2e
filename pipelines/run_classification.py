"""
pipelines/run_classification.py — 텍스트 분류(intent) 코스를 평범한 파이썬으로 E2E 실행.

    python pipelines/run_classification.py --stages all              # data→train→grpo→deploy→eval
    python pipelines/run_classification.py --dry-run                 # 과금 리소스 0으로 전 경로 검증
    python pipelines/run_classification.py --stages data,train       # model_data 를 상태 파일에 기록
    python pipelines/run_classification.py --stages deploy,eval      # 그 값을 읽어 배포·평가
    python pipelines/run_classification.py --stages cleanup          # 🔴 endpoint 삭제(시간당 과금 정지)

같은 로직의 노트북: tracks/02_classification/*.ipynb (에이전틱 단계는 노트북 전용).
설정은 <repo>/config.yaml, 시크릿은 env — 스테이지 구현은 pipelines/_common.py 에 있다.

🔴 이 코스가 다른 코스와 다른 점 — 값 자체는 여기 두지 않는다(두 곳에 두면 노트북과 CLI 가 어긋난다):
  · GRPO 스테이지가 있다(노트북 02a). TrackSpec.grpo_reward_kind='classification' 이라 라벨 일치를
    프로그램으로 채점할 수 있어서다 — 요약·QA 코스는 이 값이 비어 있어 grpo 가 자동 skip 된다.
    reward 종류·학습 길이·서빙 컨텍스트는 tracks/build_all_tracks.py 의 이 코스 TrackSpec 이 정한다.
  · 시드·평가 지표(banking77 → accuracy/macro-F1)는 tracks/02_classification/track_data.py 가 정한다.
    라벨 목록은 반드시 그쪽 load_label_names() 로 얻는다(PolyAI/banking77 직접 로드는 실패한다).
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_classification.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines` 가
# 안 된다 → 리포 루트를 넣어 준다(python -m pipelines.run_classification 로 실행할 때는 이미 들어 있다).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="classification"))
