"""
pipelines/run_extraction.py — 정보추출→JSON 코스(01, 플래그십)를 평범한 파이썬으로 E2E 실행.

    python pipelines/run_extraction.py --stages all              # data→train→grpo→deploy→eval
    python pipelines/run_extraction.py --dry-run                 # 과금 리소스 0으로 전 경로 검증
    python pipelines/run_extraction.py --stages data,train       # model_data 를 상태 파일에 기록
    python pipelines/run_extraction.py --stages deploy,eval      # 그 값을 읽어 배포·평가
    python pipelines/run_extraction.py --stages cleanup          # 🔴 endpoint 삭제(시간당 과금 정지)

이 코스만의 값은 전부 tracks/01_extraction_to_json 쪽에 있고 이 파일에는 다시 적지 않는다
(두 곳에 두면 노트북과 CLI 가 다른 설정으로 돈다 — _common.CourseSpec 독스트링 참고):
  · TrackSpec(_build_notebooks._flagship_spec): max_seq_length 2048(툴 스키마 JSON 이 길다),
    gen_max_tokens 256, grpo_reward_kind='extraction' → 이 코스는 **GRPO 단계가 있다**.
  · track_data.py: glaive-function-calling-v2(apache-2.0, ungated) 시드,
    SYSTEM_PROMPT + to_messages(system 을 첫 user 턴에 fold — Gemma instruct 가 system role 을 거부).
  · 평가는 eval_kind='extraction' → JSON 유효성 + 함수명/인자 F1(LLM-judge 아님, 무료).

같은 로직의 노트북: tracks/01_extraction_to_json/*.ipynb (05_agentic/06_agentcore 는 노트북 전용 —
질문을 바꿔 가며 답을 보는 탐색적 작업이라 배치로 얹을 이득이 없다).
설정은 <repo>/config.yaml, 시크릿은 env — 스테이지 구현은 pipelines/_common.py 에 있다.
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_extraction.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines` 가
# 안 된다 → 리포 루트를 넣어 준다(python -m pipelines.run_extraction 로 실행할 때는 이미 들어 있다).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="extraction"))
