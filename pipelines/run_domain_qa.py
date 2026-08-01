"""
pipelines/run_domain_qa.py — 도메인 QA / instruction 코스를 평범한 파이썬으로 E2E 실행.

    python pipelines/run_domain_qa.py --stages all              # data → train → deploy → eval
    python pipelines/run_domain_qa.py --dry-run                 # 과금 리소스 0으로 전 경로 검증
    python pipelines/run_domain_qa.py --stages data,train       # model_data 를 상태 파일에 기록
    python pipelines/run_domain_qa.py --stages deploy,eval      # 그 값을 읽어 배포·평가
    python pipelines/run_domain_qa.py --stages cleanup          # 🔴 endpoint 삭제(시간당 과금 정지)

같은 로직의 노트북: tracks/04_domain_qa/*.ipynb (에이전틱 단계는 노트북 전용).
설정은 <repo>/config.yaml, 시크릿은 env — 스테이지 구현은 pipelines/_common.py 에 있다.

이 코스에 대해 알아 둘 것 (값의 출처는 전부 다른 파일이다 — 여기 복제하지 않는다):
  · 시드는 databricks/databricks-dolly-15k (cc-by-sa-3.0, ungated). instruction + 선택적 context
    → input, response → output 으로 접는다(tracks/04_domain_qa/track_data.py).
    🔴 CC-BY-SA 는 share-alike 다 — 이 데이터로 만든 모델·데이터셋을 배포하면 라이선스가 전파된다.
  · dolly 는 train 스플릿만 있다 → held-out 은 학습 구간 뒤를 결정론적으로 잘라 쓴다
    (stage_data 가 eval.jsonl 로 남긴다. 합성 데이터로 평가하면 teacher 모방도만 재게 된다).
  · 지표는 Bedrock LLM-judge(correctness/helpfulness/groundedness)가 주, ROUGE-L 은 보조다 —
    정답이 자유형 문장이라 exact-match 가 성립하지 않는다(common/eval_utils.py).
  · 🔴 grpo 스테이지가 없다. reward 를 프로그램으로 채점할 수 없어서이며, 요청하면 조용히
    건너뛰지 않고 이유와 함께 **거부**한다(_common.unsupported_reason).
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_domain_qa.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines` 가
# 안 된다 → 리포 루트를 넣어 준다(python -m pipelines.run_domain_qa 로 실행할 때는 이미 들어 있다).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="domain_qa"))
