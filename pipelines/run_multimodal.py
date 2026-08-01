"""
pipelines/run_multimodal.py — 멀티모달 추출(영수증 이미지→JSON) 코스를 평범한 파이썬으로 E2E 실행.

    python pipelines/run_multimodal.py --stages all              # data → train → deploy
    python pipelines/run_multimodal.py --stages data,train       # model_data 를 상태 파일에 기록
    python pipelines/run_multimodal.py --stages deploy,eval      # 그 값을 읽어 배포
    python pipelines/run_multimodal.py --dry-run                 # 과금 리소스 0으로 전 경로 검증
    python pipelines/run_multimodal.py --stages cleanup          # 🔴 endpoint 삭제(시간당 과금 정지)

같은 로직의 노트북: tracks/05_multimodal_extraction/*.ipynb (에이전틱 단계는 노트북 전용).
설정은 <repo>/config.yaml, 시크릿은 env — 스테이지 구현은 pipelines/_common.py 에 있다.
상태 파일은 .pipeline_state/mm_extraction.json (`-h` 가 경로를 찍는다).

🔴 이 코스가 다른 네 코스와 **구조적으로** 다른 점 — 스테이지 목록부터 다르다.
   값은 전부 CourseSpec(_common._mm_track_spec)에 있고 여기서 다시 정의하지 않는다.

  · 스테이지: data → train → deploy 뿐이다 (grpo·eval 없음, `--stages all` 이 그 세 개로 해석된다).

  · grpo 없음 — `--stages grpo` 는 **거부**한다(조용히 건너뛰지 않는다: 사람이 GPU 몇 시간을
    기대하기 때문이다). scripts/ 에 train_grpo.py 가 아예 없고, prompt 소스 3종(holdout/synth/
    failures)이 전부 성립하지 않는다 — 이 코스의 prompt 는 이미지다. 전문은
    _common.unsupported_reason() 이 출력한다.

  · eval 없음 — 04 평가 노트북이 없는 코스다. 검증 지점은 deploy 스테이지의 스모크 추론이고,
    samples/ground_truth.json 의 정답과 눈으로 대조한다. `--stages deploy,eval` 은 다섯 코스에
    공통으로 안내하는 명령이라 거부하지 않고 이유만 찍는다(deploy 성공을 실패로 만들지 않는다).

  · 합성 데이터 단계 없음 — 이미지 합성은 별개 문제다. data 스테이지는 cord-v2 시드가 실제로
    로드되는지만 확인하고 S3 에 아무것도 올리지 않는다. 학습 데이터는 train_mm.py 가 컨테이너
    안에서 load_dataset 으로 받는다(이미지가 parquet 에 내장돼 로컬 왕복이 비싸다 — 캐시 없이
    1건 로드 ~40초 실측). 그래서 train 잡에 input 채널이 붙지 않는다.

  · 학습 스크립트가 train_mm.py 다 (train.py 아님). vision tower 는 freeze 하고 language
    submodule 에만 LoRA 를 건다 — 멀티모달 base 에 all-linear LoRA 를 붙이면 vision proj
    (ClippableLinear)에서 크래시한다(실측).

  · 서빙은 **멀티모달 그대로** 다 — 텍스트 코스의 re-export 단계가 여기엔 없다. 대신 이미지 입력을
    열어야 하므로 deploy 가 mm_limit={"image": 1} 을 넣는다(텍스트 코스는 이 키를 넣지 않는다).
    serve_max_model_len=2048 — 입력이 이미지라 텍스트 컨텍스트가 짧고, 아티팩트가 vision tower 를
    포함해 24GB GPU 에서 예산이 빠듯하다(가중치 15.18 GiB 실측 → max_num_seqs·mem_util 로 OOM 방지).
    gen_max_tokens=768 — 정답 JSON 이 최대 592 토큰(실측 100건)이라 512 로는 잘린다.
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_multimodal.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines` 가
# 안 된다 → 리포 루트를 넣어 준다(python -m pipelines.run_multimodal 로 실행할 때는 이미 들어 있다).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pipelines._common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_course="mm_extraction"))
