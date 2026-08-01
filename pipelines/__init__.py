"""
pipelines — 코스별 E2E 를 **평범한 파이썬**으로 돌리는 두 번째 진입점.

    python pipelines/run_extraction.py --stages data,train    # model_data 를 기록
    python pipelines/run_extraction.py --stages deploy        # 그 값을 읽어 배포

🔴 tracks/*/scripts/ 와 혼동하지 말 것 — 그쪽은 **SageMaker 컨테이너 안에서** 도는 학습 코드
   (train.py / train_grpo.py / train_mm.py)이고, 이 폴더는 **개발자 머신에서** 그것을 제출·감시하는
   오케스트레이션 코드다.

노트북(tracks/*/*.ipynb)은 그대로 남는다 — 여기는 같은 로직의 두 번째 진입점이지 대체가 아니다.
설정은 커밋되는 <repo>/config.yaml, 시크릿은 env(HF_TOKEN / SAGEMAKER_ROLE_ARN / AWS_REGION).
스테이지 간 값(model_data / endpoint_name ...)은 .pipeline_state/<course>.json 으로 넘긴다
(%store 는 IPython 전용이고 코스를 넘어 전역이라 코스 간 오염이 났다 — pipelines/_common.py 참고).

라이브러리 위생: 이 __init__ 은 무거운 것을 import 하지 않는다(argparse 오류가 밀리초 안에 나야 한다).
"""

__all__ = ["_common", "_config"]
