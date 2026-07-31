"""common — Gemma E2E 툴킷 공통 레이어 (라이브러리).

로깅 정석: 라이브러리는 핸들러를 설정하지 않는다. 패키지 루트 로거(gemma_e2e)에 NullHandler만 달아
import 부작용을 0으로 하고, 출력 구성(핸들러·레벨)은 앱/노트북이 common.logging_utils.setup_logging()
으로 결정한다. ('No handlers could be found' 경고 방지 + 라이브러리 위생.)
"""
import logging

logging.getLogger("gemma_e2e").addHandler(logging.NullHandler())
