"""Gemma E2E 예제의 공통 라이브러리입니다.

패키지에서는 `NullHandler`만 등록하고 출력 형식과 로그 수준은 실행 환경에서 설정합니다.
"""
import logging

logging.getLogger("gemma_e2e").addHandler(logging.NullHandler())
