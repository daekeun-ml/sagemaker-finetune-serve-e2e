"""
common/logging_utils.py — 로깅 설정 헬퍼 (앱/노트북 진입점에서 1회 호출)

파이썬 로깅 정석:
  - **라이브러리 모듈**(common/*)은 핸들러를 설정하지 않는다. `logging.getLogger(__name__)`만 쓰고,
    출력 방식(핸들러·레벨·포맷) 결정은 **애플리케이션의 책임**으로 남긴다.
    (common/__init__.py 가 패키지 루트에 NullHandler를 달아 import 부작용을 0으로 만든다.)
  - **앱/노트북/스크립트 진입점**만 여기 setup_logging()을 1회 호출해 핸들러를 구성한다.

사용:
    from common.logging_utils import setup_logging
    setup_logging()                 # 노트북 최상단에서 1회 (LOG_LEVEL env 존중)
    import logging; log = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 이 킷의 로거 네임스페이스 루트. common.* / gemma.* 모두 이 아래로 전파된다.
TOOLKIT_LOGGER = "gemma_e2e"

# 시끄러운 서드파티 로거 (원치 않는 DEBUG 소음 억제)
_NOISY = ("botocore", "boto3", "urllib3", "s3transfer", "sagemaker", "httpx", "httpcore")

_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    *,
    fmt: str = DEFAULT_FORMAT,
    stream=None,
    quiet_third_party: bool = True,
    force: bool = False,
) -> logging.Logger:
    """루트 로깅을 1회 구성(멱등). 앱/노트북 진입점에서 호출.

    level: 문자열("INFO")/정수/None(=env LOG_LEVEL, 기본 INFO).
    force: True면 이미 구성됐어도 핸들러를 교체(노트북에서 레벨 바꿔 재호출 시).
    반환: 이 킷의 루트 로거(gemma_e2e) — 필요하면 직접 써도 됨.
    """
    global _CONFIGURED
    level = _resolve_level(level)
    root = logging.getLogger()

    if force or not _CONFIGURED:
        # 노트북/재실행 환경에서 핸들러 중복 누적 방지: 기존 핸들러 정리 후 1개만.
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=DATE_FORMAT))
        root.addHandler(handler)
        _CONFIGURED = True

    root.setLevel(level)
    logging.getLogger(TOOLKIT_LOGGER).setLevel(level)

    if quiet_third_party:
        for name in _NOISY:
            logging.getLogger(name).setLevel(max(logging.WARNING, level))

    return logging.getLogger(TOOLKIT_LOGGER)


def get_logger(name: str) -> logging.Logger:
    """킷 네임스페이스(gemma_e2e.<name>) 로거. 라이브러리 모듈이 __name__ 대신 써도 됨."""
    return logging.getLogger(f"{TOOLKIT_LOGGER}.{name}")


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return logging.getLevelName(level.upper()) if not level.isdigit() else int(level)
    return int(level)
