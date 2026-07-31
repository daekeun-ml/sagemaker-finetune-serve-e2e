"""
common/display_utils.py — 노트북에서 추론 결과를 읽기 좋게 렌더링

왜 필요한가: `print(pred[:400])` 로 찍으면 (1) 요약·QA처럼 긴 응답이 잘리고, (2) 추출 트랙의
JSON이 한 줄로 뭉쳐 구조가 안 보이고, (3) 입력 문서가 길면 셀 출력을 뒤덮어 정작 봐야 할
예측이 스크롤 밖으로 밀린다. 그래서 입력은 접어 두고(<details>) 예측은 형식에 맞게 강조한다.

🔴 노트북 밖(스크립트·CI·pytest)에서도 import·호출이 안전해야 한다 → IPython이 없거나 커널이
   아니면 평문 print로 자동 폴백한다(예외를 던지지 않는다).
"""
from __future__ import annotations

import html
import json
from typing import Any

__all__ = ["show_inference", "stream_inference", "show_image_inference",
           "as_markdown", "in_notebook"]


def in_notebook() -> bool:
    """IPython 커널(주피터/VS Code) 안에서 실행 중인지. 콘솔·스크립트면 False."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    # ZMQInteractiveShell = 주피터 커널. TerminalInteractiveShell(ipython 콘솔)은 렌더링 불가.
    return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"


def _pretty(text: Any) -> tuple[str, str]:
    """예측 문자열을 (렌더용 본문, 코드펜스 언어) 로 정규화.

    JSON이면 indent를 넣어 구조를 보이게 하고, 아니면 원문 그대로 둔다.
    ```json 펜스로 감싸면 노트북이 문법 강조까지 해 준다.
    """
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = s.strip()
    # 모델이 ```json ... ``` 로 감싸 답하는 경우가 흔하다 → 펜스를 벗겨야 파싱된다.
    if s.startswith("```"):
        body = s.split("\n", 1)[1] if "\n" in s else ""
        s = body.rsplit("```", 1)[0].strip() if "```" in body else body.strip()
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s, ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2), "json"
    return s, ""


def as_markdown(input_text: str, prediction: Any, index: int | None = None,
                title: str = "실시간 추론", input_preview: int = 400) -> str:
    """렌더링할 마크다운 문자열을 반환(테스트·재사용 용이하도록 표시와 분리)."""
    body, lang = _pretty(prediction)
    head = f"##### {title}" + (f" {index}" if index is not None else "")

    inp = (input_text or "").strip()

    def _pre(t: str, small: bool = False) -> str:
        # 🔴 반드시 <pre>로 감싼다 — <sub>나 평문으로 두면 내용이 **마크다운으로 해석**된다.
        #    실측(billsum 법안): 입력에 백틱이 79개(``and'' 같은 구식 인용부호) 들어 있어
        #    미리보기가 인라인 코드스팬을 열어 버리고, 홀수로 끝나면 뒤의 **PREDICTION**
        #    까지 삼켜 "응답이 안 보이는" 증상이 났다. html.escape는 <>&만 처리하고
        #    백틱·*·_ 는 통과시키므로 escape만으로는 부족하다. <pre> 안은 마크다운 비활성.
        style = "white-space:pre-wrap" + (";font-size:0.85em;opacity:0.7" if small else "")
        return f"<pre style='{style}'>{html.escape(t)}</pre>"

    # 긴 입력만 <details>로 접는다 — 요약 트랙은 입력이 수천 자라 펼쳐 두면 예측이 스크롤 밖으로
    # 밀린다. 짧으면 접을 이유가 없다(미리보기와 전문이 같아 같은 내용이 두 번 보인다).
    if len(inp) <= input_preview:
        input_block = f"**INPUT** ({len(inp):,}자)\n\n{_pre(inp)}"
    else:
        input_block = (f"<details><summary><b>INPUT</b> ({len(inp):,}자) — 클릭하면 전체</summary>"
                       f"{_pre(inp)}</details>\n\n"
                       f"{_pre(inp[:input_preview] + ' …', small=True)}")
    parts = [head, input_block, "**PREDICTION**", _render_body(body, lang)]
    return "\n\n".join(parts)


def _render_body(body: str, lang: str) -> str:
    """예측 본문 렌더 — JSON은 코드펜스, 평문은 <pre>.

    🔴 평문을 '> 인용문'으로 감싸면 안 된다: 예측에 백틱이나 '>'가 들어오면(요약이 원문의
       ``인용'' 을 그대로 옮기는 일이 흔하다) 마크다운이 깨져 응답이 사라진다.
       <pre>는 그런 문자를 전부 문자 그대로 보여 준다.
    """
    if lang:
        # JSON은 이미 json.dumps 결과라 백틱이 들어올 수 없다(있으면 문자열 안에 escape됨).
        return f"```{lang}\n{body}\n```"
    return f"<pre style='white-space:pre-wrap'>{html.escape(body)}</pre>"


def show_inference(input_text: str, prediction: Any, index: int | None = None,
                   title: str = "실시간 추론", input_preview: int = 400) -> None:
    """추론 1건을 노트북에 렌더링(노트북 밖이면 평문 출력).

    사용:
        for i, ex in enumerate(holdout, 1):
            show_inference(ex['input'], predict(ex['input']), index=i)
    """
    if not in_notebook():
        body, _ = _pretty(prediction)
        bar = f"===== {title}" + (f" {index}" if index is not None else "") + " ====="
        print(f"\n{bar}\nINPUT     : {(input_text or '')[:input_preview]}\nPREDICTION: {body}")
        return
    from IPython.display import Markdown, display
    display(Markdown(as_markdown(input_text, prediction, index, title, input_preview)))


def stream_inference(input_text: str, pieces: Any, index: int | None = None,
                     title: str = "실시간 추론(스트리밍)", input_preview: int = 400) -> str:
    """생성 조각(iterable)을 **받는 즉시 화면에 갱신**하고, 완성 문자열을 반환한다.

    요약처럼 응답이 긴 트랙에서 완성까지 기다리지 않게 한다(실측 첫 토큰 0.5s vs 전체 9.2s).

    사용:
        from common import aws_utils
        pieces = aws_utils.stream_sagemaker_chat(endpoint_name, msgs, region=config.AWS_REGION)
        text = stream_inference(ex['input'], pieces, index=1)

    반환값을 그대로 쓰면 스트리밍 후 후처리(파싱·저장)도 가능하다.
    """
    if not in_notebook():   # 스크립트/CI: 조각을 그대로 흘려 출력
        buf = []
        print(f"\n===== {title}" + (f" {index}" if index is not None else "") + " =====")
        for p in pieces:
            buf.append(p)
            print(p, end="", flush=True)
        print()
        return "".join(buf)

    from IPython.display import Markdown, display
    head = f"##### {title}" + (f" {index}" if index is not None else "")
    inp = (input_text or "").strip()
    # 입력 블록은 한 번만 그리고, 그 아래 예측만 갱신한다(입력을 매 조각마다 다시 그리면 느리다).
    display(Markdown(as_markdown(inp, "", index=index, title=title,
                                 input_preview=input_preview).rsplit("**PREDICTION**", 1)[0]))
    handle = display(Markdown("**PREDICTION** *(생성 중…)*"), display_id=True)
    buf: list[str] = []
    for p in pieces:
        buf.append(p)
        # 🔴 스트리밍 중에는 <pre>로만 그린다 — 조각 단위로는 JSON이 미완성이고, 백틱이
        #    반쪽만 도착하면 마크다운이 깨진다(완성 후 아래에서 정식 렌더로 교체한다).
        handle.update(Markdown("**PREDICTION** *(생성 중…)*\n\n"
                               + f"<pre style='white-space:pre-wrap'>{html.escape(''.join(buf))}</pre>"))
    text = "".join(buf)
    body, lang = _pretty(text)          # 완성 후: JSON이면 들여쓰기·문법강조로 다시 렌더
    handle.update(Markdown("**PREDICTION**\n\n" + _render_body(body, lang)))
    return text


def show_image_inference(image: Any, prediction: Any, index: int | None = None,
                         title: str = "이미지 추론", max_width: int = 360) -> None:
    """멀티모달 추론 1건 — 입력 이미지와 예측을 함께 렌더링.

    image: PIL.Image (노트북 밖이면 크기만 출력하고 렌더링은 건너뜀).
    """
    body, lang = _pretty(prediction)
    if not in_notebook():
        bar = f"===== {title}" + (f" {index}" if index is not None else "") + " ====="
        print(f"\n{bar}\nINPUT     : <image {getattr(image, 'size', '?')}>\nPREDICTION: {body}")
        return
    from IPython.display import Markdown, display
    head = f"##### {title}" + (f" {index}" if index is not None else "")
    display(Markdown(head))
    try:  # 이미지는 Markdown에 못 실으므로 별도 display (thumbnail로 축소 — 영수증은 세로로 길다)
        thumb = image.copy()
        thumb.thumbnail((max_width, max_width * 4))
        display(thumb)
    except Exception:  # noqa: BLE001 — 렌더링 실패가 추론 결과 확인을 막지 않도록
        print("(이미지 렌더링 실패 — 예측만 표시합니다)")
    display(Markdown("**PREDICTION**\n\n" + _render_body(body, lang)))
