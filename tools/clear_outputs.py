#!/usr/bin/env python
r"""노트북 셀 출력과 실행 번호를 제거합니다.

출력은 diff와 파일 크기를 키우고 엔드포인트 이름 같은 계정 정보를 남길 수 있습니다.

빌더(`tracks/*/_build_notebooks.py`, `tracks/build_all_tracks.py`)를 돌리면 노트북이 새로
생성되므로 출력도 함께 사라진다. 이 스크립트는 **재빌드 없이** 출력만 지울 때 쓴다
(직접 수정한 노트북이 있거나, 커밋 직전 정리용).

사용:
    python tools/clear_outputs.py                 # 리포 전체 .ipynb
    python tools/clear_outputs.py tracks/03_summarization
    python tools/clear_outputs.py --check         # 지우지 않고 목록만(CI/커밋 훅용, 남아 있으면 exit 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def clear(path: Path, check_only: bool = False) -> tuple[bool, int, float]:
    """(변경됨?, 지운 출력 셀 수, 절약 KB). check_only면 파일을 쓰지 않는다."""
    original = path.read_text(encoding="utf-8")
    nb = json.loads(original)
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            n += 1
        cell["outputs"] = []
        cell["execution_count"] = None
    if not n:
        return False, 0, 0.0
    # 빌더와 같은 직렬화 형식(indent=1, ensure_ascii=False)으로 써서 불필요한 diff를 막는다.
    new = json.dumps(nb, ensure_ascii=False, indent=1)
    saved = (len(original) - len(new)) / 1024
    if not check_only:
        path.write_text(new, encoding="utf-8")
    return True, n, saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None,
                    help="파일 또는 디렉토리(생략 시 리포 전체)")
    ap.add_argument("--check", action="store_true",
                    help="지우지 않고 검사만 수행하며 출력이 남아 있으면 exit 1")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    targets: list[Path] = []
    for p in (args.paths or [repo]):
        p = Path(p)
        targets.extend(sorted(p.rglob("*.ipynb")) if p.is_dir() else [p])
    # 서드파티와 사본 디렉토리는 검사하지 않습니다.
    SKIP = {".venv", "venv", ".git", ".ipynb_checkpoints", "node_modules", "__pycache__"}
    targets = [t for t in targets if not (SKIP & set(t.parts))]

    dirty, total_saved = 0, 0.0
    for t in targets:
        changed, n, saved = clear(t, check_only=args.check)
        if changed:
            dirty += 1
            total_saved += saved
            verb = "남아 있음" if args.check else "정리"
            # 리포 밖 경로를 인자로 줄 수도 있으니 relative_to 실패를 허용한다.
            try:
                shown = t.relative_to(repo)
            except ValueError:
                shown = t
            print(f"  {verb}: {shown}  (셀 {n}개, {saved:.0f}KB)")

    if not dirty:
        print(f"노트북 {len(targets)}개에 출력이 없습니다.")
        return 0
    if args.check:
        print(f"\n노트북 {dirty}개에 출력이 남아 있습니다. `python tools/clear_outputs.py`를 실행하세요.")
        return 1
    print(f"\n노트북 {dirty}개를 정리했습니다. 절약한 용량: {total_saved:.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
