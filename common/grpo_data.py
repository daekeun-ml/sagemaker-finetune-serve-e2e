"""GRPO 학습용 프롬프트 데이터를 준비합니다.

SFT에 사용한 데이터를 그대로 재사용하면 데이터 누출이 생기고 학습 신호가 약해질 수 있습니다.
프롬프트 소스는 `holdout`, `synth`, `failures`를 지원하며 권장 순서는
`failures`, `synth`, `holdout`입니다.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

SOURCES = ("holdout", "synth", "failures")


# ---------------------------------------------------------------------------
# 공통: {"messages": [...]} JSONL 읽기/쓰기
# ---------------------------------------------------------------------------
def _read_messages_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_grpo_jsonl(rows: list[dict], path: str, *, min_rows: int = 1) -> str:
    """GRPO 학습 파일을 저장합니다. 데이터가 부족하면 업로드 전에 실패합니다."""
    if len(rows) < min_rows:
        raise ValueError(
            f"GRPO 학습 데이터가 {len(rows)}건입니다. 최소 {min_rows}건이 필요합니다.\n"
            "커널 캐시, Bedrock 합성 로그, failures 소스의 평가 결과를 확인하세요.\n"
            "즉시 진행해야 하면 GRPO_PROMPT_SOURCE='holdout'을 사용할 수 있습니다."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {path} ({len(rows)}건)")
    return path


# ---------------------------------------------------------------------------
# 1) holdout: SFT가 사용하지 않은 구간
# ---------------------------------------------------------------------------
def from_holdout(train_jsonl: str, n: int, *, sft_used: int) -> list[dict]:
    """SFT가 사용한 앞부분을 제외하고 최대 n건을 반환합니다."""
    rows = _read_messages_jsonl(train_jsonl)
    tail = rows[sft_used:]
    if not tail:
        raise ValueError(
            f"{train_jsonl}에 SFT 미사용 구간이 없습니다(전체 {len(rows)}건, 사용 {sft_used}건).\n"
            "데이터를 더 만들거나 GRPO_PROMPT_SOURCE='synth' 또는 'failures'를 사용하세요."
        )
    if len(tail) < n:
        print(f"미사용 데이터가 {len(tail)}건뿐이므로 요청한 {n}건 대신 가능한 만큼 사용합니다.")
    return tail[:n]


# ---------------------------------------------------------------------------
# 2) synth: Bedrock으로 프롬프트 생성
# ---------------------------------------------------------------------------
# 난이도 지시는 생성 프롬프트에만 적용하고 critique 기준은 유지합니다.
_HARDEN = {
    "extraction": (
        " IMPORTANT: these examples are for reinforcement-learning (GRPO), so they must be "
        "CHALLENGING, not typical: every generated call MUST carry at least TWO typed arguments "
        "(never an empty arguments object), and the user text must express those values "
        "INDIRECTLY so the model has to infer them (e.g. 'next Friday' rather than a date, "
        "'the cheaper option' rather than a number). Vary function names and argument shapes."
    ),
    "classification": (
        " IMPORTANT: these examples are for reinforcement-learning (GRPO), so they must be "
        "CHALLENGING: prefer messages that sit near the boundary between two similar intents, "
        "use indirect or emotional phrasing, and cover the rarer intents rather than the "
        "most common ones. Still assign exactly one correct label."
    ),
}


def harden_instruction(task_instruction: str, kind: str) -> str:
    """task_instruction에 GRPO용 난이도 제약을 추가합니다."""
    return task_instruction + _HARDEN.get(kind, "")


def from_synth(*, task_instruction: str, seed_texts: list[str], n: int,
               model_id: str, region: str,
               to_messages: Callable[[dict], list[dict[str, str]]],
               kind: str = "",
               harden: bool = True,
               # 생성 호출보다 critique 호출이 많으므로 병렬도를 별도로 조정할 수 있습니다.
               max_workers: int = 12,
               batch_size: int = 10,
               progress_cb: Callable[[int, int], None] | None = None) -> list[dict]:
    """Bedrock으로 GRPO 프롬프트와 채점용 reference를 생성합니다."""
    from common.synth import bedrock_synth as bs

    gen_instruction = harden_instruction(task_instruction, kind) if harden else None
    if gen_instruction and gen_instruction != task_instruction:
        print(f"생성 프롬프트에 난이도 조건을 적용합니다(kind={kind}).")

    print(f"합성 시작: 목표 {n}건, 동시 호출 {max_workers}, 배치 크기 {batch_size}")
    print(f"예상 호출: 생성 약 {max(1, n // batch_size)}회, critique 약 {n}회")

    examples = bs.generate_grounded(
        task_instruction=task_instruction,
        gen_instruction=gen_instruction,
        seed_texts=seed_texts,
        n_total=n,
        model_id=model_id,
        region=region,
        to_messages=to_messages,
        max_workers=max_workers,
        batch_size=batch_size,
        progress_cb=progress_cb,
    )
    return [{"messages": ex.messages} for ex in examples]


# ---------------------------------------------------------------------------
# 3) failures: SFT 모델이 틀린 데이터
# ---------------------------------------------------------------------------
def _is_failure(pred: Any, gold: str, kind: str) -> bool:
    """예측이 실패인지 판정. 04_evaluate 의 지표와 같은 기준을 쓴다."""
    p = "" if pred is None else str(pred).strip()
    g = (gold or "").strip()
    if not p:
        return True
    if kind == "extraction":
        # JSON 파싱 실패와 함수명 또는 인자 불일치를 실패로 처리합니다.
        try:
            po = json.loads(p)
        except (json.JSONDecodeError, TypeError):
            return True
        try:
            go = json.loads(g)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(po, dict) or not isinstance(go, dict):
            return True
        if po.get("name") != go.get("name"):
            return True
        return po.get("arguments") != go.get("arguments")
    if kind == "classification":
        return p.lower() != g.lower()          # 라벨 정확 일치
    # 자유서술 과제는 프로그램으로 정확히 채점하기 어려워 지원하지 않습니다.
    raise ValueError(
        f"reward_kind={kind!r} 는 프로그램적 채점이 어려워 failures 수집을 지원하지 않습니다. "
        "GRPO는 추출과 분류 트랙에만 제공됩니다.")


def from_failures(heldout: list[dict], preds: list[Any], *, kind: str,
                  to_messages: Callable[[dict], list[dict[str, str]]],
                  max_n: int | None = None) -> list[dict]:
    """평가에서 틀린 데이터만 골라 GRPO 메시지 형식으로 반환합니다."""
    if len(heldout) != len(preds):
        raise ValueError(f"heldout({len(heldout)})과 preds({len(preds)}) 길이가 다릅니다.")
    fails = [ex for ex, pr in zip(heldout, preds) if _is_failure(pr, ex.get("output", ""), kind)]
    print(f"실패 사례: {len(fails)}/{len(heldout)} ({len(fails)/max(1,len(heldout)):.0%})")
    if not fails:
        raise ValueError(
            "SFT 모델이 held-out 데이터를 모두 맞혀 실패 사례가 없습니다.\n"
            "N_EVAL을 늘리거나 GRPO_PROMPT_SOURCE='synth'를 사용하세요."
        )
    if max_n:
        fails = fails[:max_n]
    return [{"messages": to_messages(ex)} for ex in fails]


# ---------------------------------------------------------------------------
# 진단: 이 데이터가 GRPO에 쓸 만한가
# ---------------------------------------------------------------------------
def describe(rows: list[dict], *, source: str) -> dict:
    """GRPO 학습 데이터 요약 + advantage 관점의 경고."""
    n = len(rows)
    users = []
    for r in rows:
        msgs = r.get("messages") or []
        u = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        users.append(str(u))
    uniq = len(set(users))
    info = {"source": source, "n": n, "unique_prompts": uniq}
    print(f"GRPO 데이터: {n}건, 고유 프롬프트 {uniq}건, source={source}")
    if uniq < n:
        print(f"중복 프롬프트가 {n - uniq}건 있습니다.")
    if source == "holdout":
        print("holdout은 SFT와 같은 분포이므로 학습 신호가 약할 수 있습니다.")
        print("reward가 변하지 않으면 'failures' 또는 'synth'를 사용하세요.")
    return info
