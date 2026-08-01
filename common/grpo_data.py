"""
common/grpo_data.py — GRPO(RL) 학습용 prompt 소스 준비

🔴 왜 별도 모듈인가 — RL은 SFT와 **다른 데이터**가 필요하다:

| | SFT | GRPO/RL |
|---|---|---|
| 목적 | 정답을 모방해 형식·기본 능력 습득 | **SFT가 실패하는 구간**을 개선 |
| 필요한 것 | (입력, 정답) 쌍 | **prompt** (+ 프로그램적 채점용 reference) |
| 정답 사용 | 학습 입력으로 직접 사용 | reward 계산에만 사용(모델에 보여주지 않음) |

SFT에 쓴 데이터를 그대로 GRPO에 넣으면 두 가지 문제가 생긴다:
  1) **누출** — 모델이 이미 그 정답에 fit돼 있어 '학습한 것으로 다시 학습'하는 셈.
  2) **advantage ≈ 0 (더 근본적)** — GRPO는 prompt당 num_generations개를 생성해 **그룹 내 상대 비교**로
     학습한다. 이미 잘하는 prompt는 rollout 8개가 모두 만점이 되어 그룹 내 편차가 사라지고,
     advantage가 0에 수렴해 **gradient가 거의 흐르지 않는다**. 즉 시간·GPU만 쓰고 배우는 게 없다.
     → 슬라이스만 분리해도 같은 분포라면 이 문제는 그대로 남는다(누출만 막힘).

그래서 prompt 소스를 3가지로 나눈다(GRPO_PROMPT_SOURCE):

  'holdout'  — SFT가 쓰지 않은 구간. 무료·즉시. 누출은 막지만 같은 분포라 advantage 문제는 남는다.
               (튜토리얼 기본값: 추가 비용·선행 단계 없이 파이프라인을 끝까지 볼 수 있게)
  'synth'    — Bedrock으로 **prompt만** 생성. RL은 정답이 불필요하니 SFT 합성보다 싸고 빠르다.
               분포를 넓힐 수 있어 advantage 확보에 유리. (Bedrock 호출 과금)
  'failures' — 04_evaluate에서 **SFT 모델이 틀린 건만** 수집. reward 신호가 가장 강한 구간에
               집중하므로 실무에서 효율이 가장 좋다. (04_evaluate 선행 필요)

실무 권장 순서: failures > synth > holdout.
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
    """GRPO 학습 파일 저장. train_grpo.py 가 {"messages": [...]} 를 읽어 prompt/reference로 변환한다.

    🔴 0건(또는 min_rows 미만)이면 **여기서 즉시 실패**한다(실측 교훈 2026-07-31):
       빈 파일을 S3에 올리고 다음 셀로 넘어가면 SageMaker 학습을 제출한 뒤 몇 분 기다려서야
       실패한다. 데이터가 없다는 건 노트북에서 바로 알 수 있으므로 그 자리에서 세우는 게 낫다.
    """
    if len(rows) < min_rows:
        raise ValueError(
            f"GRPO 학습 데이터가 {len(rows)}건입니다(최소 {min_rows}건 필요) — 저장/업로드를 중단합니다.\n"
            "  자주 있는 원인:\n"
            "  1) 🔴 **커널이 옛 모듈을 캐시** — common/*.py 를 고친 뒤 커널을 재시작하지 않으면\n"
            "     노트북이 예전 코드를 계속 씁니다. Kernel → Restart 후 처음부터 다시 실행하세요.\n"
            "  2) Bedrock 합성 수율 저조 — 위 로그의 'batch gen failed' 사유를 확인하세요.\n"
            "  3) 'failures' 소스인데 실패 사례가 없음 — SFT가 다 맞혔다는 뜻(좋은 신호).\n"
            "  → 급하면 GRPO_PROMPT_SOURCE='holdout' 으로 바꿔 즉시 진행할 수 있습니다."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {path} ({len(rows)}건)")
    return path


# ---------------------------------------------------------------------------
# 1) holdout — SFT가 쓰지 않은 구간
# ---------------------------------------------------------------------------
def from_holdout(train_jsonl: str, n: int, *, sft_used: int) -> list[dict]:
    """SFT가 학습에 쓴 앞 sft_used건 **이후** 구간에서 n건.

    🔴 겹치면 누출이다. train.jsonl 은 시드+합성을 이어붙인 파일이고 SFT는 앞에서부터
       config.NUM_SEED_SAMPLES + 합성분을 쓰므로, 그 뒤 구간을 GRPO에 넘긴다.
    ⚠️ 같은 분포이므로 advantage 문제는 남는다(모듈 독스트링 참고). 무료·즉시라는 장점 때문에 기본값.
    """
    rows = _read_messages_jsonl(train_jsonl)
    tail = rows[sft_used:]
    if not tail:
        raise ValueError(
            f"{train_jsonl} 에 SFT 미사용 구간이 없습니다(총 {len(rows)}건, SFT 사용 {sft_used}건).\n"
            f"  → 01_data_and_synthetic 에서 데이터를 더 만들거나, GRPO_PROMPT_SOURCE='synth'/'failures' 를 쓰세요."
        )
    if len(tail) < n:
        print(f"⚠️ 미사용 구간이 {len(tail)}건뿐입니다(요청 {n}건) — 있는 만큼만 씁니다.")
    return tail[:n]


# ---------------------------------------------------------------------------
# 2) synth — Bedrock으로 prompt 생성 (정답은 채점용으로만)
# ---------------------------------------------------------------------------
# 🔴 GRPO 전용 '난이도' 지시 — 생성 프롬프트에만 덧붙인다(critique에는 넣지 않는다).
#    왜 필요한가 (실측 2026-07-31, 추출 트랙):
#      SFT 미사용 시드 구간을 그대로 seed로 쓰면 생성물이 seed를 따라간다. glaive 뒷부분은
#      인자 없는 함수(`{"arguments": {}}`)가 94%라, 합성 8건이 **전부 인자 0개**로 나왔다.
#      인자가 없으면 채점이 사실상 "함수명 맞았나"뿐이어서 SFT가 이미 잘하고,
#      rollout이 전부 만점 → advantage≈0 → 학습 신호가 없다(holdout과 다를 바 없음).
#    왜 critique에는 넣지 않는가:
#      critique가 같은 지시를 받으면 "인자 2개 이상"을 기준으로 채점해 seed(인자 0개)와 다르다며
#      groundedness를 낮춰 **8/8 전부 기각**했다(실측). 생성만 어렵게 하고 채점은 도메인 기준으로 둔다.
_HARDEN = {
    "extraction": (
        " IMPORTANT — these examples are for reinforcement-learning (GRPO), so they must be "
        "CHALLENGING, not typical: every generated call MUST carry at least TWO typed arguments "
        "(never an empty arguments object), and the user text must express those values "
        "INDIRECTLY so the model has to infer them (e.g. 'next Friday' rather than a date, "
        "'the cheaper option' rather than a number). Vary function names and argument shapes."
    ),
    "classification": (
        " IMPORTANT — these examples are for reinforcement-learning (GRPO), so they must be "
        "CHALLENGING: prefer messages that sit near the boundary between two similar intents, "
        "use indirect or emotional phrasing, and cover the rarer intents rather than the "
        "most common ones. Still assign exactly one correct label."
    ),
}


def harden_instruction(task_instruction: str, kind: str) -> str:
    """task_instruction 에 GRPO용 난이도 제약을 덧붙인다(해당 kind가 없으면 원문 그대로)."""
    return task_instruction + _HARDEN.get(kind, "")


def from_synth(*, task_instruction: str, seed_texts: list[str], n: int,
               model_id: str, region: str,
               to_messages: Callable[[dict], list[dict[str, str]]],
               kind: str = "",
               harden: bool = True,
               # 🔴 병렬도 — GRPO 기본값은 SFT 합성보다 높게 잡는다(실측 2026-07-31).
               #    측정: max_workers=4, batch_size=5 로 20건에 **3분 40초** → 100건이면 15~18분.
               #    호출 수 = 생성(n/batch_size) + critique(후보 1개당 1회)이고 **critique가 다수**다.
               #    Bedrock 한도는 분당 250 요청(us-west-2 실조회)이라 12 동시는 안전한 범위.
               #    batch_size를 키우면 1회 호출로 여러 건을 받아 생성 호출 수 자체가 줄어든다.
               max_workers: int = 12,
               batch_size: int = 10,
               progress_cb: Callable[[int, int], None] | None = None) -> list[dict]:
    """Bedrock 합성으로 **새 분포의** prompt를 만든다(SFT 합성과 같은 파이프라인 재사용).

    🔴 RL은 정답이 학습 입력이 아니지만, 이 kit의 reward는 프로그램적 채점이라 reference가 필요하다.
       (JSON 유효성 + 함수명/인자 일치, 라벨 정확 일치 등) 그래서 SFT 합성과 같은 (input, output)
       형태로 만들되, **학습 시에는 output이 reward 계산에만 쓰인다**(train_grpo.py `_to_grpo`).

    🔴 harden=True(기본): kind 별 '난이도 제약'을 **생성 프롬프트에만** 덧붙인다(_HARDEN 참고).
       이게 없으면 seed 분포를 그대로 따라가 advantage≈0이 되어 합성한 의미가 없다(실측).
    ⚠️ SFT 합성과 같은 seed를 주면 분포가 또 겹친다. seed_texts 는 SFT가 쓰지 않은 시드 구간에서
       뽑아 넘기세요(노트북이 그렇게 호출한다).
    """
    from common.synth import bedrock_synth as bs

    gen_instruction = harden_instruction(task_instruction, kind) if harden else None
    if gen_instruction and gen_instruction != task_instruction:
        print(f"난이도 제약 적용(kind={kind}) — 생성 프롬프트에만, critique는 원래 기준 유지")

    print(f"합성 시작: 목표 {n}건 | 동시 호출 {max_workers} | 배치당 {batch_size}건")
    print("  (호출 수 ≈ 생성 " + str(max(1, n // batch_size)) + "회 + critique ~" + str(n) + "회 — critique가 다수입니다)")

    examples = bs.generate_grounded(
        task_instruction=task_instruction,      # critique 기준(도메인 정합성)
        gen_instruction=gen_instruction,        # 생성 기준(어렵게)
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
# 3) failures — SFT 모델이 틀린 건만 (RL 효율이 가장 좋은 경로)
# ---------------------------------------------------------------------------
def _is_failure(pred: Any, gold: str, kind: str) -> bool:
    """예측이 실패인지 판정. 04_evaluate 의 지표와 같은 기준을 쓴다."""
    p = "" if pred is None else str(pred).strip()
    g = (gold or "").strip()
    if not p:
        return True
    if kind == "extraction":
        # JSON 파싱 실패 또는 함수명 불일치 → 실패
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
    # 요약·QA 등 자유서술: 프로그램적 채점이 어려워 GRPO 대상이 아니다(이 kit은 제공하지 않음)
    raise ValueError(
        f"reward_kind={kind!r} 는 프로그램적 채점이 어려워 failures 수집을 지원하지 않습니다. "
        "GRPO는 추출·분류 트랙에만 제공됩니다.")


def from_failures(heldout: list[dict], preds: list[Any], *, kind: str,
                  to_messages: Callable[[dict], list[dict[str, str]]],
                  max_n: int | None = None) -> list[dict]:
    """04_evaluate 의 (heldout, preds)에서 **틀린 건만** 골라 GRPO prompt로 만든다.

    🔴 왜 가장 효과적인가: 이미 맞히는 prompt는 rollout이 전부 만점 → advantage 0 → 학습 없음.
       틀린 prompt는 rollout 간 점수 편차가 커서 GRPO가 실제로 신호를 얻는다.
    반환은 {"messages": [...]} 리스트(SFT와 같은 형태) — train_grpo.py 가 prompt/reference로 분해한다.
    """
    if len(heldout) != len(preds):
        raise ValueError(f"heldout({len(heldout)})과 preds({len(preds)}) 길이가 다릅니다.")
    fails = [ex for ex, pr in zip(heldout, preds) if _is_failure(pr, ex.get("output", ""), kind)]
    print(f"실패 사례: {len(fails)}/{len(heldout)}건 ({len(fails)/max(1,len(heldout)):.0%})")
    if not fails:
        raise ValueError(
            "실패 사례가 없습니다 — SFT 모델이 held-out을 모두 맞혔습니다.\n"
            "  → GRPO로 더 얻을 것이 적다는 뜻입니다(좋은 신호). N_EVAL을 키워 더 어려운 케이스를 찾거나,\n"
            "    GRPO_PROMPT_SOURCE='synth' 로 새 분포를 만들어 시도하세요."
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
    print(f"GRPO 데이터: {n}건 (고유 prompt {uniq}건, source={source})")
    if uniq < n:
        print(f"⚠️ 중복 prompt {n - uniq}건 — 같은 prompt를 여러 번 학습해도 얻는 게 적습니다.")
    if source == "holdout":
        print("ℹ️ holdout은 SFT와 **같은 분포**입니다. 누출은 막았지만 모델이 이미 잘하는 구간이면")
        print("   rollout이 모두 만점이 되어 advantage≈0(학습 신호 없음)이 될 수 있습니다.")
        print("   → 학습 후 reward가 거의 변하지 않으면 'failures' 또는 'synth'로 바꿔 보세요.")
    return info
