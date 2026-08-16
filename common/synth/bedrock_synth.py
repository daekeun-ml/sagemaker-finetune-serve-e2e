"""시드 데이터를 근거로 합성 학습 데이터를 생성합니다.

Bedrock Converse로 후보를 만들고 근거성과 관련성을 평가한 뒤 PII와 중복을 제거합니다.
모델 ID는 호출부에서 전달합니다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..aws_utils import bedrock_converse

# 라이브러리 로거: 핸들러 설정 안 함(앱이 setup_logging으로 구성). kit 네임스페이스 하위.
logger = logging.getLogger("gemma_e2e.synth")


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class SynthExample:
    messages: list[dict[str, str]]   # 표준 messages (gemma_format.build_messages 결과)
    groundedness: float              # 0~1, critique 점수
    relevance: float                 # 0~1
    source_seed_idx: int             # 어느 seed에 grounded 됐는지 (추적)


# ---------------------------------------------------------------------------
# 1) 생성 프롬프트 (task별 커스터마이즈 가능)
# ---------------------------------------------------------------------------
GEN_SYSTEM = (
    "You are a senior data-labeling expert creating high-quality, DIVERSE synthetic "
    "fine-tuning examples. You MUST ground every example in the provided seed sample: "
    "reuse its domain, style, entities, and label space. Do not invent facts outside "
    "the seed's domain. Output STRICTLY valid JSON only, no prose."
)


def build_generation_prompt(task_instruction: str, seed_rows: list[str], n: int) -> str:
    """seed chunk를 근거로 n개 예시를 생성하도록 요청하는 프롬프트."""
    seeds = "\n".join(f"[SEED {i}] {s}" for i, s in enumerate(seed_rows))
    return (
        f"{task_instruction}\n\n"
        f"Below are {len(seed_rows)} real seed samples from the target dataset. "
        f"Generate {n} NEW, diverse examples grounded in the SAME domain, style and "
        f"label space as these seeds (do not copy them verbatim):\n\n{seeds}\n\n"
        f'Return a JSON array of exactly {n} objects, each: '
        f'{{"input": "<user-facing input text>", "output": "<the target completion / label / JSON>"}}. '
        f"JSON array only."
    )


# ---------------------------------------------------------------------------
# 2) critique 프롬프트
# ---------------------------------------------------------------------------
CRITIQUE_SYSTEM = (
    "You are a strict data-quality judge. Score a synthetic example against the seed "
    "domain. Output STRICT JSON only."
)


def build_critique_prompt(task_instruction: str, seed_rows: list[str], example: dict) -> str:
    seeds = "\n".join(f"[SEED {i}] {s}" for i, s in enumerate(seed_rows[:5]))
    return (
        f"Task: {task_instruction}\n\n"
        f"Reference seeds (domain/style/label space):\n{seeds}\n\n"
        f"Candidate example:\n{json.dumps(example, ensure_ascii=False)}\n\n"
        "Rate 0.0-1.0 on two axes and return JSON only: "
        '{"groundedness": <0-1, how well it matches the seed domain/label space>, '
        '"relevance": <0-1, how well output solves the task for the input>, '
        '"reason": "<one short sentence>"}'
    )


# ---------------------------------------------------------------------------
# 3) 파서 / 필터
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Any:
    """Converse 응답에서 JSON(배열/객체) 추출. 방어적."""
    text = text.strip()
    # 코드펜스 제거
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 첫 [ ... ] 또는 { ... } 블록 시도
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = text.find(opener), text.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"JSON parse failed: {text[:200]}...")


# 순수 숫자열을 PII로 오인하지 않도록 전화번호와 카드번호에 구분자를 요구합니다.
_PII_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),                                  # email
    re.compile(r"(?<!\d)\+?\d{1,3}[\s.\-()]\d[\d\s.\-()]{6,}\d(?!\d)"),           # phone (구분자 필수)
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                         # US SSN
    re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{3,4}\b"),                        # card (그룹 구분자 필수)
]


def has_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _as_text(v: Any) -> str:
    """모델이 output을 dict/list(JSON)로 줄 수도 있으므로 안전하게 문자열화."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(v)


def _dedup_key(example: dict) -> str:
    norm = re.sub(r"\s+", " ", (_as_text(example.get("input")) + " " + _as_text(example.get("output"))).lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 4) 메인 생성 루프
# ---------------------------------------------------------------------------
def _gen_one_batch(task_instruction, seed_texts, batch_size, seeds_per_batch,
                   batch_no, model_id, region):
    """단일 배치 생성(1 Bedrock 호출) → 후보 dict 리스트 + 이 배치의 seed_chunk 반환."""
    start = ((batch_no - 1) * seeds_per_batch) % max(1, len(seed_texts))
    seed_chunk = (seed_texts + seed_texts)[start : start + seeds_per_batch]
    prompt = build_generation_prompt(task_instruction, seed_chunk, batch_size)
    # 추론 토큰과 JSON 출력을 수용하도록 배치 크기에 비례해 출력 한도를 늘립니다.
    # Claude 5 이상에서는 호환성을 위해 temperature를 전달하지 않습니다.
    _max_out = max(4096, 900 * batch_size)
    raw = bedrock_converse(model_id=model_id, region=region, user_text=prompt,
                           system_text=GEN_SYSTEM, max_tokens=_max_out)
    cands = _extract_json(raw)
    if isinstance(cands, dict):
        cands = [cands]
    return [c for c in cands if isinstance(c, dict) and "input" in c and "output" in c], seed_chunk


def _critique_one(task_instruction, seed_chunk, ex, model_id, region):
    """단일 후보의 근거성과 관련성을 평가합니다."""
    raw = bedrock_converse(model_id=model_id, region=region,
                           user_text=build_critique_prompt(task_instruction, seed_chunk, ex),
                           system_text=CRITIQUE_SYSTEM, max_tokens=2048)
    crit = _extract_json(raw)
    return float(crit.get("groundedness", 0)), float(crit.get("relevance", 0))


def generate_grounded(
    *,
    task_instruction: str,
    seed_texts: list[str],
    n_total: int,
    # 생성 조건과 평가 기준을 분리할 때 사용합니다.
    gen_instruction: str | None = None,
    model_id: str,
    region: str,
    to_messages: Callable[[dict], list[dict[str, str]]],
    min_groundedness: float = 0.6,
    min_relevance: float = 0.6,
    batch_size: int = 5,
    seeds_per_batch: int = 4,
    max_batches: int | None = None,
    max_workers: int = 4,
    progress_cb: Callable[[int, int], None] | None = None,
    accepted_ref: list | None = None,
    verbose: bool = True,
) -> list[SynthExample]:
    """seed에 grounded된 합성 예시를 n_total개 생성 (critique 통과분만). **병렬 + 진행 표시**.

    Bedrock 호출은 I/O 바운드이므로, 생성 배치와 critique를 ThreadPoolExecutor로 동시에 처리해
    순차 대비 크게 빨라진다.
    - max_workers: 동시 Bedrock 호출 수(기본 4). throttling이 나면 낮춘다.
    - progress_cb(done, total): 예시가 채택될 때마다 호출(노트북 진행바/실시간 미리보기에 연결).
    - accepted_ref: 빈 list를 넘기면 [accepted] 를 담아 progress_cb 시점에 채택 예시에 접근 가능하게 한다.
    - to_messages: {"input","output"} dict → 표준 messages 리스트로 바꾸는 트랙별 어댑터.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # tqdm 있으면 진행바, 없으면 간단한 inline print 폴백
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=n_total, desc="synth", unit="ex")
    except Exception:  # noqa: BLE001
        bar = None

    accepted: list[SynthExample] = []
    seen: set[str] = set()
    if accepted_ref is not None:
        accepted_ref.append(accepted)   # 호출부가 progress_cb 시점에 채택 예시 리스트에 접근 가능

    def _note(done: int):
        if bar is not None:
            bar.n = done; bar.refresh()
        elif verbose:
            print(f"\r  synthetic: {done}/{n_total} accepted", end="", flush=True)
        if progress_cb:
            progress_cb(done, n_total)

    round_no = 0
    stale_rounds = 0        # 연속 '진전 0' 라운드 수 (수율 나쁠 때 무한루프 방지)
    MAX_STALE = 3

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(accepted) < n_total:
            if max_batches is not None and round_no >= max_batches:
                if verbose:
                    logger.warning("max_batches(%d)에 도달해 %d/%d에서 중단합니다",
                                   max_batches, len(accepted), n_total)
                break
            if stale_rounds >= MAX_STALE:
                if verbose:
                    logger.warning("%d회 연속 새 예시가 없어 %d/%d에서 중단합니다 "
                                   "(수율 저조: seed 다양성/기준 완화 고려)", MAX_STALE, len(accepted), n_total)
                break
            before = len(accepted)
            # 이번 라운드에 max_workers개 배치를 병렬로 띄운다(목표 도달까지 라운드 반복).
            gen_futs = {pool.submit(_gen_one_batch, gen_instruction or task_instruction, seed_texts, batch_size,
                                    seeds_per_batch, round_no + i + 1, model_id, region): round_no + i + 1
                        for i in range(max_workers)}
            round_no += max_workers

            # 생성 결과 수집 → critique 후보 큐 구성(중복/PII 선필터)
            crit_futs = {}
            for gf in as_completed(gen_futs):
                try:
                    cands, seed_chunk = gf.result()
                except (ValueError, KeyError) as e:
                    # 원인 확인을 위해 예외 유형과 메시지를 함께 남깁니다.
                    if verbose:
                        logger.info("batch gen failed, skipped (%s: %s)", type(e).__name__, e)
                    continue
                for ex in cands:
                    # output이 dict/list(JSON)로 오는 경우 문자열로 정규화 (to_messages/PII/저장이 str 기대)
                    ex["input"] = _as_text(ex.get("input"))
                    ex["output"] = _as_text(ex.get("output"))
                    if has_pii(ex["input"]) or has_pii(ex["output"]):
                        continue
                    key = _dedup_key(ex)
                    if key in seen:
                        continue
                    seen.add(key)  # 선점(중복 critique 방지)
                    crit_futs[pool.submit(_critique_one, task_instruction, seed_chunk, ex,
                                          model_id, region)] = ex

            # critique 병렬 수집
            for cf in as_completed(crit_futs):
                if len(accepted) >= n_total:
                    break
                ex = crit_futs[cf]
                try:
                    g, r = cf.result()
                except (ValueError, KeyError, TypeError):
                    continue
                if g >= min_groundedness and r >= min_relevance:
                    accepted.append(SynthExample(messages=to_messages(ex),
                                                 groundedness=g, relevance=r, source_seed_idx=0))
                    _note(len(accepted))

            stale_rounds = stale_rounds + 1 if len(accepted) == before else 0

    if bar is not None:
        bar.close()
    elif verbose:
        print()  # inline 진행 줄 마감
    return accepted[:n_total]


def save_jsonl(examples: list[SynthExample], path: str) -> None:
    """학습 포맷(JSONL, 'messages' 컬럼)으로 저장 → SFTTrainer conversational 입력."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex.messages}, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples -> %s", len(examples), path)
