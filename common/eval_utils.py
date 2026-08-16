"""코스별 파인튜닝 품질을 평가하는 지표를 제공합니다.

평가에는 합성 데이터나 학습 데이터를 쓰지 않고 별도로 분리한 시드 데이터만 사용합니다.
"""
from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# 공통: 모델 출력에서 JSON 추출 (bedrock_synth와 동일 규약)
# ---------------------------------------------------------------------------
def extract_json_obj(text: str) -> dict | None:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    candidates = [t]
    if "{" in t and "}" in t:
        candidates.append(t[t.find("{"): t.rfind("}") + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# JSON 추출: valid_json_rate, name_accuracy, arg_f1
# ---------------------------------------------------------------------------
def _norm_val(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _arg_pairs(args: Any) -> set:
    if not isinstance(args, dict):
        return set()
    return {(k, _norm_val(v)) for k, v in args.items()}


def eval_extraction(pairs: list[tuple[str, dict]]) -> dict:
    """pairs: [(pred_text, gold_json_dict), ...]. gold는 {"name","arguments"} dict."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    valid = name_ok = exact = 0
    tp = fp = fn = 0
    for pred_text, gold in pairs:
        gold_name = gold.get("name")
        gold_args = _arg_pairs(gold.get("arguments", {}))
        pred = extract_json_obj(pred_text)
        structural_ok = isinstance(pred, dict) and "name" in pred and isinstance(pred.get("arguments", {}), dict)
        if structural_ok:
            valid += 1
            if pred.get("name") == gold_name:
                name_ok += 1
            pred_args = _arg_pairs(pred.get("arguments", {}))
        else:
            pred_args = set()
        inter = pred_args & gold_args
        tp += len(inter)
        fp += len(pred_args - gold_args)
        fn += len(gold_args - pred_args)
        if structural_ok and pred.get("name") == gold_name and pred_args == gold_args:
            exact += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    arg_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "valid_json_rate": valid / n, "name_accuracy": name_ok / n,
        "arg_precision": prec, "arg_recall": rec, "arg_f1": arg_f1, "exact_match": exact / n,
    }


# ---------------------------------------------------------------------------
# 분류: 예측 라벨 정규화 + accuracy/macro-F1
# ---------------------------------------------------------------------------
def normalize_label(pred_text: str, label_set: list[str]) -> str:
    """자유 형식 출력을 정규화해 닫힌 라벨 집합에 매칭합니다."""
    p = pred_text.strip().strip(".").strip().lower().replace(" ", "_")
    lut = {l.lower(): l for l in label_set}
    if p in lut:
        return lut[p]
    # substring
    for l in label_set:
        if l.lower() in p or p in l.lower():
            return l
    # fuzzy (rapidfuzz 있으면)
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(p, label_set, scorer=fuzz.ratio)
        if match:
            return match[0]
    except ImportError:
        pass
    return label_set[0] if label_set else pred_text  # 최후 폴백


def eval_classification(pairs: list[tuple[str, str]], label_set: list[str]) -> dict:
    """pairs: [(pred_text, gold_label), ...]. sklearn 필요."""
    from sklearn.metrics import accuracy_score, f1_score

    preds = [normalize_label(p, label_set) for p, _ in pairs]
    golds = [g for _, g in pairs]
    return {
        "n": len(pairs),
        "accuracy": accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, labels=label_set, average="macro", zero_division=0),
        "weighted_f1": f1_score(golds, preds, labels=label_set, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# 요약/QA: ROUGE-L (rouge_score)
# ---------------------------------------------------------------------------
def eval_rouge(pairs: list[tuple[str, str]]) -> dict:
    """pairs: [(pred_summary, gold_summary), ...]. rouge_score 필요."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, gold in pairs:
        s = scorer.score(gold, pred)
        for k in agg:
            agg[k] += s[k].fmeasure
    n = max(1, len(pairs))
    return {"n": len(pairs), **{k: v / n for k, v in agg.items()}}


# ---------------------------------------------------------------------------
# 요약/QA: Bedrock LLM-judge (groundedness/coverage 또는 correctness/helpfulness)
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = "You are a strict evaluation judge. Output STRICT JSON only, no prose."


def llm_judge(
    *, model_id: str, region: str, source: str, prediction: str, reference: str,
    rubric: str, axes: list[str],
) -> dict:
    """Bedrock Converse로 1-5 채점. axes 각 축을 JSON으로 반환. common.aws_utils 재사용."""
    from .aws_utils import bedrock_converse

    axes_spec = ", ".join(f'"{a}": <1-5 int>' for a in axes)
    prompt = (
        f"{rubric}\n\n"
        f"[SOURCE]\n{source[:4000]}\n\n"
        f"[REFERENCE ANSWER]\n{reference[:2000]}\n\n"
        f"[MODEL OUTPUT]\n{prediction[:2000]}\n\n"
        f'Return JSON only: {{{axes_spec}, "reason": "<one sentence>"}}'
    )
    raw = bedrock_converse(model_id=model_id, region=region, user_text=prompt,
                           system_text=_JUDGE_SYSTEM, max_tokens=256, temperature=0.0)
    obj = extract_json_obj(raw) or {}
    scores = {a: float(obj.get(a, 0)) for a in axes}
    scores["reason"] = obj.get("reason", "")
    return scores


def aggregate_judge(results: list[dict], axes: list[str]) -> dict:
    """LLM 평가 결과의 축별 평균을 계산합니다."""
    n = max(1, len(results))
    out = {"n": len(results)}
    for a in axes:
        out[f"mean_{a}"] = sum(r.get(a, 0) for r in results) / n
    out["mean_overall"] = sum(out[f"mean_{a}"] for a in axes) / max(1, len(axes))
    return out
