"""시드와 합성 데이터의 분포, 다양성, 품질, 누출을 점검합니다.

길이, 중복, 토큰 절단, 클래스 균형, 출력 스키마를 확인하고 필요한 조치를 출력합니다.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _pair_from(obj: Any) -> tuple[str, str]:
    """지원하는 예시 형식을 입력과 출력 문자열 쌍으로 변환합니다."""
    if hasattr(obj, "messages"):
        msgs = obj.messages
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
        return str(user), str(asst)
    if isinstance(obj, dict) and "input" in obj:
        return str(obj.get("input", "")), str(obj.get("output", ""))
    return str(obj), ""


def _stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0, "mean": 0.0, "median": 0.0, "max": 0, "p90": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n, "min": s[0], "max": s[-1],
        "mean": round(sum(s) / n, 1),
        "median": s[n // 2],
        "p90": s[min(n - 1, int(n * 0.9))],
    }


def _dedup_rate(texts: list[str]) -> float:
    """정규화 후 중복 비율(0~1). 낮을수록 다양성 좋음."""
    if not texts:
        return 0.0
    norm = [re.sub(r"\s+", " ", t.lower()).strip() for t in texts]
    uniq = len(set(norm))
    return round(1 - uniq / len(norm), 3)


def compare(seed_examples: list, synth_examples: list, *, plot: bool = True) -> dict:
    """seed vs 합성 핵심 통계 비교 + (선택) 길이 분포 히스토그램. 통계 dict 반환."""
    seed_pairs = [_pair_from(e) for e in seed_examples]
    synth_pairs = [_pair_from(e) for e in synth_examples]

    def _report(pairs):
        in_lens = [len(i) for i, _ in pairs]
        out_lens = [len(o) for _, o in pairs]
        return {
            "count": len(pairs),
            "input_chars": _stats(in_lens),
            "output_chars": _stats(out_lens),
            "input_dedup_rate": _dedup_rate([i for i, _ in pairs]),
            "output_dedup_rate": _dedup_rate([o for _, o in pairs]),
        }

    result = {"seed": _report(seed_pairs), "synth": _report(synth_pairs)}

    # 표 출력
    print("=" * 60)
    print(f"{'metric':<22}{'seed':>18}{'synth':>18}")
    print("-" * 60)

    def _row(label, sv, tv):
        print(f"{label:<22}{str(sv):>18}{str(tv):>18}")

    _row("count", result["seed"]["count"], result["synth"]["count"])
    for fld in ("input_chars", "output_chars"):
        for k in ("mean", "median", "p90", "max"):
            _row(f"{fld}.{k}", result["seed"][fld][k], result["synth"][fld][k])
    _row("input_dedup_rate", result["seed"]["input_dedup_rate"], result["synth"]["input_dedup_rate"])
    _row("output_dedup_rate", result["seed"]["output_dedup_rate"], result["synth"]["output_dedup_rate"])
    print("=" * 60)

    # 원래 출력이 반복되는 데이터셋을 고려해 시드 대비 증가 폭으로 판단합니다.
    s_dup = result["seed"]["output_dedup_rate"]
    t_dup = result["synth"]["output_dedup_rate"]
    if t_dup > 0.3 and t_dup > s_dup + 0.15:
        print(f"합성 출력 중복률({t_dup})이 시드({s_dup})보다 높습니다. "
              "시드 다양성과 생성 temperature를 확인하세요.")
    elif t_dup > 0.3:
        print(f"출력 중복률이 높지만 합성({t_dup})과 시드({s_dup})가 비슷합니다. "
              "데이터셋 특성일 수 있습니다.")
    sm, tm = result["seed"]["output_chars"]["mean"], result["synth"]["output_chars"]["mean"]
    if tm and sm and (tm > sm * 2 or tm < sm * 0.5):
        print(f"합성 출력 평균 길이({tm})가 시드({sm})와 크게 다릅니다. 분포를 확인하세요.")

    if plot:
        _plot_len_hist(seed_pairs, synth_pairs)
    return result


def _plot_len_hist(seed_pairs, synth_pairs) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("(matplotlib가 없어 차트를 생략합니다. `pip install matplotlib`로 설치하세요.)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, idx, title in [(axes[0], 0, "input length (chars)"), (axes[1], 1, "output length (chars)")]:
        sv = [len(p[idx]) for p in seed_pairs]
        tv = [len(p[idx]) for p in synth_pairs]
        bins = 20
        if sv:
            ax.hist(sv, bins=bins, alpha=0.5, label="seed", density=True)
        if tv:
            ax.hist(tv, bins=bins, alpha=0.5, label="synth", density=True)
        ax.set_title(title)
        ax.set_xlabel("chars"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 1) 토큰 길이
# ---------------------------------------------------------------------------
def token_length_report(seed_examples: list, synth_examples: list, tokenizer,
                        *, max_seq_length: int | None = None, plot: bool = True) -> dict:
    """전체 학습 프롬프트의 토큰 길이와 절단 위험을 계산합니다."""
    def _lens(pairs):
        out = []
        for i, o in pairs:
            # 학습 시 실제 입력 = user + assistant 전체(chat template 마커 포함). 마커 오버헤드까지
            # 반영하려면 apply_chat_template이 정확하지만, 실패 시 단순 합산으로 폴백한다.
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": i}, {"role": "assistant", "content": o}],
                    tokenize=False)
                out.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            except Exception:  # noqa: BLE001
                out.append(len(tokenizer(i + o, add_special_tokens=False)["input_ids"]))
        return out

    sp = [_pair_from(e) for e in seed_examples]
    tp = [_pair_from(e) for e in synth_examples]
    s_tok, t_tok = _lens(sp), _lens(tp)
    all_tok = s_tok + t_tok

    def _p(vals, q):
        if not vals:
            return 0
        v = sorted(vals)
        return v[min(len(v) - 1, int(len(v) * q))]

    res = {"seed_tokens": _stats(s_tok), "synth_tokens": _stats(t_tok),
           "p99_all": _p(all_tok, 0.99), "max_all": max(all_tok) if all_tok else 0}

    print("=" * 68)
    print(f"{'토큰 길이(전체 프롬프트)':<26}{'seed':>18}{'synth':>18}")
    print("-" * 68)
    for k in ("mean", "median", "p90", "max"):
        print(f"{k:<26}{str(res['seed_tokens'][k]):>18}{str(res['synth_tokens'][k]):>18}")
    print(f"{'p99 (seed+synth)':<26}{'':>18}{res['p99_all']:>18}")

    if max_seq_length:
        over_s = sum(1 for v in s_tok if v > max_seq_length)
        over_t = sum(1 for v in t_tok if v > max_seq_length)
        rate = round((over_s + over_t) / max(1, len(all_tok)), 3)
        res.update({"max_seq_length": max_seq_length, "truncated": over_s + over_t,
                    "truncated_rate": rate})
        print("-" * 68)
        print(f"max_seq_length={max_seq_length}, 절단 예시 {over_s + over_t}건 ({rate:.1%})")
        if rate > 0.02:
            # p99를 덮는 가장 작은 2의 거듭제곱(512 이상)을 권장
            need = res["p99_all"]
            rec = 512
            while rec < need:
                rec *= 2
            print(f"절단률이 {rate:.1%}입니다. max_seq_length를 {rec}로 올리거나 "
                  "긴 예시를 제외하거나 요약하세요.")
        else:
            print("절단 위험이 낮습니다.")
    print("=" * 68)

    if plot and all_tok:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3.2))
            if s_tok:
                ax.hist(s_tok, bins=25, alpha=0.5, label="seed", density=True)
            if t_tok:
                ax.hist(t_tok, bins=25, alpha=0.5, label="synth", density=True)
            if max_seq_length:
                ax.axvline(max_seq_length, color="red", ls="--",
                           label=f"max_seq_length={max_seq_length}")
            ax.set_title("token length (full prompt)")
            ax.set_xlabel("tokens"); ax.set_ylabel("density"); ax.legend()
            fig.tight_layout(); plt.show()
        except Exception:  # noqa: BLE001
            pass
    return res


# ---------------------------------------------------------------------------
# 2) 근사 중복과 시드 복제
# ---------------------------------------------------------------------------
def near_duplicate_report(seed_examples: list, synth_examples: list,
                          *, threshold: int = 90, sample: int = 400) -> dict:
    """합성 데이터 내부의 근사 중복과 시드 복제 비율을 측정합니다."""
    sp = [_pair_from(e) for e in seed_examples]
    tp = [_pair_from(e) for e in synth_examples]
    s_in = [i for i, _ in sp][:sample]
    t_in = [i for i, _ in tp][:sample]
    if not t_in:
        print("(합성 데이터가 없어 근사중복 점검을 건너뜁니다)")
        return {}
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        print("(rapidfuzz가 없어 근사 중복 점검을 생략합니다. `uv pip install rapidfuzz`로 설치하세요.)")
        return {}

    import numpy as np

    # (a) 합성 내부 근사중복: 상삼각만 보고 '자기 자신 제외 최고 유사도'가 threshold 이상인 비율
    m = np.asarray(process.cdist(t_in, t_in, scorer=fuzz.token_set_ratio, workers=-1), dtype=float)
    np.fill_diagonal(m, 0.0)
    self_max = m.max(axis=1) if len(t_in) > 1 else np.zeros(1)
    near_dup = int((self_max >= threshold).sum())
    near_rate = round(near_dup / len(t_in), 3)

    # (b) seed 표절: 각 합성이 어떤 seed와도 threshold 이상 닮았는지
    leak_rate, leak = 0.0, 0
    if s_in:
        ms = np.asarray(process.cdist(t_in, s_in, scorer=fuzz.token_set_ratio, workers=-1), dtype=float)
        seed_max = ms.max(axis=1)
        leak = int((seed_max >= threshold).sum())
        leak_rate = round(leak / len(t_in), 3)

    res = {"threshold": threshold, "sampled": len(t_in),
           "near_dup": near_dup, "near_dup_rate": near_rate,
           "seed_copy": leak, "seed_copy_rate": leak_rate,
           "mean_self_similarity": round(float(self_max.mean()), 1) if len(t_in) > 1 else 0.0}

    print("=" * 68)
    print(f"근사중복 점검 (유사도>={threshold}, 샘플 {len(t_in)}건)")
    print("-" * 68)
    print(f"  합성 내부 근사중복 : {near_dup}건 ({near_rate:.1%})   평균 최고유사도 {res['mean_self_similarity']}")
    print(f"  seed 표절(거의 동일): {leak}건 ({leak_rate:.1%})")
    if near_rate > 0.15:
        print(f"합성 데이터의 근사 중복률이 높습니다({near_rate:.1%} > 15%). "
              "temperature, 시드 샘플링, 프롬프트의 다양성 조건을 조정하세요.")
    if leak_rate > 0.10:
        print(f"시드와 거의 같은 합성 데이터 비율이 높습니다({leak_rate:.1%} > 10%). "
              "평가 데이터는 합성에 사용하지 않은 구간에서 분리하세요.")
    if near_rate <= 0.15 and leak_rate <= 0.10:
        print("근사 중복과 시드 복제 비율이 허용 범위입니다.")
    print("=" * 68)
    return res


# ---------------------------------------------------------------------------
# 3) 어휘 다양성
# ---------------------------------------------------------------------------
def lexical_diversity(seed_examples: list, synth_examples: list, *, top: int = 5) -> dict:
    """고유 n-gram 비율과 문장 시작 표현의 편중을 비교합니다."""
    def _ngrams(texts, n):
        tot, uniq = 0, set()
        for t in texts:
            w = re.findall(r"\w+", t.lower())
            g = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))]
            tot += len(g); uniq.update(g)
        return round(len(uniq) / tot, 3) if tot else 0.0

    def _head(texts, n=3):
        c = Counter()
        for t in texts:
            w = re.findall(r"\w+", t.lower())[:n]
            if w:
                c[" ".join(w)] += 1
        return c

    s_in = [i for i, _ in (_pair_from(e) for e in seed_examples)]
    t_in = [i for i, _ in (_pair_from(e) for e in synth_examples)]
    res = {
        "seed": {"distinct_1": _ngrams(s_in, 1), "distinct_2": _ngrams(s_in, 2)},
        "synth": {"distinct_1": _ngrams(t_in, 1), "distinct_2": _ngrams(t_in, 2)},
    }
    t_head = _head(t_in)
    top_share = round(t_head.most_common(1)[0][1] / len(t_in), 3) if t_in and t_head else 0.0
    res["synth"]["top_head_share"] = top_share

    print("=" * 68)
    print(f"{'어휘 다양성(input)':<26}{'seed':>18}{'synth':>18}")
    print("-" * 68)
    for k in ("distinct_1", "distinct_2"):
        print(f"{k:<26}{res['seed'][k]:>18}{res['synth'][k]:>18}")
    print("-" * 68)
    print(f"합성 시작 3-gram top{top}:")
    for g, c in t_head.most_common(top):
        print(f"    {c:>5}건 ({c / max(1, len(t_in)):>5.1%})  '{g}'")
    sd2, td2 = res["seed"]["distinct_2"], res["synth"]["distinct_2"]
    if sd2 and td2 and td2 < sd2 * 0.7:
        print(f"합성 distinct_2({td2})가 시드({sd2})의 70% 미만입니다. "
              "temperature를 올리거나 프롬프트에 어휘와 문체 변화를 요구하세요.")
    if top_share > 0.3:
        print(f"합성 데이터의 {top_share:.0%}가 같은 3단어로 시작합니다. "
              "도입 문구를 다양화하세요.")
    if not (sd2 and td2 and td2 < sd2 * 0.7) and top_share <= 0.3:
        print("어휘 다양성이 양호합니다.")
    print("=" * 68)
    return res


# ---------------------------------------------------------------------------
# 4) 라벨과 클래스 균형
# ---------------------------------------------------------------------------
def label_balance(seed_examples: list, synth_examples: list, *, plot: bool = True) -> dict:
    """분류 데이터의 클래스 분포와 누락된 라벨을 확인합니다."""
    def _labels(pairs):
        c = Counter()
        for _, o in pairs:
            lab = re.sub(r"\s+", " ", str(o).strip().lower())
            if lab and len(lab) <= 40:      # 라벨로 볼 수 있을 만큼 짧은 것만
                c[lab] += 1
        return c

    sp = [_pair_from(e) for e in seed_examples]
    tp = [_pair_from(e) for e in synth_examples]

    # JSON이나 자유 서술 출력을 클래스 라벨로 오인하지 않도록 먼저 형식을 확인합니다.
    def _looks_categorical(pairs) -> bool:
        outs = [str(o).strip() for _, o in pairs if str(o).strip()]
        if not outs:
            return False
        if sum(1 for o in outs if o[:1] in "{[") > len(outs) * 0.2:
            return False                      # JSON 출력 트랙(추출)
        short = [o for o in outs if len(o) <= 40]
        if len(short) < len(outs) * 0.8:
            return False                      # 긴 자유 서술
        return len(set(o.lower() for o in short)) <= max(2, len(short) * 0.5)

    if not _looks_categorical(tp):
        print("(출력이 라벨 형식이 아니므로 클래스 균형 점검을 건너뜁니다.)")
        return {}

    s_c, t_c = _labels(sp), _labels(tp)
    if not t_c:
        print("(라벨을 추출하지 못해 클래스 균형 점검을 건너뜁니다)")
        return {}

    labels = sorted(set(s_c) | set(t_c), key=lambda k: -(s_c[k] + t_c[k]))
    print("=" * 68)
    print(f"{'label':<28}{'seed':>10}{'synth':>10}{'synth %':>12}")
    print("-" * 68)
    t_tot = sum(t_c.values())
    for lab in labels[:15]:
        print(f"{lab[:27]:<28}{s_c[lab]:>10}{t_c[lab]:>10}{t_c[lab] / max(1, t_tot):>11.1%}")
    missing = [l for l in s_c if t_c[l] == 0]
    shares = [t_c[l] / max(1, t_tot) for l in labels if t_c[l]]
    imbalance = round(max(shares) / min(shares), 1) if len(shares) > 1 else 1.0
    print("-" * 68)
    print(f"클래스 수: seed={len(s_c)}, synth={len(t_c)} | 최다/최소 비율 {imbalance}x")
    if missing:
        print(f"합성 데이터에 없는 시드 클래스 {len(missing)}개: {missing[:5]}. "
              "해당 클래스를 지정해 추가 생성하세요.")
    if imbalance > 5:
        print(f"클래스 불균형이 {imbalance}배입니다. 소수 클래스를 추가 생성하거나 "
              "클래스 가중치와 오버샘플링을 검토하세요.")
    if not missing and imbalance <= 5:
        print("클래스 균형이 양호합니다.")
    print("=" * 68)

    if plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            show = labels[:12]
            x = np.arange(len(show)); w = 0.4
            fig, ax = plt.subplots(figsize=(max(7, len(show) * 0.9), 3.2))
            ax.bar(x - w / 2, [s_c[l] for l in show], w, label="seed")
            ax.bar(x + w / 2, [t_c[l] for l in show], w, label="synth")
            ax.set_xticks(x); ax.set_xticklabels([l[:14] for l in show], rotation=30, ha="right")
            ax.set_ylabel("count"); ax.set_title("label distribution"); ax.legend()
            fig.tight_layout(); plt.show()
        except Exception:  # noqa: BLE001
            pass
    return {"seed": dict(s_c), "synth": dict(t_c), "missing_in_synth": missing,
            "imbalance_ratio": imbalance}


# ---------------------------------------------------------------------------
# 5) 출력 유효성
# ---------------------------------------------------------------------------
def output_validity(synth_examples: list, *, expect_json: bool = True,
                    required_keys: tuple[str, ...] = ("name", "arguments")) -> dict:
    """합성 출력이 목표 스키마를 지키는지 검사합니다."""
    tp = [_pair_from(e) for e in synth_examples]
    outs = [o for _, o in tp]
    if not outs:
        return {}
    if not expect_json:
        empty = sum(1 for o in outs if not o.strip())
        print(f"출력 점검: 총 {len(outs)}건, 빈 출력 {empty}건")
        return {"n": len(outs), "empty": empty}

    import json
    ok, bad, miss = 0, [], 0
    for o in outs:
        try:
            obj = json.loads(o)
        except (json.JSONDecodeError, TypeError):
            bad.append(o[:60]); continue
        ok += 1
        if isinstance(obj, dict) and not all(k in obj for k in required_keys):
            miss += 1
    rate = round(ok / len(outs), 3)
    print("=" * 68)
    print(f"출력 유효성: JSON 파싱 {ok}/{len(outs)} ({rate:.1%}) | 필수키 누락 {miss}건")
    if bad:
        print(f"  파싱 실패 예: {bad[:2]}")
    if rate < 0.98:
        print(f"JSON 파싱률이 {rate:.1%}입니다. 실패 예시를 제외하거나 "
              "생성 프롬프트에 스키마를 다시 명시하세요.")
    elif miss:
        print(f"필수 키({required_keys})가 누락된 예시 {miss}건을 제외하세요.")
    else:
        print("출력 스키마를 준수합니다.")
    print("=" * 68)
    return {"n": len(outs), "json_ok": ok, "json_rate": rate, "missing_keys": miss}


# ---------------------------------------------------------------------------
# 6) 한 번에 실행 (노트북 기본 경로)
# ---------------------------------------------------------------------------
def quick_report(seed_examples: list, synth_examples: list, *, tokenizer=None,
                 max_seq_length: int | None = None, expect_json: bool = False,
                 plot: bool = True) -> dict:
    """compare + 근사중복 + 어휘다양성 (+토크나이저/JSON 제공 시 해당 항목)을 순서대로 실행.

    tokenizer와 max_seq_length를 주면 토큰 길이와 절단 위험도 확인합니다.
    """
    out = {"basic": compare(seed_examples, synth_examples, plot=plot)}
    if tokenizer is not None:
        out["tokens"] = token_length_report(seed_examples, synth_examples, tokenizer,
                                           max_seq_length=max_seq_length, plot=plot)
    out["near_dup"] = near_duplicate_report(seed_examples, synth_examples)
    out["diversity"] = lexical_diversity(seed_examples, synth_examples)
    lb = label_balance(seed_examples, synth_examples, plot=plot)
    if lb:
        out["labels"] = lb
    if expect_json:
        out["validity"] = output_validity(synth_examples)
    return out


# ---------------------------------------------------------------------------
# JSON 추출 코스 전용 함수명과 인자 키 커버리지
# ---------------------------------------------------------------------------
def json_field_coverage(seed_examples: list, synth_examples: list) -> dict:
    """함수 호출 JSON의 함수명과 인자 키 분포를 비교합니다."""
    import json

    def _names_keys(pairs):
        names, keys = Counter(), Counter()
        for _, out in pairs:
            try:
                obj = json.loads(out)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                if "name" in obj:
                    names[obj["name"]] += 1
                args = obj.get("arguments")
                if isinstance(args, dict):
                    keys.update(args.keys())
        return names, keys

    sp = [_pair_from(e) for e in seed_examples]
    tp = [_pair_from(e) for e in synth_examples]
    s_names, s_keys = _names_keys(sp)
    t_names, t_keys = _names_keys(tp)

    print("=" * 60)
    print(f"함수명 종류: seed={len(s_names)}, synth={len(t_names)} "
          f"(합성 신규 함수명 {len(set(t_names) - set(s_names))}개)")
    print(f"인자 키 종류: seed={len(s_keys)}, synth={len(t_keys)}")
    print("합성 top 함수명:", dict(t_names.most_common(5)))
    print("=" * 60)
    return {"seed_names": dict(s_names), "synth_names": dict(t_names),
            "seed_keys": dict(s_keys), "synth_keys": dict(t_keys)}
