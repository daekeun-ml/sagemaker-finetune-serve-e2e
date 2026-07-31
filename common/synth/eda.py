"""
common/synth/eda.py — seed vs 합성 데이터 EDA (분포·다양성·품질·누출 점검)

합성 데이터가 seed 분포를 잘 따라갔는지, 학습에 넣기 전에 정량 점검한다.
- 무거운 의존성 없이 동작(통계는 순수 python). 차트는 matplotlib, 근사중복은 rapidfuzz가 있으면 사용.
- 노트북 01_data_and_synthetic 에서 seed 예시 리스트와 합성 결과(SynthExample)를 넘겨 호출.

🔴 설계 원칙: "보고 나면 무엇을 바꿀지가 명확한 지표"만 넣는다. 숫자만 늘리는 지표는 판단을 흐린다.
   각 함수는 문제를 발견하면 ⚠️ 와 함께 **구체적 조치**를 출력한다.

제공 함수:
  compare()              길이/중복 기본 통계 + 히스토그램         → 분포 이탈, 완전중복
  token_length_report()  실제 토크나이저 기준 토큰 길이 + 절단 위험 → max_seq_length 결정
  near_duplicate_report() 근사중복(rapidfuzz) + seed 표절          → temperature/seed 다양성, 누출
  lexical_diversity()    distinct-n / 어휘 다양성 + 시작 n-gram 편중 → 프롬프트 템플릿 고착
  label_balance()        라벨/클래스 분포 비교                     → 클래스 불균형
  output_validity()      JSON 파싱률·스키마 준수(추출 트랙)        → 생성 품질
  quick_report()         위 항목을 한 번에 실행(노트북 기본 경로)

입력 형식:
- seed_examples: [{"input": str, "output": str}, ...]  (track_data.load_seed_examples 결과)
- synth_examples: [SynthExample, ...] 또는 [{"input","output"}] — messages에서 자동 추출 지원.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _pair_from(obj: Any) -> tuple[str, str]:
    """SynthExample(messages) 또는 {"input","output"} → (input_text, output_text)."""
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

    # 간단 경고(휴리스틱)
    # 🔴 절대 임계값만 쓰면 오탐이 난다: 인자 없는 함수 호출({"name":...,"arguments":{}})처럼
    #    output이 원래 반복되는 데이터셋은 seed 자체의 중복률이 이미 높다(실측 0.53).
    #    그래서 'seed 대비'로 판단한다 — seed보다 뚜렷히 나쁠 때만 경고.
    s_dup = result["seed"]["output_dedup_rate"]
    t_dup = result["synth"]["output_dedup_rate"]
    if t_dup > 0.3 and t_dup > s_dup + 0.15:
        print(f"⚠️ 합성 output 중복률({t_dup})이 seed({s_dup})보다 뚜렷히 높습니다 "
              "— seed 다양성/생성 temperature 점검 권장.")
    elif t_dup > 0.3:
        print(f"ℹ️ output 중복률이 높지만(합성 {t_dup}) seed({s_dup})도 비슷합니다 "
              "— 데이터셋 특성(인자 없는 함수 등)일 수 있어 경고로 보지 않습니다.")
    sm, tm = result["seed"]["output_chars"]["mean"], result["synth"]["output_chars"]["mean"]
    if tm and sm and (tm > sm * 2 or tm < sm * 0.5):
        print(f"⚠️ 합성 output 평균 길이({tm})가 seed({sm})와 크게 다릅니다 — 분포 이탈 가능.")

    if plot:
        _plot_len_hist(seed_pairs, synth_pairs)
    return result


def _plot_len_hist(seed_pairs, synth_pairs) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("(matplotlib 미설치 — 차트 생략. pip install matplotlib 로 활성화)")
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
# 1) 토큰 길이 — max_seq_length 결정에 직결 (문자 길이로는 알 수 없다)
# ---------------------------------------------------------------------------
def token_length_report(seed_examples: list, synth_examples: list, tokenizer,
                        *, max_seq_length: int | None = None, plot: bool = True) -> dict:
    """실제 토크나이저로 '학습에 들어갈 전체 프롬프트' 토큰 길이를 재고 절단 위험을 계산한다.

    🔴 왜 문자 길이로 부족한가: 학습이 자르는 단위는 토큰이고, 한국어·JSON·코드는 문자당 토큰 수가
       크게 다르다(한글은 문자당 ~1토큰, 영어는 ~0.25토큰). 문자 p90이 안전해 보여도 토큰으로는
       max_seq_length를 넘어 **정답 뒷부분이 잘린 채 학습**될 수 있다 — 이러면 모델이 잘린 출력을
       정답으로 배운다(끝나지 않는 JSON 등).

    조치로 이어지는 출력: 절단 비율과 함께 권장 max_seq_length(p99 기준, 2의 거듭제곱 근사)를 제시.
    """
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
        print(f"max_seq_length={max_seq_length} → 절단 예시 {over_s + over_t}건 ({rate:.1%})")
        if rate > 0.02:
            # p99를 덮는 가장 작은 2의 거듭제곱(512 이상)을 권장
            need = res["p99_all"]
            rec = 512
            while rec < need:
                rec *= 2
            print(f"⚠️  절단률이 {rate:.1%}입니다(>2%). 잘린 출력을 정답으로 학습하면 모델이 미완성 응답을 배웁니다.\n"
                  f"    조치: max_seq_length를 {rec}로 올리거나(메모리↑), 긴 예시를 제외/요약하세요.")
        else:
            print("✅ 절단 위험 낮음.")
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
# 2) 근사중복 + seed 표절 — 완전중복만 보면 놓친다
# ---------------------------------------------------------------------------
def near_duplicate_report(seed_examples: list, synth_examples: list,
                          *, threshold: int = 90, sample: int = 400) -> dict:
    """합성 내부 근사중복 + 합성이 seed를 거의 그대로 베낀 비율(표절)을 측정한다.

    🔴 왜 완전중복(dedup_rate)으로 부족한가: LLM은 같은 문장을 토씨 하나만 바꿔 반복하는 경향이
       있다("서울 날씨 알려줘" / "서울 날씨 알려 줘"). 완전중복은 0%로 나오지만 실질 다양성은 낮다.
    🔴 seed 표절: 합성이 seed와 거의 같으면 데이터를 늘린 효과가 없고, 그 seed를 held-out으로
       쓸 경우 **평가 누출**이 된다(점수가 부풀려짐).

    threshold: token_set_ratio 유사도(0~100). 90 이상이면 사실상 같은 문장으로 본다.
    sample: 비교 비용은 O(n²)이므로 이 개수까지만 샘플링(기본 400 → 16만 쌍, 수 초).
    """
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
        print("(rapidfuzz 미설치 — 근사중복 점검 생략. uv pip install rapidfuzz)")
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
        print(f"⚠️  합성끼리 너무 닮았습니다({near_rate:.1%} > 15%) — 실질 다양성이 낮습니다.\n"
              "    조치: 생성 temperature를 올리거나, seed를 더 다양하게 샘플링하거나,\n"
              "          프롬프트에 '이전 예시와 다른 상황/어휘를 쓰라'는 제약을 추가하세요.")
    if leak_rate > 0.10:
        print(f"⚠️  합성이 seed를 거의 그대로 베낀 비율이 높습니다({leak_rate:.1%} > 10%).\n"
              "    조치: 증강 효과가 없고, 해당 seed를 held-out으로 쓰면 평가 누출이 됩니다.\n"
              "          04_evaluate의 held-out은 합성에 쓰지 않은 슬라이스에서 고르세요.")
    if near_rate <= 0.15 and leak_rate <= 0.10:
        print("✅ 근사중복·표절 모두 허용 범위.")
    print("=" * 68)
    return res


# ---------------------------------------------------------------------------
# 3) 어휘 다양성 — distinct-n + 시작 n-gram 편중
# ---------------------------------------------------------------------------
def lexical_diversity(seed_examples: list, synth_examples: list, *, top: int = 5) -> dict:
    """distinct-1/2(고유 n-gram 비율)와 '문장 시작 3-gram' 편중을 비교한다.

    🔴 왜: LLM 합성은 같은 템플릿으로 시작하는 경향이 강하다("다음 텍스트에서...", "Please extract...").
       이런 고착은 모델이 **특정 도입부에만 반응**하게 만들어, 실제 사용자 입력에서 성능이 떨어진다.
       distinct-n이 seed보다 뚜렷히 낮으면 표현이 획일적이라는 신호다.
    """
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
        print(f"    {c:>5}건 ({c / max(1, len(t_in)):>5.1%})  '{g}...'")
    sd2, td2 = res["seed"]["distinct_2"], res["synth"]["distinct_2"]
    if sd2 and td2 and td2 < sd2 * 0.7:
        print(f"⚠️  합성 distinct_2({td2})가 seed({sd2})의 70% 미만 — 표현이 획일적입니다.\n"
              "    조치: temperature↑ 또는 프롬프트에 어휘/문체 변화를 명시 요구하세요.")
    if top_share > 0.3:
        print(f"⚠️  합성의 {top_share:.0%}가 같은 3단어로 시작합니다 — 도입부 템플릿이 고착됐습니다.\n"
              "    조치: 프롬프트에서 도입 문구를 다양화하거나, 생성 후 도입부를 다시 쓰세요.")
    if not (sd2 and td2 and td2 < sd2 * 0.7) and top_share <= 0.3:
        print("✅ 어휘 다양성 양호.")
    print("=" * 68)
    return res


# ---------------------------------------------------------------------------
# 4) 라벨/클래스 균형 — 분류 트랙에 특히 중요
# ---------------------------------------------------------------------------
def label_balance(seed_examples: list, synth_examples: list, *, plot: bool = True) -> dict:
    """output을 라벨로 보고 클래스 분포를 비교한다(분류 트랙). 소수 클래스 소실을 잡는다.

    🔴 왜: 합성이 다수 클래스로 쏠리면 모델이 그 클래스만 답하게 된다(정확도는 높아 보이지만
       소수 클래스 recall이 0). output이 짧은 라벨 문자열인 트랙에서 의미가 있다.
    """
    def _labels(pairs):
        c = Counter()
        for _, o in pairs:
            lab = re.sub(r"\s+", " ", str(o).strip().lower())
            if lab and len(lab) <= 40:      # 라벨로 볼 수 있을 만큼 짧은 것만
                c[lab] += 1
        return c

    sp = [_pair_from(e) for e in seed_examples]
    tp = [_pair_from(e) for e in synth_examples]

    # 🔴 라벨 트랙인지 먼저 판별한다. JSON/문장 출력을 라벨로 오인하면 의미 없는 표가 나온다.
    #    기준: output이 JSON처럼 시작하지 않고, 서로 다른 값의 종류가 전체의 절반 미만(=반복되는 범주형).
    def _looks_categorical(pairs) -> bool:
        outs = [str(o).strip() for _, o in pairs if str(o).strip()]
        if not outs:
            return False
        if sum(1 for o in outs if o[:1] in "{[") > len(outs) * 0.2:
            return False                      # JSON 출력 트랙(추출)
        short = [o for o in outs if len(o) <= 40]
        if len(short) < len(outs) * 0.8:
            return False                      # 긴 자유서술(요약·QA)
        return len(set(o.lower() for o in short)) <= max(2, len(short) * 0.5)

    if not _looks_categorical(tp):
        print("(output이 라벨 형태가 아니어서 클래스 균형 점검을 건너뜁니다 — 추출/요약/QA 트랙은 정상)")
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
        print(f"⚠️  합성에 없는 seed 클래스 {len(missing)}개: {missing[:5]}\n"
              "    조치: 해당 클래스 seed를 명시적으로 지정해 추가 생성하세요(클래스별 목표 건수 설정).")
    if imbalance > 5:
        print(f"⚠️  클래스 불균형이 {imbalance}x입니다 — 소수 클래스 recall이 낮아집니다.\n"
              "    조치: 소수 클래스를 추가 생성하거나, 학습 시 클래스 가중치/오버샘플링을 고려하세요.")
    if not missing and imbalance <= 5:
        print("✅ 클래스 균형 양호.")
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
# 5) 출력 유효성 — 학습 전에 '정답이 정답인지' 확인
# ---------------------------------------------------------------------------
def output_validity(synth_examples: list, *, expect_json: bool = True,
                    required_keys: tuple[str, ...] = ("name", "arguments")) -> dict:
    """합성 output이 목표 스키마를 지키는지 검사한다(추출→JSON 트랙).

    🔴 왜: 깨진 JSON을 정답으로 학습하면 모델이 깨진 JSON을 생성하도록 배운다. 합성 파이프라인의
       검증을 통과했더라도, 학습 직전에 한 번 더 보는 편이 값싸다.
    """
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
        print(f"⚠️  파싱률이 {rate:.1%}입니다(<98%) — 깨진 정답으로 학습하면 모델도 깨진 JSON을 냅니다.\n"
              "    조치: 합성 검증(validator)을 강화해 실패 예시를 버리거나, 생성 프롬프트에 스키마를 재명시하세요.")
    elif miss:
        print(f"⚠️  필수키({required_keys}) 누락 {miss}건 — 해당 예시를 제외하는 편이 안전합니다.")
    else:
        print("✅ 출력 스키마 준수.")
    print("=" * 68)
    return {"n": len(outs), "json_ok": ok, "json_rate": rate, "missing_keys": miss}


# ---------------------------------------------------------------------------
# 6) 한 번에 실행 (노트북 기본 경로)
# ---------------------------------------------------------------------------
def quick_report(seed_examples: list, synth_examples: list, *, tokenizer=None,
                 max_seq_length: int | None = None, expect_json: bool = False,
                 plot: bool = True) -> dict:
    """compare + 근사중복 + 어휘다양성 (+토크나이저/JSON 제공 시 해당 항목)을 순서대로 실행.

    tokenizer/max_seq_length를 주면 토큰 길이·절단 위험까지 본다(권장 — 학습 설정에 직결).
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
# 추출(→JSON) 트랙 전용: 함수명/인자 키 커버리지
# ---------------------------------------------------------------------------
def json_field_coverage(seed_examples: list, synth_examples: list) -> dict:
    """output이 function-call JSON일 때, seed vs 합성의 함수명·인자 키 분포 비교."""
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
