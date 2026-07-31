#!/usr/bin/env python
"""
train_grpo.py — Gemma GRPO(+LoRA/QLoRA) 학습 스크립트 (self-contained)

🔴 SFT(train.py)와의 차이:
  - SFT는 정답 completion을 '모방'. GRPO는 prompt당 여러 개를 생성해 **reward 함수**로 좋은 걸 강화.
  - 따라서 데이터는 {"prompt":[...user...], "reference":"<정답>"} 형태(정답은 reward 계산용).
  - reward가 '프로그램적으로 명확한' 태스크에만 적합 → 이 킷은 **추출(JSON)·분류(라벨)** 트랙에만 GRPO 노트북 제공.
  - GRPO는 prompt당 num_generations개 생성(rollout)이라 SFT보다 연산량이 크다(시간·GPU↑).

멀티모달 base(gemma-4 전부·gemma-3 4b+) 처리는 train.py와 동일:
  - AutoModelForImageTextToText로 로드, LoRA는 language_model 한정(regex), 머지 후 텍스트 재-export.

로컬 dry-run:
    python train_grpo.py --dry_run --reward_kind extraction \
        --model_id google/gemma-4-E4B-it --train_file ./sample.jsonl --output_dir ./out
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re

logger = logging.getLogger("gemma_grpo")


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ("botocore", "urllib3", "s3transfer", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _str2bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 🔴 정석 RLHF: GRPO는 보통 'SFT된 모델'에서 이어서 학습한다. base 우선순위:
    #   1) --base_model_dir (SFT 아티팩트를 마운트한 컨테이너 경로, 예 SM_CHANNEL_MODEL=/opt/ml/input/data/model)
    #   2) --model_id (HF base — SFT 없이 base에서 바로 GRPO할 때)
    p.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"))
    p.add_argument("--base_model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL"),
                   help="SFT 산출물(재-export된 텍스트 모델) 디렉토리. 있으면 이걸 base로 사용(SFT→GRPO).")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./out"))
    # 🔴 reward 종류 — 이 트랙의 성공 기준을 프로그램적으로 채점. extraction | classification.
    p.add_argument("--reward_kind", type=str, default="extraction", choices=["extraction", "classification"])
    # GRPO 하이퍼
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)   # GRPO는 SFT보다 낮게
    p.add_argument("--num_generations", type=int, default=8)      # prompt당 생성 수(그룹). 클수록 신호↑·연산↑
    # ⚠️ TRL v1.6.0 에서 GRPOConfig 의 max_prompt_length 가 제거됐다. 인자는 하위호환을 위해
    #    남겨 두지만, 지원하지 않는 버전에서는 자동으로 제외된다(아래 GRPOConfig 구성 참고).
    p.add_argument("--max_prompt_length", type=int, default=1024,
                   help="구 TRL(<1.6)에서만 사용. 최신 TRL은 프롬프트 절단을 별도로 다룬다.")
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)      # rollout 다양성
    p.add_argument("--beta", type=float, default=0.04)            # KL 계수
    p.add_argument("--max_seq_length", type=int, default=2048)
    # LoRA / QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", type=_str2bool, nargs="?", const=True, default=False)
    p.add_argument("--merge_adapter", type=_str2bool, nargs="?", const=True, default=True)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--max_train_samples", type=int, default=None, help="학습에 쓸 최대 샘플 수(앞에서부터). 미지정 시 전체.")
    p.add_argument("--dry_run", type=_str2bool, nargs="?", const=True, default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# reward 함수 (프로그램적 — reference와 completion 비교)
# ---------------------------------------------------------------------------
def _extract_json(text: str):
    """completion에서 첫 JSON 객체를 관대하게 추출."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


def reward_extraction(completions, reference, **kwargs):
    """추출(JSON) reward: 유효 JSON(0.3) + name 일치(0.3) + arguments 키/값 F1(0.4). 0~1."""
    refs = reference  # GRPO가 dataset의 'reference' 컬럼을 kwargs로 전달(리스트, 그룹 브로드캐스트)
    out = []
    for comp, ref in zip(completions, refs):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        pred = _extract_json(text)
        r = 0.0
        if pred is not None and isinstance(pred, dict):
            r += 0.3                                   # 유효 JSON
            gold = _extract_json(ref) or {}
            if pred.get("name") and pred.get("name") == gold.get("name"):
                r += 0.3                               # 함수명 일치
            pa, ga = pred.get("arguments", {}), gold.get("arguments", {})
            if isinstance(pa, dict) and isinstance(ga, dict) and ga:
                inter = sum(1 for k in ga if k in pa and str(pa[k]) == str(ga[k]))
                prec = inter / max(1, len(pa)); rec = inter / max(1, len(ga))
                f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
                r += 0.4 * f1                          # 인자 F1
        out.append(r)
    return out


def reward_classification(completions, reference, **kwargs):
    """분류 reward: 예측 라벨이 정답 라벨과 정확 일치하면 1.0, (부분 포함 0.3) 아니면 0."""
    out = []
    for comp, ref in zip(completions, reference):
        text = (comp[-1]["content"] if isinstance(comp, list) else comp).strip().lower()
        gold = ref.strip().lower()
        # 첫 토큰/줄을 라벨로 간주(모델이 라벨만 출력하도록 프롬프트됨)
        pred = text.splitlines()[0].strip() if text else ""
        if pred == gold:
            out.append(1.0)
        elif gold and gold in text:
            out.append(0.3)          # 라벨이 텍스트에 포함(형식 어긋남) — 약한 보상
        else:
            out.append(0.0)
    return out


REWARDS = {"extraction": reward_extraction, "classification": reward_classification}


# ---------------------------------------------------------------------------
# 데이터: {"messages":[user, assistant]} → {"prompt":[user], "reference": assistant}
# ---------------------------------------------------------------------------
def resolve_train_path(args) -> str:
    if args.train_file:
        return args.train_file
    ch = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    for fn in sorted(os.listdir(ch)):
        if fn.endswith(".jsonl"):
            return os.path.join(ch, fn)
    raise FileNotFoundError(f"No training file (.jsonl) in: {ch}")


def _to_grpo(example):
    """SFT용 messages를 GRPO용 prompt/reference로 변환. reference는 reward 계산용(학습 입력 아님)."""
    msgs = example["messages"]
    prompt = [m for m in msgs if m["role"] != "assistant"]
    reference = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    return {"prompt": prompt, "reference": reference}


def _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger) -> int:
    """KV-shared 레이어의 k_norm/k_proj/v_proj를 base에서 복원(train.py와 동일 로직·근거).

    🔴 gemma-4 E2B/E4B는 뒤쪽 num_kv_shared_layers개 레이어가 앞 레이어의 KV를 재사용하고,
       transformers는 그 레이어에 k_norm/k_proj/v_proj 모듈을 만들지 않는다 → save_pretrained 시
       원본에 있던 텐서가 소실 → vLLM은 전 레이어에 k_norm을 등록하므로 "weights not initialized"
       로 엔진 초기화 실패(vLLM issue #44788). 이 텐서는 연산에 쓰이지 않아(shared 레이어는 앞
       레이어 KV 재사용) base 값 복원은 정확도에 무해하다. 12B/26B/31B는 shared=0이라 무관.
    """
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0) or 0)
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    # model_id는 문자열 또는 후보 목록. GRPO는 base_src(=SFT 산출 디렉터리)가 먼저 오고, 그게
    # 픽스 이전 산출물이면 54개가 없으므로 HF base로 폴백한다.
    sources = [s for s in ([model_id] if isinstance(model_id, str) else list(model_id or [])) if s]
    if n_shared <= 0 or n_layers <= 0 or not sources:
        return 0
    first = n_layers - n_shared
    want = [f"model.layers.{i}.self_attn.{n}.weight"
            for i in range(first, n_layers) for n in ("k_norm", "k_proj", "v_proj")]
    need = [k for k in want if k not in save_sd]
    if not need:
        return 0
    try:
        from safetensors import safe_open
        from transformers.utils import cached_file
    except ImportError as e:
        logger.warning("KV-shared 복원 실패(safetensors 없음: %s)", e)
        return 0

    revived, remaining = 0, set(need)
    for src in sources:
        if not remaining:
            break
        try:  # 샤딩 인덱스가 있으면 그 목록, 없으면 단일 파일
            import json as _json
            with open(cached_file(src, "model.safetensors.index.json", token=hf_token)) as f:
                files = sorted(set(_json.load(f)["weight_map"].values()))
        except Exception:
            files = ["model.safetensors"]
        for fname in files:
            if not remaining:
                break
            try:
                path = cached_file(src, fname, token=hf_token)
            except Exception:
                continue  # 다음 샤드/다음 소스로
            with safe_open(path, framework="pt", device="cpu") as f:
                avail = set(f.keys())
                for tk in list(remaining):
                    sfx = tk[len("model."):]
                    for bk in (f"model.language_model.{sfx}", f"language_model.{sfx}", tk):
                        if bk in avail:
                            save_sd[tk] = f.get_tensor(bk)
                            remaining.discard(tk); revived += 1
                            break
    if remaining:
        logger.warning("KV-shared 복원 불완전: %d/%d개(실패 예: %s)",
                       revived, len(need), sorted(remaining)[:3])
    else:
        logger.info("KV-shared 텐서 %d개 복원(레이어 %d~%d) → vLLM/SGLang 서빙 가능",
                    revived, first, n_layers - 1)
    return revived


def _reexport_text_only(merged, full_cfg, tokenizer, output_dir, logger,
                        model_id=None, hf_token=None):
    """멀티모달 머지 모델 → language 서브모듈만 텍스트 arch로 재-export (train.py와 동일 로직).

    model_id/hf_token: E2B/E4B의 KV-shared dead weight를 base에서 복원하는 데 필요.
    """
    import torch
    import transformers as T
    text_cfg = full_cfg.text_config
    arch = (full_cfg.architectures or [""])[0]
    text_cls_name = "Gemma4UnifiedForCausalLM" if "Unified" in arch else "Gemma4ForCausalLM"
    TextCls = getattr(T, text_cls_name, None)
    if TextCls is None:
        from transformers import AutoModelForCausalLM as TextCls  # type: ignore
    lm_prefix = next((n + "." for n, _ in merged.named_modules() if n.endswith("language_model")), None)
    if lm_prefix is None:
        raise RuntimeError("language_model submodule not found")
    msd = merged.state_dict()
    text_sd = {}
    for k, v in msd.items():
        if k.startswith(lm_prefix):
            text_sd["model." + k[len(lm_prefix):]] = v
        elif k.startswith("lm_head."):
            text_sd[k] = v
    # 🔴 OOM 방지: meta 뼈대 + assign=True로 사본 없이 이식(TextCls(text_cfg)는 fp32 사본 생성).
    import gc
    from accelerate import init_empty_weights
    with init_empty_weights():
        text_model = TextCls(text_cfg)
    text_model.load_state_dict(text_sd, strict=False, assign=True)
    text_model = text_model.to(torch.bfloat16)
    # 🔴 KV-shared dead weight 복원 후 명시 state_dict로 저장(모델엔 그 모듈이 없어 이 방법뿐).
    save_sd = text_model.state_dict()
    if _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger):
        save_sd = {k: (v.to(torch.bfloat16) if hasattr(v, "to") else v) for k, v in save_sd.items()}
    text_model.save_pretrained(output_dir, safe_serialization=True, state_dict=save_sd)
    tokenizer.save_pretrained(output_dir)
    del text_model, text_sd, msd, save_sd
    gc.collect()
    logger.info("Re-exported TEXT-ONLY (%s) to %s", text_cls_name, output_dir)


def main() -> None:
    _configure_logging()
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)
    from trl import GRPOConfig, GRPOTrainer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.dry_run:
        args.epochs = 1
        args.num_generations = min(args.num_generations, 4)
        args.max_completion_length = min(args.max_completion_length, 64)
        logger.info("DRY-RUN: epochs=1, num_generations<=4, short completions")

    # ---- base 모델 소스 결정: SFT 산출물(base_model_dir) 우선, 없으면 HF base(model_id) ----
    if args.base_model_dir and os.path.isfile(os.path.join(args.base_model_dir, "config.json")):
        base_src = args.base_model_dir
        logger.info("GRPO from SFT checkpoint: %s (정석 RLHF: SFT→GRPO)", base_src)
    else:
        base_src = args.model_id
        logger.info("GRPO from HF base: %s (SFT 없이 base에서 GRPO)", base_src)

    # ---- 멀티모달 감지 (train.py와 동일) ----
    #   SFT 산출물은 텍스트 재-export(gemma4_text)라 is_multimodal=False로 감지됨 → CausalLM 로드·재-export 불필요.
    _cfg = AutoConfig.from_pretrained(base_src, token=hf_token)
    is_multimodal = getattr(_cfg, "text_config", None) is not None or hasattr(_cfg, "vision_config")
    logger.info("model_type=%s multimodal=%s reward_kind=%s", _cfg.model_type, is_multimodal, args.reward_kind)

    # ---- 데이터 ----
    train_path = resolve_train_path(args)
    ds = load_dataset("json", data_files=train_path, split="train")
    if args.dry_run:
        ds = ds.select(range(min(16, len(ds))))
    elif args.max_train_samples and args.max_train_samples > 0:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
    ds = ds.map(_to_grpo, remove_columns=[c for c in ds.column_names if c != "reference"])
    logger.info("GRPO examples: %d (file: %s)", len(ds), train_path)

    # ---- 토크나이저 / 모델 (base_src = SFT 산출물 또는 HF base) ----
    tokenizer = AutoTokenizer.from_pretrained(base_src, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = dict(attn_implementation=args.attn_implementation, dtype=torch.bfloat16, token=hf_token)
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if is_multimodal:
        model = AutoModelForImageTextToText.from_pretrained(base_src, **model_kwargs)
        lora_targets = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        modules_to_save = None
    else:
        model = AutoModelForCausalLM.from_pretrained(base_src, **model_kwargs)
        lora_targets = "all-linear"
        modules_to_save = ["lm_head", "embed_tokens"]
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             bias="none", task_type="CAUSAL_LM",
                             target_modules=lora_targets, modules_to_save=modules_to_save)

    # ---- GRPO ----
    # 🔴 GRPOConfig 인자는 TRL 버전마다 바뀐다(실측 2026-07-31):
    #    `max_prompt_length` 는 **TRL v1.6.0에서 제거**됐다(릴리스 노트: "Remove invalid
    #    max_prompt_length argument from GRPO"). 컨테이너가 trl>=1.8 을 설치하므로 그걸 그대로
    #    넘기면 `TypeError: GRPOConfig.__init__() got an unexpected keyword argument` 로 죽는다.
    #    → 지원되는 필드만 골라 넘긴다. 새 인자가 생겨도 이 방식이면 코드 수정 없이 견딘다.
    import dataclasses as _dc
    _supported = {f.name for f in _dc.fields(GRPOConfig)}
    _wanted = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,   # 구 TRL(<1.6)에만 존재
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "beta": args.beta,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 5 if args.dry_run else 10,
        "save_strategy": "no" if args.dry_run else "epoch",
        # 체크포인트 1개만 — /opt/ml/model 전체가 아티팩트로 업로드된다(업로드도 MaxRuntime 포함).
        "save_total_limit": 1,
        "report_to": "none",
    }
    _dropped = sorted(set(_wanted) - _supported)
    if _dropped:
        logger.info("GRPOConfig: 이 TRL 버전이 지원하지 않는 인자 제외 → %s", _dropped)
    grpo_config = GRPOConfig(**{k: v for k, v in _wanted.items() if k in _supported})
    reward_fn = REWARDS[args.reward_kind]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training (num_generations=%d)...", args.num_generations)
    trainer.train()

    # ---- 저장 (train.py와 동일: 멀티모달이면 텍스트 재-export) ----
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        from peft import PeftModel
        # 🔴 OOM 방지: 학습 모델/trainer 먼저 해제 후 base(bf16)를 CPU에 로드(train.py와 동일).
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        base = _loader.from_pretrained(
            base_src, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation, token=hf_token, device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        del base
        gc.collect()
        if is_multimodal:
            # base_src(SFT 산출물)를 먼저 보고, 없으면 HF base(args.model_id)로 폴백.
            _reexport_text_only(merged, _cfg, tokenizer, args.output_dir, logger,
                                model_id=[base_src, args.model_id], hf_token=hf_token)
        else:
            merged.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        logger.info("Merged model saved to serving root: %s", args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.dry_run:
        logger.info("DRY-RUN complete — GRPO pipeline OK.")


if __name__ == "__main__":
    main()
