#!/usr/bin/env python
"""
train_mm.py — Gemma-4 멀티모달 SFT (이미지→JSON) + LoRA. self-contained.

🔴 텍스트 트랙(train.py)과의 차이:
  - 입력에 '이미지'가 포함 → `AutoProcessor`(Gemma4Processor)로 pixel_values 생성.
  - 모델은 `AutoModelForImageTextToText`(멀티모달 전체). LoRA는 language_model 한정(regex).
  - 🔴 서빙도 멀티모달로 하므로 **텍스트 재-export를 하지 않는다**(vision tower 유지).
    → 배포 시 vLLM이 이미지 입력을 받도록 그대로 서빙.
  - TRL SFTTrainer가 processing_class=processor를 받으면 내장 VLM collator로 이미지를 자동 처리.
  - 데이터: {"messages":[{role:user, content:[{type:image},{type:text}]}, {role:assistant,...}]}

로컬 dry-run:
    python train_mm.py --dry_run --model_id google/gemma-4-E4B-it \
        --seed_dataset naver-clova-ix/cord-v2 --output_dir ./out
  (학습 데이터는 항상 --seed_dataset 에서 이미지와 함께 내려받습니다. dry-run은 앞 16건만 씁니다.)

요구: torchvision (Gemma4 image processor 의존).
"""
from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("gemma_mm")


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
    p.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"))
    p.add_argument("--seed_dataset", type=str, default="naver-clova-ix/cord-v2")
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./out"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", type=_str2bool, nargs="?", const=True, default=True)
    p.add_argument("--merge_adapter", type=_str2bool, nargs="?", const=True, default=True)
    p.add_argument("--freeze_vision", type=_str2bool, nargs="?", const=True, default=True,
                   help="vision tower를 얼리고 language LoRA만 학습(권장, 안정적).")
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--dry_run", type=_str2bool, nargs="?", const=True, default=False)
    return p.parse_args()


def _simplify_gt(ground_truth: str) -> str:
    """cord-v2 ground_truth → 간결 타깃 JSON (track_data._simplify_gt와 동일 로직, self-contained)."""
    import json
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return ground_truth
    parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else {}
    menu = parse.get("menu", []) if isinstance(parse, dict) else []
    if isinstance(menu, dict):
        menu = [menu]
    items = [{"name": m.get("nm", ""), "count": m.get("cnt", ""), "price": m.get("price", "")}
             for m in menu if isinstance(m, dict)]
    return json.dumps({"menu": items}, ensure_ascii=False)


INSTRUCTION = ("You are a receipt-parsing engine. Extract the receipt into strict JSON with a 'menu' array "
               "of {name, count, price} items. Output ONLY valid JSON, no prose.")


def _to_messages(example):
    """TRL VLM 포맷: messages는 텍스트만 + 별도 images 컬럼. collator가 이미지 placeholder를 주입한다.
    (messages content 안에 {type:image}를 직접 넣으면 'images ≠ placeholders' 에러 — 실측 확인.)"""
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": _simplify_gt(example.get("ground_truth", ""))},
        ],
    }


def _revive_kv_shared_from_base(save_sd, cfg, model_id, hf_token, logger) -> int:
    """KV-shared 레이어의 k_norm/k_proj/v_proj를 base에서 복원(멀티모달 키 접두사 버전).

    🔴 gemma-4 E2B/E4B는 뒤쪽 num_kv_shared_layers개 레이어가 앞 레이어의 KV를 재사용하고,
       transformers는 그 레이어에 k_norm/k_proj/v_proj 모듈을 만들지 않는다
       (modeling_gemma4.py "Layers sharing kv states don't need any weight matrices").
       → save_pretrained 시 원본에 있던 텐서가 소실(E4B 실측 54개 = 18층 × 3).
       vLLM/SGLang은 k_norm을 전 레이어에 등록하므로 "weights not initialized" ValueError로
       엔진 초기화 실패(vLLM issue #44788). 이 텐서는 연산에 쓰이지 않아(shared 레이어는 앞 레이어
       KV 재사용) base 값 복원은 정확도에 무해하다. 12B/26B/31B는 shared=0이라 무관.

    train.py의 동일 함수와 다른 점: 멀티모달 arch를 그대로 저장하므로 키가
    'model.language_model.layers.N...' 형태다(텍스트 재-export의 'model.layers.N...'이 아님).
    """
    tc = getattr(cfg, "text_config", None) or cfg
    n_shared = int(getattr(tc, "num_kv_shared_layers", 0) or 0)
    n_layers = int(getattr(tc, "num_hidden_layers", 0) or 0)
    if n_shared <= 0 or n_layers <= 0 or not model_id:
        return 0
    first = n_layers - n_shared
    # 저장 대상 키 접두사를 실제 state_dict에서 추론(model.language_model. / language_model.)
    prefix = next((p for p in ("model.language_model.", "language_model.")
                   if any(k.startswith(p) for k in save_sd)), None)
    if prefix is None:
        logger.warning("KV-shared 복원 생략: language_model 키 접두사를 찾지 못했습니다")
        return 0
    want = [f"{prefix}layers.{i}.self_attn.{n}.weight"
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
    try:  # 샤딩 인덱스가 있으면 그 목록, 없으면 단일 파일
        import json as _json
        with open(cached_file(model_id, "model.safetensors.index.json", token=hf_token)) as f:
            files = sorted(set(_json.load(f)["weight_map"].values()))
    except Exception:
        files = ["model.safetensors"]

    revived, remaining = 0, set(need)
    for fname in files:
        if not remaining:
            break
        try:
            path = cached_file(model_id, fname, token=hf_token)
        except Exception:
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            avail = set(f.keys())
            for tk in list(remaining):
                sfx = tk[len(prefix):]
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


def main() -> None:
    _configure_logging()
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.dry_run:
        args.epochs = 1
        logger.info("DRY-RUN: epochs=1, small subset")

    _cfg = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    assert getattr(_cfg, "vision_config", None) is not None or getattr(_cfg, "text_config", None) is not None, (
        f"{args.model_id} 은 멀티모달 모델이 아닙니다 — 이미지→JSON 트랙엔 gemma-4 등 멀티모달 base가 필요합니다.")
    logger.info("multimodal model_type=%s", _cfg.model_type)

    # ---- 데이터 (cord-v2: image + ground_truth) ----
    n = 16 if args.dry_run else (args.max_train_samples or 500)
    ds = load_dataset(args.seed_dataset, split="train", token=hf_token)
    if len(ds) > n:
        ds = ds.select(range(n))
    ds = ds.map(_to_messages, remove_columns=list(ds.column_names))  # 원본(image, ground_truth) 제거 → images+messages만
    logger.info("MM examples: %d (dataset: %s, cols=%s)", len(ds), args.seed_dataset, ds.column_names)

    # ---- processor / 모델 ----
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    model_kwargs = dict(attn_implementation=args.attn_implementation, dtype=torch.bfloat16, token=hf_token)
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    # ---- vision tower freeze (권장) ----
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name.lower() or "audio" in name.lower():
                param.requires_grad = False
                frozen += 1
        logger.info("froze %d vision/audio params (language LoRA만 학습)", frozen)

    # ---- LoRA: language_model 한정 (텍스트 트랙과 동일 이유 — vision proj는 ClippableLinear라 제외) ----
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5 if args.dry_run else 10,
        save_strategy="no" if args.dry_run else "epoch",
        # 🔴 체크포인트 1개만 유지 — /opt/ml/model 전체가 model.tar.gz로 업로드되므로 쌓이면
        #    아티팩트가 커지고 업로드가 길어진다(업로드 시간도 MaxRuntime에 포함). 서빙엔 불필요.
        save_total_limit=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},   # VLM: collator가 이미지 처리(사전 토크나이즈 스킵)
        remove_unused_columns=False,                     # image 컬럼 유지
    )

    # 🔴 processing_class=processor → TRL 내장 VLM collator가 이미지를 pixel_values로 자동 변환.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=processor,
    )

    logger.info("Starting multimodal SFT...")
    trainer.train()

    # ---- 저장 (🔴 멀티모달 서빙 유지 — 텍스트 재-export 하지 않음) ----
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        processor.save_pretrained(adapter_dir)
        from peft import PeftModel
        # 🔴 OOM 방지: 학습 모델/trainer 먼저 해제 후 base(bf16)를 CPU에 로드. 멀티모달 full 모델은
        #    vision+audio 포함이라 특히 크다(호스트 RAM 여유 필요 — g6.4xlarge 이상 권장).
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base = AutoModelForImageTextToText.from_pretrained(
            args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation, token=hf_token, device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        del base
        gc.collect()
        # 멀티모달 전체 모델을 루트에 저장 → vLLM이 이미지 입력을 받는 멀티모달 endpoint로 서빙.
        # 🔴 저장 직전에 KV-shared dead weight를 base에서 복원(E2B/E4B). 모델 객체엔 그 모듈이 아예
        #    없으므로 명시 state_dict 전달이 유일한 방법이다. 상세: _revive_kv_shared_from_base.
        save_sd = merged.state_dict()
        _cfg = getattr(merged, "config", None)
        if _cfg is not None and _revive_kv_shared_from_base(
                save_sd, _cfg, args.model_id, hf_token, logger):
            save_sd = {k: (v.to(torch.bfloat16) if hasattr(v, "to") else v)
                       for k, v in save_sd.items()}
        merged.save_pretrained(args.output_dir, safe_serialization=True, state_dict=save_sd)
        processor.save_pretrained(args.output_dir)
        del merged, save_sd
        gc.collect()
        logger.info("Merged MULTIMODAL model saved to serving root: %s", args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if args.dry_run:
        logger.info("DRY-RUN complete — multimodal pipeline OK.")


if __name__ == "__main__":
    main()
