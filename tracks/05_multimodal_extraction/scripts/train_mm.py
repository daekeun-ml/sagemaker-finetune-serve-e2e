#!/usr/bin/env python
"""Gemma 4 멀티모달 SFT와 LoRA 학습 스크립트입니다.

텍스트 트랙과의 차이:
  - 입력 이미지는 `AutoProcessor`로 pixel_values를 생성.
  - 모델은 `AutoModelForImageTextToText`(멀티모달 전체). LoRA는 language_model 한정(regex).
  - vision tower를 유지하고 멀티모달 모델 그대로 서빙.
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
import json
import logging
import os

logger = logging.getLogger("gemma_mm")


def _job_name() -> str:
    """이 컨테이너가 속한 SageMaker 학습 잡 이름. 로컬 실행이면 빈 문자열.

    SageMaker 가 TRAINING_JOB_NAME 을 넣어 준다(실측). 그게 없는 경우를 대비해 SM_TRAINING_ENV
    JSON 의 job_name 도 본다. 둘 다 실제 컨테이너에서 값이 확인된 경로다.
    """
    name = (os.environ.get("TRAINING_JOB_NAME") or "").strip()
    if name:
        return name
    try:
        return str(json.loads(os.environ.get("SM_TRAINING_ENV") or "{}").get("job_name", "") or "")
    except (ValueError, AttributeError):
        return ""


class _Mlflow:
    """MLflow 추적을 준비합니다. 연결 실패 시 추적만 끄고 학습은 계속합니다.

    파이프라인의 부모 run은 활성화하지 않고 자식 run만 만듭니다. Trainer의 MLflowCallback은
    `MLFLOW_RUN_ID`를 이어받아 step metric을 기록합니다.
    """

    def __init__(self) -> None:
        self.report_to = "none"
        self.run_name = _job_name()  # 자식 run 이름
        self._child = None           # 이 컨테이너가 만든 자식 run ID
        self._mlflow = None

        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            return                   # 추적 비활성
        try:
            import mlflow
        except ImportError:
            logger.warning("[mlflow] MLFLOW_TRACKING_URI is set but mlflow is not installed; "
                           "tracking disabled (add mlflow to scripts/requirements.txt).")
            return

        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME") or ""
        try:
            # 학습 시작 전에 연결과 experiment 접근을 확인합니다.
            mlflow.set_tracking_uri(uri)
            exp = mlflow.set_experiment(experiment) if experiment else None
        except Exception as e:       # noqa: BLE001. 권한, 삭제된 experiment, 네트워크 무엇이든
            logger.warning("[mlflow] connection check failed; tracking disabled, training "
                           "continues: %s: %s", type(e).__name__, e)
            return

        self._mlflow = mlflow
        self.report_to = "mlflow"
        logger.info("[mlflow] logging step metrics to %s (experiment=%s)",
                    uri, experiment or "(default)")

        # 부모 run이 있으면 자식 run을 미리 만듭니다.
        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        if not parent_run_id:
            return                   # 부모가 없으면 콜백이 최상위 run을 만듭니다.
        if (os.environ.get("RANK") or "0") != "0":
            return                   # 분산 학습이면 rank 0 만
        try:
            # 환경변수의 experiment를 우선하고, 없을 때만 부모 run의 experiment를 사용합니다.
            client = mlflow.MlflowClient()
            exp_id = exp.experiment_id if exp is not None else \
                client.get_run(parent_run_id).info.experiment_id
            child = client.create_run(
                experiment_id=exp_id,
                run_name=self.run_name or None,
                tags={"mlflow.parentRunId": parent_run_id},
            )
            # MLflowCallback은 이 환경변수로 기존 run을 이어받습니다.
            os.environ["MLFLOW_RUN_ID"] = child.info.run_id
            self._child = child.info.run_id
            logger.info("[mlflow] child run %s created under parent %s",
                        self._child, parent_run_id)
        except Exception as e:       # noqa: BLE001. 부모가 지워졌을 수도 있다
            logger.warning("[mlflow] could not create a child run; logging to a top-level run "
                           "instead: %s: %s", type(e).__name__, e)

    def __enter__(self) -> _Mlflow:
        return self

    def __exit__(self, exc_type: object, *_rest: object) -> None:
        """남아 있는 자식 run을 닫습니다. 부모 run은 파이프라인이 관리합니다."""
        if self._mlflow is None or self._child is None:
            return
        try:
            if self._mlflow.active_run() is not None:
                self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        except Exception as e:       # noqa: BLE001
            logger.warning("[mlflow] failed to end the child run (ignored): %s: %s",
                           type(e).__name__, e)


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
    """cord-v2 ground_truth를 간결한 타깃 JSON으로 변환합니다."""
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
    messages content에는 이미지 placeholder를 직접 넣지 않습니다."""
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": _simplify_gt(example.get("ground_truth", ""))},
        ],
    }


def _revive_kv_shared_from_base(save_sd, cfg, model_id, hf_token, logger) -> int:
    """멀티모달 키 경로에서 KV 공유 레이어의 서빙 검증용 텐서를 복원합니다.

    train.py의 동일 함수와 다른 점: 멀티모달 arch를 그대로 저장하므로 키가
    'model.language_model.layers.N...' 형태다(텍스트 re-export의 'model.layers.N...'이 아님).
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
        logger.warning("skipping KV-shared restore: could not find the language_model key prefix")
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
        logger.warning("KV-shared restore failed (safetensors missing: %s)", e)
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
        logger.warning("KV-shared restore incomplete: %d/%d (failure example: %s)",
                       revived, len(need), sorted(remaining)[:3])
    else:
        logger.info("restored %d KV-shared tensors (layers %d-%d); vLLM/SGLang can serve this",
                    revived, first, n_layers - 1)
    return revived


def _prune_artifact(output_dir: str, logger) -> None:
    """서빙에 쓰이지 않는 체크포인트와 머지용 어댑터를 아티팩트에서 제거합니다."""
    import shutil

    removed = 0
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("checkpoint-") or name == "adapter":
            size = sum(os.path.getsize(os.path.join(r, f))
                       for r, _, fs in os.walk(path) for f in fs)
            shutil.rmtree(path, ignore_errors=True)
            removed += size
            logger.info("pruned %s (%.2f GB): not used for serving", name, size / 1024**3)
    if removed:
        logger.info("removed %.2f GB from the artifact: shorter upload and lower S3 cost",
                    removed / 1024**3)


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
        f"{args.model_id} 은 멀티모달 모델이 아닙니다. 이미지-JSON 트랙에는 멀티모달 base가 필요합니다.")
    logger.info("multimodal model_type=%s", _cfg.model_type)

    # ---- 데이터 (cord-v2: image + ground_truth) ----
    n = 16 if args.dry_run else (args.max_train_samples or 500)
    ds = load_dataset(args.seed_dataset, split="train", token=hf_token)
    if len(ds) > n:
        ds = ds.select(range(n))
    ds = ds.map(_to_messages, remove_columns=list(ds.column_names))  # images와 messages만 남깁니다.
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
        logger.info("froze %d vision/audio params (training language LoRA only)", frozen)

    # LoRA는 language_model 경로에만 적용합니다.
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )

    # MLflow 연결 결과를 SFTConfig의 report_to에 반영합니다.
    mlf = _Mlflow()

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
        # 업로드 아티팩트가 커지지 않도록 체크포인트 하나만 유지합니다.
        save_total_limit=1,
        report_to=mlf.report_to,
        # SageMaker 학습 Job 이름으로 MLflow run과 CloudWatch 로그를 연결합니다.
        run_name=mlf.run_name or None,
        dataset_kwargs={"skip_prepare_dataset": True},   # VLM: collator가 이미지 처리(사전 토크나이즈 스킵)
        remove_unused_columns=False,                     # image 컬럼 유지
    )

    # TRL VLM collator가 이미지를 pixel_values로 변환합니다.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=processor,
    )

    logger.info("Starting multimodal SFT...")
    # 컨텍스트 종료 시 남아 있는 자식 run을 정리합니다.
    with mlf:
        trainer.train()

    # 멀티모달 서빙을 위해 텍스트 모델로 다시 저장하지 않습니다.
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        processor.save_pretrained(adapter_dir)
        from peft import PeftModel
        # 병합 전에 학습 모델을 해제하고 base를 CPU에 로드합니다.
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
        # 멀티모달 전체 모델을 서빙 루트에 저장하고 필요한 KV 공유 텐서를 복원합니다.
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

    if not args.dry_run:
        _prune_artifact(args.output_dir, logger)

    if args.dry_run:
        logger.info("DRY-RUN complete: multimodal pipeline OK.")


if __name__ == "__main__":
    main()
