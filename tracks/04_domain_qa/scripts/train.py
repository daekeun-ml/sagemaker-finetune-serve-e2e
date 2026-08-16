#!/usr/bin/env python
"""Gemma SFT와 LoRA/QLoRA 학습 스크립트입니다.

SageMaker가 ``source_dir``만 컨테이너에 올리므로 common 패키지에 의존하지 않습니다.

로컬 dry-run (개발환경 GPU에서 파이프라인 검증):
    python train.py --dry_run \
        --model_id google/gemma-4-E4B-it \
        --train_file ./sample.jsonl \
        --output_dir ./out

SageMaker (HuggingFace estimator entry_point):
    hyperparameters={"model_id": "...", "epochs": 3, "use_qlora": True, ...}
    채널: SM_CHANNEL_TRAIN(=/opt/ml/input/data/train), 모델은 SM_MODEL_DIR(=/opt/ml/model)로.

구성:
  - TRL SFTTrainer가 conversational ``messages`` 데이터에 chat template을 적용합니다.
  - LoRA: r=16/alpha=16/dropout=0.05, target_modules='all-linear',
    modules_to_save=['lm_head','embed_tokens'] (특수토큰 학습).
  - bf16 필수 (fp16는 Gemma에서 오버플로/NaN). gradient_checkpointing(use_reentrant=False).
  - attn_implementation='eager' 가 Gemma 안전 기본 (soft-cap/sliding-window 정합성).
  - gated 모델(gemma-3/2/3n)은 HF_TOKEN env 필요. gemma-4 계열은 불필요(apache-2.0/ungated).

멀티모달 base의 텍스트 SFT:
  - 로드: AutoModelForImageTextToText(멀티모달 전체). LoRA는 language_model 한정 target_modules.
  - 저장: 머지 후 language 서브모듈만 텍스트 arch(*ForCausalLM, model_type=*_text)로 re-export
    하여 vLLM이 텍스트 경로로 로드하도록 합니다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

# SageMaker와 CloudWatch에서 수집할 수 있도록 logging을 사용합니다.
logger = logging.getLogger("gemma_train")


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
    """진입점 전용 로깅 구성(멱등). SageMaker/CloudWatch는 stdout을 수집한다."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 서드파티 소음 완화
    for noisy in ("botocore", "urllib3", "s3transfer", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _str2bool(v) -> bool:
    """SageMaker HuggingFace estimator는 모든 하이퍼파라미터를 `--key value`로 직렬화하므로
    boolean도 `--use_qlora True` 형태로 전달됩니다."""
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 모델/데이터
    p.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"))
    p.add_argument("--train_file", type=str, default=None, help="로컬 dry-run용 JSONL. 미지정 시 SM_CHANNEL_TRAIN 사용")
    p.add_argument("--eval_file", type=str, default=None)
    # 출력 (SageMaker는 SM_MODEL_DIR로)
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./out"))
    # 학습 하이퍼
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    # 로컬 bare flag와 SageMaker의 명시적 boolean 값을 모두 지원합니다.
    p.add_argument("--packing", type=_str2bool, nargs="?", const=True, default=True)
    # LoRA / QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", type=_str2bool, nargs="?", const=True, default=False, help="4bit nf4 양자화(작은 GPU)")
    p.add_argument("--merge_adapter", type=_str2bool, nargs="?", const=True, default=True, help="학습 후 LoRA를 base에 머지(서빙용)")
    # attention
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    # 데이터 파일은 유지하고 앞 N건만 학습에 사용합니다.
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="학습에 쓸 최대 샘플 수(앞에서부터). 미지정 시 train.jsonl 전체 사용.")
    # dry-run
    p.add_argument("--dry_run", type=_str2bool, nargs="?", const=True, default=False, help="소량의 짧은 학습으로 파이프라인 검증")
    return p.parse_args()


def _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger) -> int:
    """KV 공유 레이어의 서빙 검증용 텐서를 base 체크포인트에서 복원합니다.

    E2B와 E4B는 ``save_pretrained`` 과정에서 일부 미사용 텐서가 빠질 수 있지만 vLLM은 해당 키를
    요구합니다. 연산에는 쓰이지 않으므로 base 값을 복원해도 학습 결과는 바뀌지 않습니다.
    """
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0) or 0)
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    if n_shared <= 0 or n_layers <= 0:
        return 0  # KV 공유가 없는 모델
    # model_id는 문자열 또는 후보 목록(앞에서부터 시도). 로컬 디렉터리와 HF repo id 모두 가능.
    sources = [s for s in ([model_id] if isinstance(model_id, str) else list(model_id or [])) if s]
    if not sources:
        logger.warning("skipping KV-shared restore (no model_id); vLLM weight validation may fail")
        return 0

    first = n_layers - n_shared
    want = [f"model.layers.{i}.self_attn.{n}.weight"
            for i in range(first, n_layers)
            for n in ("k_norm", "k_proj", "v_proj")]
    need = [k for k in want if k not in save_sd]
    if not need:
        logger.info("KV-shared tensors already intact (%d found); no restore needed", len(want))
        return 0

    # base 체크포인트에서 필요한 텐서만 읽습니다.
    try:
        from safetensors import safe_open
        from transformers.utils import cached_file
    except ImportError as e:
        logger.warning("KV-shared restore failed (safetensors/transformers utils missing: %s)", e)
        return 0

    # 멀티모달 접두사 유무를 모두 지원합니다.
    def _base_keys(tk):
        suffix = tk[len("model."):]                       # layers.N.self_attn.k_norm.weight
        return (f"model.language_model.{suffix}", f"language_model.{suffix}", tk)

    revived = 0
    remaining = set(need)
    for src in sources:
        if not remaining:
            break
        try:  # 샤딩 인덱스가 없으면 단일 파일을 사용합니다.
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
                continue  # 다음 샤드 또는 소스를 확인합니다.
            with safe_open(path, framework="pt", device="cpu") as f:
                avail = set(f.keys())
                for tk in list(remaining):
                    for bk in _base_keys(tk):
                        if bk in avail:
                            save_sd[tk] = f.get_tensor(bk)
                            remaining.discard(tk)
                            revived += 1
                            break

    if remaining:
        logger.warning("KV-shared restore incomplete: %d/%d restored, %d failed (e.g. %s)",
                       revived, len(need), len(remaining), sorted(remaining)[:3])
    else:
        logger.info("restored %d KV-shared tensors (layers %d-%d); vLLM/SGLang can serve this",
                    revived, first, n_layers - 1)
    return revived


def _reexport_text_only(merged, full_cfg, tokenizer, output_dir, logger,
                        model_id=None, hf_token=None, revive_kv_shared=True) -> None:
    """멀티모달 머지 모델의 language 서브모듈을 텍스트 모델로 다시 저장합니다.

    필요하면 E2B와 E4B의 KV 공유 텐서도 base 체크포인트에서 복원합니다.
    """
    import torch
    import transformers as T

    text_cfg = full_cfg.text_config
    arch = (full_cfg.architectures or [""])[0]
    text_cls_name = "Gemma4UnifiedForCausalLM" if "Unified" in arch else "Gemma4ForCausalLM"
    TextCls = getattr(T, text_cls_name, None)
    if TextCls is None:  # 미래 아키텍처 폴백: text_config로 AutoModelForCausalLM 시도
        logger.warning("%s not found in transformers; falling back to AutoModelForCausalLM(text_config)", text_cls_name)
        from transformers import AutoModelForCausalLM as TextCls  # type: ignore

    # language_model 서브트리 경로를 찾습니다.
    lm_prefix = next((n + "." for n, _ in merged.named_modules() if n.endswith("language_model")), None)
    if lm_prefix is None:
        raise RuntimeError("language_model submodule not found in merged multimodal model")

    # language 서브트리 가중치만 추출합니다.
    msd = merged.state_dict()
    text_sd = {}
    for k, v in msd.items():
        if k.startswith(lm_prefix):
            text_sd["model." + k[len(lm_prefix):]] = v
        elif k.startswith("lm_head."):
            text_sd[k] = v

    # meta 모델에 기존 텐서를 할당해 추가 사본을 만들지 않습니다.
    import gc
    from accelerate import init_empty_weights
    with init_empty_weights():
        text_model = TextCls(text_cfg)
    # assign=True로 meta 텐서를 실제 텐서로 교체합니다.
    missing, unexpected = text_model.load_state_dict(text_sd, strict=False, assign=True)
    if missing or unexpected:
        logger.warning("re-export key mismatch: missing=%d unexpected=%d", len(missing), len(unexpected))
        for k in list(missing)[:5]:
            logger.warning("  missing: %s", k)
    text_model = text_model.to(torch.bfloat16)

    # 모델 객체에 없는 KV 공유 텐서는 명시적 state_dict에 추가합니다.
    save_sd = text_model.state_dict()
    if revive_kv_shared:
        n = _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger)
        if n:
            save_sd = {k: v.to(torch.bfloat16) if hasattr(v, "to") else v for k, v in save_sd.items()}
    text_model.save_pretrained(output_dir, safe_serialization=True, state_dict=save_sd)
    tokenizer.save_pretrained(output_dir)
    del text_model, text_sd, msd, save_sd
    gc.collect()
    logger.info("Re-exported TEXT-ONLY model (%s, model_type=%s) to serving root: %s",
                text_cls_name, getattr(text_cfg, "model_type", "?"), output_dir)


def resolve_train_path(args) -> str:
    if args.train_file:
        return args.train_file
    ch = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    # 채널 디렉토리에서 첫 .jsonl
    for fn in sorted(os.listdir(ch)):
        if fn.endswith(".jsonl"):
            return os.path.join(ch, fn)
    raise FileNotFoundError(f"No training file (.jsonl) found in: {ch}")


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

    # 무거운 import는 실행 시점에 (dry-run 인자 파싱 실패 시 빠른 종료)
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # 멀티모달 base는 로더, LoRA 대상, 저장 경로가 다릅니다.
    #    - config에 vision_config/audio_config 또는 text_config가 있으면 멀티모달로 취급.
    #    - 로드: AutoModelForCausalLM은 멀티모달 arch(*ForConditionalGeneration)를 반환하므로,
    #      멀티모달이면 AutoModelForImageTextToText로 명시 로드(전체 모델). 텍스트 전용이면 CausalLM.
    _cfg = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    _text_cfg = getattr(_cfg, "text_config", None)
    is_multimodal = _text_cfg is not None or hasattr(_cfg, "vision_config")
    logger.info("model_type=%s arch=%s multimodal=%s", _cfg.model_type,
                (_cfg.architectures or ["?"])[0], is_multimodal)

    # dry-run에서는 소량 데이터와 짧은 시퀀스를 사용합니다.
    if args.dry_run:
        args.epochs = 1
        args.max_seq_length = min(args.max_seq_length, 512)
        logger.info("DRY-RUN: epochs=1, max_seq_length<=512, up to 32 rows")

    # ---- 데이터 로드 (conversational: {"messages":[...]}) ----
    train_path = resolve_train_path(args)
    ds = load_dataset("json", data_files=train_path, split="train")
    total = len(ds)
    if args.dry_run:
        ds = ds.select(range(min(32, len(ds))))          # dry-run: 32건 고정
    elif args.max_train_samples and args.max_train_samples > 0:
        # 시간/비용 조절: 앞 N건만 사용(데이터 파일은 그대로). 예: --max_train_samples 100
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
        logger.info("max_train_samples=%d: using %d of %d rows", args.max_train_samples, len(ds), total)
    logger.info("Training examples: %d  (file: %s)", len(ds), train_path)

    # ---- 토크나이저 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 모델 (bf16 필수, QLoRA면 4bit) ----
    # transformers 5에서는 dtype 인자를 사용합니다.
    model_kwargs = dict(
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        token=hf_token,
    )
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # 멀티모달 base는 ImageTextToText 로더로 전체(언어+vision/audio) 로드. 텍스트 전용이면 CausalLM.
    if is_multimodal:
        model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # ---- LoRA target 결정 ----
    # 멀티모달 모델은 지원되지 않는 vision/audio 선형층을 피하도록 language_model 경로만 선택합니다.
    if is_multimodal:
        lora_targets = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        # 멀티모달에서 embed/lm_head를 modules_to_save로 두면 vision 임베딩까지 얽힐 수 있어 생략(순수 텍스트 LoRA).
        modules_to_save = None
    else:
        lora_targets = "all-linear"
        modules_to_save = ["lm_head", "embed_tokens"]  # 텍스트 전용: 특수토큰 학습
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
        modules_to_save=modules_to_save,
    )

    # ---- packing 안전장치 ----
    # 샘플 간 교차 오염을 막기 위해 flash attention에서만 packing을 사용합니다.
    use_packing = args.packing and args.attn_implementation in (
        "flash_attention_2", "flash_attention_3",
    )
    if args.packing and not use_packing:
        logger.warning("packing disabled for attn_implementation='%s' (prevents cross-contamination); "
                       "packing is auto-enabled only with flash_attention_2.", args.attn_implementation)

    # MLflow 연결 결과를 SFTConfig의 report_to에 반영합니다.
    mlf = _Mlflow()

    # ---- SFTConfig ----
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        packing=use_packing,
        bf16=True,                      # Gemma는 fp16에서 NaN이 발생할 수 있습니다.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5 if args.dry_run else 10,
        save_strategy="no" if args.dry_run else "epoch",
        # 업로드 아티팩트가 불필요하게 커지지 않도록 체크포인트 하나만 유지합니다.
        save_total_limit=1,
        report_to=mlf.report_to,
        # SageMaker 학습 Job 이름으로 MLflow run과 CloudWatch 로그를 연결합니다.
        run_name=mlf.run_name or None,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # ---- Trainer (SFTTrainer가 messages에 chat template 자동 적용) ----
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    # 컨텍스트 종료 시 남아 있는 자식 run을 정리합니다.
    with mlf:
        trainer.train()

    # ---- 저장 ----
    # 서빙 루트에는 config.json과 가중치를 포함한 완전한 HF 모델을 저장합니다.
    # 멀티모달 base는 language 서브모듈만 텍스트 모델로 다시 저장합니다.
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Adapter saved (merge source): %s", adapter_dir)

        logger.info("Merging LoRA adapter into base model...")
        from peft import PeftModel
        # 병합 전에 학습 모델을 해제해 호스트 메모리를 확보합니다.
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 병합용 base는 CPU에 bf16으로 로드합니다.
        _loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        base = _loader.from_pretrained(
            args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation, token=hf_token, device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        del base
        gc.collect()

        if is_multimodal:
            # 멀티모달 머지 모델을 텍스트 전용 모델로 다시 저장합니다.
            # model_id/hf_token: KV-shared dead weight를 base에서 복원하는 데 필요(E2B/E4B).
            _reexport_text_only(merged, _cfg, tokenizer, args.output_dir, logger,
                                model_id=args.model_id, hf_token=hf_token)
        else:
            # 텍스트 전용 base: 머지 모델을 그대로 루트에 저장.
            merged.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            logger.info("Merged text model saved to serving root: %s", args.output_dir)
        # gemma-3/2/3n을 사용하면 해당 Gemma Terms를 준수해야 합니다.
    else:
        # 머지 안 함(또는 dry-run): 어댑터를 루트에 저장.
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Adapter saved (no merge): %s", args.output_dir)

    # 서빙에 쓰이지 않는 파일은 업로드 전에 제거합니다.
    if not args.dry_run:
        _prune_artifact(args.output_dir, logger)

    if args.dry_run:
        logger.info("DRY-RUN complete: pipeline OK. Run without --dry_run for real training.")


if __name__ == "__main__":
    main()
