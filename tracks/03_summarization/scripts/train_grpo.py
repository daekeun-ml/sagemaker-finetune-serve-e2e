#!/usr/bin/env python
"""Gemma GRPO와 LoRA/QLoRA 학습 스크립트입니다.

SFT와의 차이:
  - SFT는 정답 completion을 '모방'. GRPO는 prompt당 여러 개를 생성해 **reward 함수**로 좋은 걸 강화.
  - 따라서 데이터는 {"prompt":[...user...], "reference":"<정답>"} 형태(정답은 reward 계산용).
  - 프로그램으로 reward를 계산할 수 있는 추출과 분류 트랙만 지원.
  - prompt당 여러 rollout을 생성하므로 SFT보다 연산량이 큼.

멀티모달 base 처리는 train.py와 동일:
  - AutoModelForImageTextToText로 로드, LoRA는 language_model 한정(regex), 머지 후 텍스트 re-export.

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
    # SFT 산출물을 우선 사용하고 없으면 HF base를 사용합니다.
    #   1) --base_model_dir (SFT 아티팩트를 마운트한 컨테이너 경로, 예 SM_CHANNEL_MODEL=/opt/ml/input/data/model)
    #   2) --model_id (SFT 없이 base에서 바로 GRPO할 때)
    p.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"))
    p.add_argument("--base_model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL"),
                   help="SFT 산출물 디렉토리. 있으면 GRPO의 base로 사용합니다.")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./out"))
    # 프로그램으로 계산할 reward 종류입니다.
    p.add_argument("--reward_kind", type=str, default="extraction", choices=["extraction", "classification"])
    # GRPO 하이퍼
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)   # GRPO는 SFT보다 낮게
    p.add_argument("--num_generations", type=int, default=8)      # prompt당 생성 수
    # TRL v1.6.0에서 GRPOConfig의 max_prompt_length가 제거되었습니다. 인자는 하위호환을 위해
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
# 프로그램형 reward 함수
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
            out.append(0.3)          # 라벨이 포함됐지만 형식이 다른 경우
        else:
            out.append(0.0)
    return out


REWARDS = {"extraction": reward_extraction, "classification": reward_classification}


# ---------------------------------------------------------------------------
# 데이터를 prompt와 reference 형식으로 변환합니다.
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
    """KV 공유 레이어의 서빙 검증용 텐서를 base에서 복원합니다."""
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
        logger.warning("KV-shared restore failed (safetensors missing: %s)", e)
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
        logger.warning("KV-shared restore incomplete: %d/%d (failure example: %s)",
                       revived, len(need), sorted(remaining)[:3])
    else:
        logger.info("restored %d KV-shared tensors (layers %d-%d); vLLM/SGLang can serve this",
                    revived, first, n_layers - 1)
    return revived


def _reexport_text_only(merged, full_cfg, tokenizer, output_dir, logger,
                        model_id=None, hf_token=None):
    """멀티모달 머지 모델의 language 서브모듈을 텍스트 모델로 다시 저장합니다.

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
    # meta 모델에 기존 텐서를 할당해 추가 사본을 만들지 않습니다.
    import gc
    from accelerate import init_empty_weights
    with init_empty_weights():
        text_model = TextCls(text_cfg)
    text_model.load_state_dict(text_sd, strict=False, assign=True)
    text_model = text_model.to(torch.bfloat16)
    # 모델 객체에 없는 KV 공유 텐서는 명시적 state_dict에 추가합니다.
    save_sd = text_model.state_dict()
    if _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger):
        save_sd = {k: (v.to(torch.bfloat16) if hasattr(v, "to") else v) for k, v in save_sd.items()}
    text_model.save_pretrained(output_dir, safe_serialization=True, state_dict=save_sd)
    tokenizer.save_pretrained(output_dir)
    del text_model, text_sd, msd, save_sd
    gc.collect()
    logger.info("Re-exported TEXT-ONLY (%s) to %s", text_cls_name, output_dir)


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


def _resolve_sft_base(base_model_dir: str | None, logger) -> str | None:
    """SFT 채널에서 base 모델 디렉토리를 찾고 tar 아티팩트면 압축을 풉니다."""
    import tarfile

    if not base_model_dir or not os.path.isdir(base_model_dir):
        return None

    # 이미 풀린 모델 디렉토리를 우선 사용합니다.
    if os.path.isfile(os.path.join(base_model_dir, "config.json")):
        return base_model_dir

    # (2) tar.gz 가 있으면 푼다
    tars = [f for f in os.listdir(base_model_dir) if f.endswith((".tar.gz", ".tgz"))]
    if not tars:
        return None
    src = os.path.join(base_model_dir, tars[0])
    dest = os.path.join(base_model_dir, "_extracted")
    if not os.path.isfile(os.path.join(dest, "config.json")):
        os.makedirs(dest, exist_ok=True)
        logger.info("extracted SFT artifact: %s -> %s", src, dest)
        with tarfile.open(src, "r:gz") as tf:
            # filter="data"는 심볼릭 링크와 상위 경로 탈출을 막습니다. Python 3.12 이상에서
            # 지원하고 3.14 부터는 생략하면 경고가 뜬다. 우리가 만든 아티팩트라 위험은 없지만
            # 명시해 두면 컨테이너의 파이썬이 올라가도 그대로 돈다.
            try:
                tf.extractall(dest, filter="data")
            except TypeError:            # Python 3.11 이하에는 filter 인자가 없습니다.
                tf.extractall(dest)      # noqa: S202. 이 프로젝트가 만든 학습 산출물입니다.
    if os.path.isfile(os.path.join(dest, "config.json")):
        return dest
    logger.warning("extracted, but config.json is missing: %s", dest)
    return None


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
    sft_dir = _resolve_sft_base(args.base_model_dir, logger)
    if sft_dir:
        base_src = sft_dir
        logger.info("GRPO from SFT checkpoint: %s (standard RLHF order: SFT then GRPO)", base_src)
    elif args.base_model_dir:
        # SFT 채널을 읽지 못하면 HF base로 폴백하지 않고 중단합니다.
        raise RuntimeError(
            f"SFT 산출물을 읽을 수 없습니다: {args.base_model_dir}\n"
            f"  디렉터리 내용: {sorted(os.listdir(args.base_model_dir))[:10]}\n"
            "  model 채널에는 학습 Job 의 output/model.tar.gz 를 넘기세요.\n"
            "  SFT 없이 base 에서 GRPO 를 하려면 --base_model_dir '' 로 비우세요.")
    else:
        base_src = args.model_id
        logger.info("GRPO from HF base: %s (no SFT stage)", base_src)

    # ---- 멀티모달 감지 (train.py와 동일) ----
    # 텍스트로 다시 저장된 SFT 산출물은 CausalLM으로 로드합니다.
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
    # TRL 버전마다 GRPOConfig 필드가 달라 지원되는 값만 전달합니다.
    import dataclasses as _dc
    _supported = {f.name for f in _dc.fields(GRPOConfig)}
    # MLflow 연결 결과를 GRPOConfig의 report_to에 반영합니다.
    mlf = _Mlflow()

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
        # 업로드 아티팩트가 커지지 않도록 체크포인트 하나만 유지합니다.
        "save_total_limit": 1,
        "report_to": mlf.report_to,
        # SageMaker 학습 Job 이름으로 MLflow run과 CloudWatch 로그를 연결합니다.
        "run_name": mlf.run_name or None,
    }
    _dropped = sorted(set(_wanted) - _supported)
    if _dropped:
        logger.info("GRPOConfig: dropped args unsupported by this TRL version: %s", _dropped)
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

    # 컨텍스트 종료 시 남아 있는 자식 run을 정리합니다.
    with mlf:
        logger.info("Starting GRPO training (num_generations=%d)...", args.num_generations)
        trainer.train()

    # ---- 저장 (train.py와 동일: 멀티모달이면 텍스트 re-export) ----
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        from peft import PeftModel
        # 병합 전에 학습 모델을 해제하고 base를 CPU에 로드합니다.
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

    if not args.dry_run:
        _prune_artifact(args.output_dir, logger)

    if args.dry_run:
        logger.info("DRY-RUN complete: GRPO pipeline OK.")


if __name__ == "__main__":
    main()
