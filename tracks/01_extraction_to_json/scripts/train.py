#!/usr/bin/env python
"""
train.py — Gemma SFT+LoRA/QLoRA 학습 스크립트 (self-contained)

🔴 이 파일은 common/ 에 의존하지 않는다 (SageMaker는 source_dir만 컨테이너에 올리므로).
   로컬 GPU dry-run 과 SageMaker HuggingFace estimator .fit() 양쪽에서 동일하게 실행.

로컬 dry-run (개발환경 GPU에서 파이프라인 검증):
    python train.py --dry_run \
        --model_id google/gemma-4-E4B-it \
        --train_file ./sample.jsonl \
        --output_dir ./out

SageMaker (HuggingFace estimator entry_point):
    hyperparameters={"model_id": "...", "epochs": 3, "use_qlora": True, ...}
    채널: SM_CHANNEL_TRAIN(=/opt/ml/input/data/train), 모델은 SM_MODEL_DIR(=/opt/ml/model)로.

근거 (정찰 2026-07 검증):
  - Gemma chat template은 -it 토크나이저에 내장 → TRL SFTTrainer가 conversational
    ('messages') 데이터셋에 자동 적용. 수동 마커 조립 금지.
  - LoRA: r=16/alpha=16/dropout=0.05, target_modules='all-linear',
    modules_to_save=['lm_head','embed_tokens'] (특수토큰 학습).
  - bf16 필수 (fp16는 Gemma에서 오버플로/NaN). gradient_checkpointing(use_reentrant=False).
  - attn_implementation='eager' 가 Gemma 안전 기본 (soft-cap/sliding-window 정합성).
  - gated 모델(gemma-3/2/3n)은 HF_TOKEN env 필요. gemma-4 계열은 불필요(apache-2.0/ungated).

🔴 멀티모달 base(gemma-4 전부 · gemma-3 4b+) 텍스트 SFT (실측 검증 2026-07-21):
  - 로드: AutoModelForImageTextToText(멀티모달 전체). LoRA는 language_model 한정 target_modules.
  - 저장: 머지 후 language 서브모듈만 텍스트 arch(*ForCausalLM, model_type=*_text)로 re-export
    → vLLM이 순수 텍스트 경로로 로드(image/audio processor 불필요). 안 그러면 서빙 시
    "Can't load image processor" 로 죽는다. (_reexport_text_only 참고)
"""
from __future__ import annotations

import argparse
import logging
import os

# 🔴 SageMaker training job으로 이식되므로 print 대신 logging.
#    이 파일은 self-contained 진입점(common 미의존)이라 자체 로거를 쓴다. 단, 로깅 '구성'은
#    import 시점이 아니라 main()에서 1회 수행한다(import 부작용 회피 = 라이브러리 위생).
logger = logging.getLogger("gemma_train")


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
    boolean도 `--use_qlora True` 형태로 전달된다. store_true는 값을 안 받아 크래시 → str2bool 사용."""
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
    # boolean 플래그: nargs="?"+const=True → 로컬 bare-flag(`--dry_run`)와 SageMaker `--use_qlora True` 모두 지원
    p.add_argument("--packing", type=_str2bool, nargs="?", const=True, default=True)
    # LoRA / QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", type=_str2bool, nargs="?", const=True, default=False, help="4bit nf4 양자화(작은 GPU)")
    p.add_argument("--merge_adapter", type=_str2bool, nargs="?", const=True, default=True, help="학습 후 LoRA를 base에 머지(서빙용)")
    # attention
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    # 학습 샘플 수 제한 (시간/비용 조절) — 데이터는 그대로 두고 앞 N건만 사용. 0/None이면 전체.
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="학습에 쓸 최대 샘플 수(앞에서부터). 미지정 시 train.jsonl 전체 사용.")
    # dry-run
    p.add_argument("--dry_run", type=_str2bool, nargs="?", const=True, default=False, help="로컬 소량·짧은 학습으로 파이프라인 검증")
    return p.parse_args()


def _revive_kv_shared_from_base(save_sd, text_cfg, model_id, hf_token, logger) -> int:
    """KV-shared 레이어의 k_norm/k_proj/v_proj를 base 체크포인트에서 복원해 save_sd에 채운다.

    🔴 왜 필요한가 (E2B/E4B에서만 발생. 12B/26B/31B는 num_kv_shared_layers=0이라 무관):
       gemma-4 E계열은 뒤쪽 `num_kv_shared_layers`개 레이어가 앞 레이어의 KV를 재사용한다.
       transformers는 그 레이어에 k_norm/k_proj/v_proj 모듈을 아예 만들지 않는다
       (modeling_gemma4.py: "Layers sharing kv states don't need any weight matrices").
       그래서 save_pretrained를 거치면 원본에 있던 그 텐서들이 **소실**된다.
       반면 vLLM(Gemma4Attention)은 k_norm을 전 레이어에 등록하므로 로드 시
       `ValueError: Following weights were not initialized from checkpoint: layers.24~41...k_norm`
       으로 엔진 초기화가 실패한다 → vLLM/SGLang 서빙 불가.
       실측(E4B, 42층 중 shared 18층): 소실 키 정확히 54개
       = 레이어 24~41 × (k_norm.weight, k_proj.weight, v_proj.weight).

    이 텐서는 **연산에 쓰이지 않는다**(shared 레이어는 forward에서 앞 레이어의 KV를 그대로 재사용:
    modeling_gemma4.py `if self.is_kv_shared_layer: key_states, value_states = shared_kv_states[...]`).
    LoRA(q/k/v/o_proj 타깃)도 그 레이어엔 모듈이 없어 학습되지 않는다. 따라서 base 값을 그대로
    되살리는 것은 정확도에 무해하며, vLLM의 weight 검증만 통과시키는 목적이다.

    참고: vLLM issue #44788 — "E4B는 vLLM 불가"가 아니라 "transformers가 저장한 E4B 체크포인트가
    vLLM 불가"다. 원본 google/gemma-4-E4B-it 체크포인트는 이 54개 텐서를 모두 갖고 있어 그대로 뜬다.

    Returns: 되살린 텐서 개수(0이면 해당 없음 — 12B+, model_id 미지정, 또는 이미 온전).
    """
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0) or 0)
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    if n_shared <= 0 or n_layers <= 0:
        return 0  # 12B/26B/31B 등: KV 공유 없음 → 소실도 없음
    # model_id는 문자열 또는 후보 목록(앞에서부터 시도). 로컬 디렉터리와 HF repo id 모두 가능.
    sources = [s for s in ([model_id] if isinstance(model_id, str) else list(model_id or [])) if s]
    if not sources:
        logger.warning("KV-shared 복원 생략(model_id 미지정) — vLLM 서빙 시 weight 검증 실패 가능")
        return 0

    first = n_layers - n_shared  # E4B: 42-18=24 → 레이어 24~41이 shared
    want = [f"model.layers.{i}.self_attn.{n}.weight"
            for i in range(first, n_layers)
            for n in ("k_norm", "k_proj", "v_proj")]
    need = [k for k in want if k not in save_sd]
    if not need:
        logger.info("KV-shared 텐서 이미 온전(%d개 확인) — 복원 불필요", len(want))
        return 0

    # base 체크포인트에서 해당 텐서만 골라 읽는다(전체 로드 아님 → RAM 절약).
    # base는 멀티모달 arch이므로 키가 model.language_model.layers.N... 형태다 → 텍스트 키로 매핑.
    try:
        from safetensors import safe_open
        from transformers.utils import cached_file
    except ImportError as e:
        logger.warning("KV-shared 복원 실패(safetensors/transformers 유틸 없음: %s)", e)
        return 0

    # 텍스트 키 → base 키 후보(멀티모달 접두사 유무 양쪽)
    def _base_keys(tk):
        suffix = tk[len("model."):]                       # layers.N.self_attn.k_norm.weight
        return (f"model.language_model.{suffix}", f"language_model.{suffix}", tk)

    revived = 0
    remaining = set(need)
    for src in sources:
        if not remaining:
            break
        try:  # 샤딩된 경우: 인덱스에서 샤드 목록. 단일 파일이면 예외 → 폴백.
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
                continue  # 이 소스에 없음 → 다음 샤드/다음 소스로
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
        logger.warning("KV-shared 복원 불완전: %d/%d개 복원, %d개 실패(예: %s)",
                       revived, len(need), len(remaining), sorted(remaining)[:3])
    else:
        logger.info("KV-shared 텐서 %d개 복원 완료(레이어 %d~%d) → vLLM/SGLang 서빙 가능",
                    revived, first, n_layers - 1)
    return revived


def _reexport_text_only(merged, full_cfg, tokenizer, output_dir, logger,
                        model_id=None, hf_token=None, revive_kv_shared=True) -> None:
    """멀티모달 머지 모델에서 language 서브모듈만 텍스트 arch(*ForCausalLM)로 re-export.

    🔴 왜: 멀티모달 config(*ForConditionalGeneration)로 저장하면 vLLM이 image/audio processor를
       찾다가 죽는다. text_config(model_type=*_text) + language_model 가중치만 저장하면 vLLM이
       순수 텍스트 경로로 로드(vision/audio tower·processor 불필요).
    실측(gemma-4 E4B/12B/26B): model.language_model.* → model.* 재키잉 + lm_head, 키 100% 매칭.
    text 클래스는 arch에 'Unified'가 있으면 Gemma4UnifiedForCausalLM, 아니면 Gemma4ForCausalLM.

    🔴 revive_kv_shared: E2B/E4B(num_kv_shared_layers>0)에서 transformers가 저장하지 않는
       KV-shared 레이어의 k_norm/k_proj/v_proj를 base 체크포인트에서 복원해 함께 저장한다.
       이게 없으면 vLLM/SGLang이 "weights not initialized" 로 엔진 초기화에 실패한다
       (상세: _revive_kv_shared_from_base 독스트링, vLLM issue #44788).
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

    # language_model 서브트리 경로 자동 탐색 (…language_model.)
    lm_prefix = next((n + "." for n, _ in merged.named_modules() if n.endswith("language_model")), None)
    if lm_prefix is None:
        raise RuntimeError("language_model submodule not found in merged multimodal model")

    # language 서브트리 가중치만 추출(텐서는 merged와 공유 — 사본 아님).
    msd = merged.state_dict()
    text_sd = {}
    for k, v in msd.items():
        if k.startswith(lm_prefix):
            text_sd["model." + k[len(lm_prefix):]] = v
        elif k.startswith("lm_head."):
            text_sd[k] = v

    # 🔴 OOM 방지: 빈 뼈대를 meta로 만들어(가중치 미할당) 사본을 안 만들고, load_state_dict(assign=True)로
    #    merged의 텐서를 그대로 이식한다. TextCls(text_cfg)를 그냥 만들면 fp32 사본(+.to(bf16) 또 사본)이 생긴다.
    import gc
    from accelerate import init_empty_weights
    with init_empty_weights():
        text_model = TextCls(text_cfg)
    # assign=True: 새 파라미터를 할당(빈 meta 텐서를 text_sd의 실제 텐서로 대체) — 추가 사본 없음.
    missing, unexpected = text_model.load_state_dict(text_sd, strict=False, assign=True)
    if missing or unexpected:
        logger.warning("re-export key mismatch: missing=%d unexpected=%d", len(missing), len(unexpected))
        for k in list(missing)[:5]:
            logger.warning("  missing: %s", k)
    text_model = text_model.to(torch.bfloat16)

    # 🔴 저장 직전에 KV-shared 레이어의 dead weight를 base에서 복원해 명시 state_dict로 넘긴다.
    #    save_pretrained(state_dict=...)를 쓰는 이유: text_model.state_dict()에는 그 모듈이 아예
    #    없어(생성되지 않음) 모델을 고치는 방법으로는 채울 수 없다.
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
    """서빙에 쓰이지 않는 것을 지운다. /opt/ml/model 전체가 tar.gz 로 S3 에 올라가기 때문이다.

    🔴 실측: E4B 학습 산출물이 11.37GB 였고 업로드에 3분 14초가 걸렸다. 그 안에는
       서빙이 절대 열지 않는 것들이 함께 들어 있었다.
         checkpoint-*/  — optimizer.pt / rng_state.pth / scheduler.pt. 학습 재개용이고,
                          이 kit 은 재개를 지원하지 않는다(save_total_limit=1 로 1개만 남겨도
                          그 1개가 수 GB 다).
         adapter/       — 머지 소스. 머지된 모델이 이미 루트에 있으므로 중복이다.
       지우면 업로드 시간과 S3 요금이 함께 줄고, 배포 시 컨테이너가 받는 양도 줄어든다.
       ⚠️ 학습을 이어서 하거나 adapter 만 따로 배포할 계획이면 이 함수를 부르지 말 것.
    """
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
            logger.info("Pruned %s (%.2f GB) — 서빙에 쓰이지 않습니다", name, size / 1024**3)
    if removed:
        logger.info("아티팩트에서 %.2f GB 제거 — 업로드 시간과 S3 요금이 그만큼 줄어듭니다",
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

    # 🔴 멀티모달 base 감지 (Gemma 4 전부 · Gemma 3 4b+). 텍스트 SFT라도 로더/타깃/저장이 달라진다.
    #    - config에 vision_config/audio_config 또는 text_config가 있으면 멀티모달로 취급.
    #    - 로드: AutoModelForCausalLM은 멀티모달 arch(*ForConditionalGeneration)를 반환하므로,
    #      멀티모달이면 AutoModelForImageTextToText로 명시 로드(전체 모델). 텍스트 전용이면 CausalLM.
    _cfg = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    _text_cfg = getattr(_cfg, "text_config", None)
    is_multimodal = _text_cfg is not None or hasattr(_cfg, "vision_config")
    logger.info("model_type=%s arch=%s multimodal=%s", _cfg.model_type,
                (_cfg.architectures or ["?"])[0], is_multimodal)

    # dry-run 오버라이드: 소량·1 step 수준으로
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
        logger.info("max_train_samples=%d → using %d of %d rows", args.max_train_samples, len(ds), total)
    logger.info("Training examples: %d  (file: %s)", len(ds), train_path)

    # ---- 토크나이저 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 모델 (bf16 필수, QLoRA면 4bit) ----
    # transformers 5.x: torch_dtype → dtype 로 이름 변경(구 이름은 deprecation 경고).
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
    # 🔴 멀티모달 gemma-4 실측(2026-07): language proj(q/k/v/o/gate/up/down_proj)는 평범한 nn.Linear지만,
    #    vision/audio tower의 동명 proj는 커스텀 `Gemma4ClippableLinear`라 peft가 지원 안 함
    #    (ValueError: Target module ... is not supported). 따라서 target_modules에 이름 리스트나
    #    'all-linear'를 주면 vision/audio proj까지 매칭돼 크래시하거나 불필요 파라미터가 붙는다.
    #    → **정규식으로 language_model 경로 한정**: language의 258개 nn.Linear만 매칭(ClippableLinear 0).
    #    실측: get_peft_model OK, lora_A 516개 부착. 텍스트 서빙은 이후 language 서브모듈만 re-export.
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
    # 🔴 packing은 여러 샘플을 한 시퀀스로 합치는데, flash-attention이 아니면 샘플 간
    #    cross-contamination(교차 오염) 위험이 있다(TRL 경고). Gemma 안전 기본인 eager/sdpa에서는
    #    packing을 끈다. flash_attention_2 일 때만 packing 허용.
    use_packing = args.packing and args.attn_implementation in (
        "flash_attention_2", "flash_attention_3",
    )
    if args.packing and not use_packing:
        logger.warning("packing disabled for attn_implementation='%s' (prevents cross-contamination); "
                       "packing is auto-enabled only with flash_attention_2.", args.attn_implementation)

    # ---- SFTConfig ----
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        packing=use_packing,
        bf16=True,                      # 🔴 fp16 금지 (Gemma NaN)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5 if args.dry_run else 10,
        save_strategy="no" if args.dry_run else "epoch",
        # 🔴 중간 체크포인트는 1개만 유지한다. 이 디렉토리(=/opt/ml/model)는 그대로 model.tar.gz로
        #    업로드되므로, epoch마다 쌓이면 아티팩트가 커져 업로드가 길어진다(실측: checkpoint 3개
        #    = 0.7GB, 서빙엔 불필요 — 서빙은 머지된 루트만 읽는다). 업로드 시간도 MaxRuntime에 포함된다.
        save_total_limit=1,
        report_to="none",
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
    trainer.train()

    # ---- 저장 ----
    # 🔴 서빙 루트(SM_MODEL_DIR=/opt/ml/model)에는 '완전한 HF 모델'(config.json + 가중치)이 와야 한다.
    #    vLLM/LMI는 HF_MODEL_ID=/opt/ml/model 루트의 config.json으로 엔진을 감지한다.
    #    ⚠️ 멀티모달 base를 텍스트로 서빙할 때: config가 멀티모달(*ForConditionalGeneration)로 남으면
    #       vLLM이 image/audio processor를 찾다가 "Can't load image processor"로 죽는다(실측). 그래서
    #       머지 후 **language 서브모듈만 텍스트 arch(*ForCausalLM)로 re-export**한다(gemma4→gemma4_text).
    #       실측 검증(E4B/12B/26B): model.language_model.* → model.* 재키잉 + lm_head, 키 100% 매칭.
    if args.merge_adapter and not args.dry_run:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Adapter saved (merge source): %s", adapter_dir)

        logger.info("Merging LoRA adapter into base model...")
        from peft import PeftModel
        # 🔴 OOM 방지: merge는 학습 모델 + base(bf16 full) + merged + (re-export시) text_model 사본이
        #    호스트 RAM에 겹쳐 쌓인다(8B면 각 ~16GB). 학습이 끝난 trainer/model을 먼저 해제해 RAM을 회수한다.
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # base는 CPU에 bf16로 로드(GPU 불필요 — merge/save는 CPU에서). low_cpu_mem_usage로 사본 최소화.
        _loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        base = _loader.from_pretrained(
            args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation, token=hf_token, device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        del base
        gc.collect()

        if is_multimodal:
            # 멀티모달 머지 모델 → 텍스트 전용으로 re-export (vLLM 텍스트 서빙용).
            # model_id/hf_token: KV-shared dead weight를 base에서 복원하는 데 필요(E2B/E4B).
            _reexport_text_only(merged, _cfg, tokenizer, args.output_dir, logger,
                                model_id=args.model_id, hf_token=hf_token)
        else:
            # 텍스트 전용 base: 머지 모델을 그대로 루트에 저장.
            merged.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            logger.info("Merged text model saved to serving root: %s", args.output_dir)
        # ⚠️ Gemma 라이선스: gemma-4=apache-2.0(제약 없음), gemma-3/2/3n=Gemma Terms 준수.
    else:
        # 머지 안 함(또는 dry-run): 어댑터를 루트에 저장.
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Adapter saved (no merge): %s", args.output_dir)

    # 🔴 /opt/ml/model 전체가 tar.gz 로 올라간다. 서빙이 열지 않는 것을 먼저 지운다.
    if not args.dry_run:
        _prune_artifact(args.output_dir, logger)

    if args.dry_run:
        logger.info("DRY-RUN complete — pipeline OK. Run without --dry_run for real training.")


if __name__ == "__main__":
    main()
