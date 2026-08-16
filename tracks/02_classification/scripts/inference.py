"""Hugging Face PyTorch Inference DLC용 생성 핸들러입니다.

messages 또는 렌더링된 입력을 받아 `transformers.generate`로 응답합니다.
이 경로는 완성된 응답을 한 번에 반환하며 토큰 스트리밍을 지원하지 않습니다.
"""
from __future__ import annotations

import json
import os


def model_fn(model_dir, context=None):
    """엔드포인트 기동 시 모델과 토크나이저를 로드합니다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 우선, GPU 메모리 작으면 4bit(bitsandbytes 있을 때)로 폴백.
    load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="eager")
    if os.environ.get("LOAD_IN_4BIT", "0") in ("1", "true", "True"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.eval()
    return {"model": model, "tokenizer": tok}


def input_fn(request_body, request_content_type="application/json"):
    """요청 파싱: chat messages 또는 raw inputs 둘 다 허용."""
    if request_content_type and "json" in request_content_type:
        return json.loads(request_body)
    return {"inputs": request_body}


def predict_fn(data, model_bundle):
    """생성. messages가 오면 chat template 적용, inputs가 오면 그대로 사용."""
    import torch

    model = model_bundle["model"]
    tok = model_bundle["tokenizer"]
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    max_new = int(data.get("max_new_tokens", params.get("max_new_tokens", 256)))
    temperature = float(data.get("temperature", params.get("temperature", 0.2)))

    if isinstance(data, dict) and data.get("messages"):
        enc = tok.apply_chat_template(
            data["messages"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(model.device)
        input_ids = enc["input_ids"]
        gen_inputs = enc
    else:
        prompt = data.get("inputs", "") if isinstance(data, dict) else str(data)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        gen_inputs = enc

    do_sample = temperature and temperature > 0
    with torch.no_grad():
        out = model.generate(**gen_inputs, max_new_tokens=max_new,
                             do_sample=do_sample,
                             temperature=temperature if do_sample else None)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return {"generated_text": text}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction, ensure_ascii=False), "application/json"
