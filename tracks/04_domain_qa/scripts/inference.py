"""
inference.py — HF PyTorch Inference DLC용 커스텀 핸들러 (transformers.generate 서빙).

🔴 왜 이 핸들러가 필요한가:
  gemma-4 E2B/E4B는 num_kv_shared_layers>0(KV-sharing)이라 vLLM/TGI가 로드 실패
  (vLLM 이슈 #44788: KV-shared 레이어 k_norm 미초기화, 버전 무관). transformers는 정상 로드/생성.
  → HF PyTorch Inference DLC(transformers>=5.5.3, gemma4 포함) 위에서 이 핸들러가
    AutoModelForCausalLM.generate로 직접 서빙한다(연속배칭 없는 단건 추론).

배치: 이 파일을 model.tar.gz 안의 code/inference.py 로 넣으면 SageMaker HF Inference Toolkit이
  자동 인식한다. (배포 노트북이 model_data에 code/를 주입 — 03 노트북 참고.)
계약(SageMaker HuggingFace Inference Toolkit): model_fn / input_fn / predict_fn / output_fn.
호출: sagemaker-runtime invoke_endpoint, ContentType/Accept = application/json.
  요청 예: {"messages":[{"role":"user","content":"..."}], "max_new_tokens":256, "temperature":0.2}
       또는 {"inputs":"<이미 렌더된 프롬프트>", "parameters":{...}}
응답: {"generated_text": "..."}

🔴 토큰 스트리밍은 이 DLC에선 불가(재시도 금지):
  HF Inference Toolkit의 handler_service.handle()가 transform_fn 결과를 `return [response]`로
  '완성본 한 번에' 버퍼링해 반환한다(소스 확인, 2026-07). 여기에 TextIteratorStreamer를 넣어도
  invoke_endpoint_with_response_stream은 단일 청크만 받는다. E4B 스트리밍이 필요하면 12B+로
  vLLM DLC 배포(native 스트리밍) 또는 커스텀 컨테이너(FastAPI+SSE) 필요. 배경: docs/05_serving_containers.md §4.6.
"""
from __future__ import annotations

import json
import os


def model_fn(model_dir, context=None):
    """엔드포인트 기동 시 1회: 모델+토크나이저 로드. gemma-4 KV-sharing은 transformers가 처리."""
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
