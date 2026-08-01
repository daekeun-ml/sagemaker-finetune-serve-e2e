"""
common/aws_utils.py — SageMaker endpoint 호출 · Bedrock Converse · CloudWatch 링크 · 비용 가드

🔴 서비스 경계 (혼동 금지 — verified-facts):
  - SageMaker endpoint 호출  = boto3 "sagemaker-runtime" 클라이언트, invoke_endpoint()
  - Bedrock Claude 호출       = boto3 "bedrock-runtime" 클라이언트, converse()
  이 둘은 별개 서비스·별개 클라이언트. "endpoint를 Bedrock API로 호출"은 잘못.
"""
from __future__ import annotations

import json
import os
from typing import Any

# 🔴 boto3는 함수 내부에서 lazy import 한다. 이 모듈의 일부 헬퍼(응답 파싱 등)와 이를 재사용하는
#    순수 로직 모듈(common/synth/bedrock_synth의 _extract_json/has_pii)이 boto3 없이도 import되도록.


# ---------------------------------------------------------------------------
# 1) SageMaker real-time endpoint 호출 (sagemaker-runtime)
# ---------------------------------------------------------------------------
def _parse_endpoint_response(body: Any) -> str:
    """서빙 컨테이너별 응답 스키마 방어적 파싱.

    - TGI/DJL generation:  {"generated_text": ...} 또는 [{"generated_text": ...}]
    - DJL LMI / vLLM OpenAI chat:  {"choices":[{"message":{"content": ...}}]}
    - OpenAI completion:  {"choices":[{"text": ...}]}
    """
    if isinstance(body, list) and body:
        # 🔴 HF Inference Toolkit은 output_fn의 (payload, content_type)을 [payload, "application/json"]
        #    리스트로 반환한다. 첫 요소만 취한다(둘째는 content-type).
        return _parse_endpoint_response(body[0])
    if isinstance(body, str):
        # 🔴 응답 본문이 JSON 문자열이면 파싱해 재귀 처리(예: '{"generated_text": "..."}').
        #    JSON이 아니면(순수 생성 텍스트) 그대로 반환.
        s = body.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return _parse_endpoint_response(json.loads(s))
            except (json.JSONDecodeError, ValueError):
                pass
        return body
    if isinstance(body, dict):
        # OpenAI 호환 (LMI messages / vLLM)
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                if "text" in c0:
                    return c0["text"]
        # generation 스키마
        if "generated_text" in body:
            gt = body["generated_text"]
            # 🔴 이중 래핑 방지: HF Inference Toolkit이 커스텀 output_fn의 JSON을 한 번 더 직렬화하면
            #    값이 다시 '{"generated_text": "..."}' 문자열이 된다. 그 경우 한 겹 더 벗긴다.
            if isinstance(gt, str):
                s = gt.strip()
                if s.startswith("{") and '"generated_text"' in s:
                    try:
                        return _parse_endpoint_response(json.loads(s))
                    except (json.JSONDecodeError, ValueError):
                        pass
            return gt
        if "outputs" in body:
            out = body["outputs"]
            return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return json.dumps(body, ensure_ascii=False)
    return str(body)


def invoke_sagemaker_endpoint(
    endpoint_name: str,
    prompt: str,
    region: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **gen_kwargs: Any,
) -> str:
    """SLM endpoint 호출 — generation 스키마 {"inputs","parameters"}.

    DJL LMI(vLLM)·HF TGI 모두 이 rolling-batch generation 스키마를 지원한다(응답 {"generated_text"}).
    OpenAI chat 스키마로 부르려면 invoke_sagemaker_chat() 사용.
    """
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            **gen_kwargs,
        },
    }
    resp = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return _parse_endpoint_response(json.loads(resp["Body"].read().decode("utf-8")))


def invoke_sagemaker_chat(
    endpoint_name: str,
    messages: list[dict[str, str]],
    region: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    **gen_kwargs: Any,
) -> str:
    """SLM endpoint 호출 — OpenAI 호환 chat 스키마 {"messages":[...]}.

    DJL LMI(vLLM)는 서버측에서 chat template을 적용하므로, 프롬프트를 직접 렌더링하지 않고
    messages를 그대로 보낼 수 있다(응답 {"choices":[{"message":{"content"}}]}).
    """
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **gen_kwargs,
    }
    resp = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return _parse_endpoint_response(json.loads(resp["Body"].read().decode("utf-8")))


def stream_sagemaker_chat(
    endpoint_name: str,
    messages: list[dict[str, str]],
    region: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    **gen_kwargs: Any,
):
    """SLM endpoint를 **토큰 스트리밍**으로 호출 — 생성되는 조각(delta)을 순서대로 yield.

    vLLM/SGLang/LMI는 OpenAI 호환 SSE(`stream: true`)를 지원하므로
    `invoke_endpoint_with_response_stream`으로 첫 토큰부터 받아볼 수 있다.
    실측(요약 트랙 endpoint, vLLM 0.26.0, ml.g6.2xlarge, 입력 5,996자):
      첫 응답 0.42s(391 조각) vs 완성 대기 16.16s → 첫 응답 체감 38배.
      완료 시각은 15.9s vs 16.2s로 사실상 같다(전체 생성 시간·throughput은 개선되지 않는다).

    🔴 청크 경계는 SSE 줄 경계와 **일치하지 않는다**(실측): PayloadPart 하나가 JSON 중간에서
       끊겨 `..."system_finger` / `print":"vllm..."` 로 나뉘어 온 사례를 확인했다.
       그래서 청크를 그때그때 파싱하면 JSONDecodeError가 난다 → 버퍼에 모아
       '\\n\\n'(SSE 이벤트 구분자) 단위로만 잘라 파싱한다.

    사용:
        for piece in stream_sagemaker_chat(ep, msgs, region=REGION):
            print(piece, end='', flush=True)
    """
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        **gen_kwargs,
    }
    resp = client.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="text/event-stream",
        Body=json.dumps(payload),
    )
    buf = b""
    for event in resp["Body"]:
        chunk = (event.get("PayloadPart") or {}).get("Bytes")
        if not chunk:
            continue
        buf += chunk
        while b"\n\n" in buf:                       # 완결된 SSE 이벤트만 처리
            raw, buf = buf.split(b"\n\n", 1)
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue                            # 깨진 이벤트는 건너뛴다(스트림 유지)
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield piece


# ---------------------------------------------------------------------------
# 2) Bedrock Claude 호출 (bedrock-runtime Converse)
# ---------------------------------------------------------------------------
def bedrock_converse(
    model_id: str,
    user_text: str,
    region: str,
    system_text: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Bedrock Claude를 Converse API로 호출. model_id는 inference-profile prefix 포함.

    🔴 model_id 하드코딩 금지 — config.BEDROCK_CLAUDE_MODEL_ID(env)에서 주입.
    ⚠️ sampling 파라미터는 모델 세대별로 제약이 다르다(자주 바뀜):
       - Claude 4.x: temperature/top_p 동시 지정 불가(둘 중 하나).
       - Claude 5+  : temperature 자체가 deprecated.
       그래서 기본은 **아무 sampling 파라미터도 보내지 않는다**(maxTokens만). 필요하면 temperature 또는
       top_p 중 하나만 명시. 지정 시에도 top_p를 우선하고, deprecated로 거부되면 조용히 제거 후 재시도한다.
    """
    import boto3
    from botocore.config import Config
    # 병렬 호출(max_workers 높음) 시 Bedrock throttling(429)에 대비해 adaptive 재시도.
    #   env BEDROCK_MAX_ATTEMPTS 로 조정(기본 8). adaptive 모드는 backoff+rate-limiting 자동.
    _boto_cfg = Config(retries={"max_attempts": int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "8")),
                                "mode": "adaptive"})
    client = boto3.client("bedrock-runtime", region_name=region, config=_boto_cfg)

    def _build_ic(use_temp: bool, use_topp: bool) -> dict[str, Any]:
        ic: dict[str, Any] = {"maxTokens": max_tokens}
        if use_topp and top_p is not None:
            ic["topP"] = top_p
        elif use_temp and temperature is not None:
            ic["temperature"] = temperature
        return ic

    def _text_of(resp: dict) -> str:
        """응답에서 text 블록만 뽑아 이어붙인다.

        🔴 `content[0]["text"]` 로 바로 꺼내면 안 된다(실측 2026-07-31):
           Claude Sonnet 5 는 **`reasoningContent` 블록을 먼저** 넣고 `text` 를 두 번째에 둘 때가 있다.
           그러면 `content[0]` 에 'text' 키가 없어 KeyError 로 죽는다.
           (증상: 합성이 조용히 0건 — 예외가 배치 단위로 삼켜져 원인이 안 보인다.)
        """
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        texts = [b["text"] for b in blocks if isinstance(b, dict) and "text" in b]
        if not texts:
            kinds = [k for b in blocks if isinstance(b, dict) for k in b]
            raise ValueError(f"Bedrock 응답에 text 블록이 없습니다(블록 종류: {kinds}).")
        return "\n".join(texts)

    def _call(ic: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": ic,
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        return _text_of(client.converse(**kwargs))

    ic = _build_ic(use_temp=True, use_topp=True)
    try:
        return _call(ic)
    except Exception as e:  # noqa: BLE001
        # 모델이 특정 sampling 파라미터를 거부하면(deprecated/동시지정 불가) maxTokens만으로 재시도
        msg = str(e).lower()
        if "temperature" in msg or "top_p" in msg or "topp" in msg:
            return _call({"maxTokens": max_tokens})
        raise


# ---------------------------------------------------------------------------
# 3) CloudWatch / SageMaker 콘솔 다이렉트 링크 (aws-handson-testing 규칙)
# ---------------------------------------------------------------------------
def cw_links(region: str, endpoint_name: str | None = None, training_job: str | None = None):
    """학습/추론 상황을 즉시 볼 수 있는 클릭 가능한 콘솔 링크(HTML) 출력.

    노트북에서 estimator.fit()·deploy()·invoke 직후 호출:
        from IPython.display import display
        display(cw_links(REGION, endpoint_name=EP))
    Logs 경로의 '/'는 콘솔 URL 규약상 이중 인코딩('$252F') 필요.
    """
    from IPython.display import HTML

    rows = []
    note = ""
    if training_job:
        tj_grp = "/aws/sagemaker/TrainingJobs".replace("/", "$252F")
        # 🔴 URL 구조 (실측 2026-07-31):
        #    로그 스트림 이름은 '<잡이름>/algo-N-<epoch>' 형태다(예: gemma-...-train-2026.../algo-1-1785409886).
        #    즉 **잡 이름만으로는 특정 스트림을 지목할 수 없다** → `/log-events`(단일 스트림 보기) 경로에
        #    잡 이름을 붙이면 화면이 어긋난다. 대신 **스트림 목록 화면 + 접두사 필터**로 보낸다:
        #      .../log-group/<그룹>$3FlogStreamNameFilter$3D<잡이름>
        #    (하위 경로의 '/'는 $252F, 쿼리 '?'는 $3F, '&'는 $26 로 이중 인코딩 — 콘솔 hash 규약)
        # ⚠️ 로그 그룹/스트림은 잡이 'Training' 단계에 진입해 첫 로그를 쓴 뒤에야 생성된다
        #    → 그 전(Starting/Pending/Downloading)엔 'log group does not exist'가 정상.
        rows.append(
            f'<li><a href="https://{region}.console.aws.amazon.com/sagemaker/home?region={region}'
            f'#/jobs/{training_job}" target="_blank">▶ SageMaker Training Job (상태·항상 존재): {training_job}</a></li>'
            f'<li><a href="https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}'
            f'#logsV2:log-groups/log-group/{tj_grp}$3FlogStreamNameFilter$3D{training_job}'
            f'" target="_blank">CloudWatch Logs — 이 잡의 스트림 목록 (Training 단계 진입 후 생성)</a></li>'
        )
        note = ("⚠️ 로그는 잡이 <b>Training</b> 단계에 들어간 뒤에 나타납니다. Starting/Pending(용량 대기)"
                "/Downloading 단계에선 로그 그룹이 아직 없어 'log group does not exist'가 정상입니다. "
                "상태는 위 SageMaker Training Job 링크에서 확인하세요.")
    if endpoint_name:
        ep_grp = "/aws/sagemaker/Endpoints".replace("/", "$252F")
        rows.append(
            f'<li><a href="https://{region}.console.aws.amazon.com/sagemaker/home?region={region}'
            f'#/endpoints/{endpoint_name}" target="_blank">▶ SageMaker Endpoint (상태): {endpoint_name}</a></li>'
            f'<li><a href="https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}'
            f'#logsV2:log-groups/log-group/{ep_grp}$252F{endpoint_name}" target="_blank">CloudWatch Logs (이 endpoint)</a></li>'
        )
    # ⚠️ 콘솔 URL 형식은 "현재 기준" — AWS가 바꾸면 갱신 필요.
    html = "<b>🔗 콘솔 바로가기</b><ul>" + "".join(rows) + "</ul>"
    if note:
        html += f"<div style='font-size:0.9em;color:#a60'>{note}</div>"
    return HTML(html)


# ---------------------------------------------------------------------------
# 3.5) S3 조건부 업로드 (내용 해시 비교 — 여러 번 실행해도 안전, 중복 업로드 방지)
# ---------------------------------------------------------------------------
def _md5(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_if_changed(local_path: str, bucket: str, key: str, region: str) -> str:
    """로컬 파일을 s3://bucket/key 에 업로드하되, **내용이 같으면 스킵**한다.

    노트북을 여러 번 돌려도 매번 재업로드하지 않도록, 로컬 MD5를 S3 객체 메타데이터(content-md5)와
    비교한다. 파일명이 같아도 내용이 바뀌면 새로 업로드하므로 stale 데이터 재사용 위험이 없다.
    반환: s3://bucket/key
    """
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=region)
    local_md5 = _md5(local_path)
    uri = f"s3://{bucket}/{key}"
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        remote_md5 = head.get("Metadata", {}).get("content-md5")
        if remote_md5 == local_md5:
            print(f"skip upload (unchanged): {uri}")
            return uri
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "403"):
            raise  # 진짜 에러(권한 등)는 전파
    s3.upload_file(local_path, bucket, key, ExtraArgs={"Metadata": {"content-md5": local_md5}})
    print(f"uploaded: {uri}")
    return uri


def training_job_status(training_job_name: str, region: str) -> dict:
    """학습 잡의 현재 상태를 한 줄로 요약 반환/출력 — 로그가 아직 없어도 진행 단계를 알 수 있다."""
    import boto3
    sm = boto3.client("sagemaker", region_name=region)
    d = sm.describe_training_job(TrainingJobName=training_job_name)
    info = {
        "status": d["TrainingJobStatus"],           # InProgress / Completed / Failed / Stopped
        "secondary": d.get("SecondaryStatus"),       # Starting / Pending / Downloading / Training / Uploading ...
        "message": (d.get("SecondaryStatusTransitions") or [{}])[-1].get("StatusMessage", ""),
        "failure": d.get("FailureReason"),
    }
    print(f"status={info['status']} / {info['secondary']} — {info['message']}")
    if info["failure"]:
        print("FailureReason:", info["failure"])
    return info


def _bucket_region(bucket: str) -> str | None:
    """S3 버킷의 실제 리전. get_bucket_location은 us-east-1을 None으로 반환한다."""
    import boto3
    from botocore.exceptions import ClientError
    try:
        loc = boto3.client("s3").get_bucket_location(Bucket=bucket)["LocationConstraint"]
    except ClientError:
        return None
    return loc or "us-east-1"


def latest_model_artifact(region: str, job_prefix: str) -> str | None:
    """해당 리전에서 job_prefix로 시작하는 **가장 최근 Completed 학습 잡**의 아티팩트 S3 URI."""
    import boto3
    sm = boto3.client("sagemaker", region_name=region)
    try:
        jobs = sm.list_training_jobs(NameContains=job_prefix, StatusEquals="Completed",
                                     SortBy="CreationTime", SortOrder="Descending",
                                     MaxResults=10)["TrainingJobSummaries"]
    except Exception:  # noqa: BLE001
        return None
    for j in jobs:
        d = sm.describe_training_job(TrainingJobName=j["TrainingJobName"])
        uri = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
        if uri:
            return uri
    return None


def ensure_model_data_in_region(model_data: str | None, region: str,
                                job_prefix: str = "") -> str:
    """model_data가 현재 리전 버킷을 가리키는지 확인하고, 아니면 같은 리전 최신 산출물로 자동 교체.

    🔴 왜 필요한가 (실측 2026-07-30): `%store`는 커널·세션·리전을 넘어 값을 유지한다. 그래서
    AWS_REGION을 바꿔도 model_data는 **옛 리전 버킷**을 계속 가리킨다. SageMaker endpoint는
    같은 리전 S3에서만 아티팩트를 읽으므로 CreateModel이 아래처럼 죽는다:
        ValidationException: Could not access model data at s3://sagemaker-us-east-1-.../model.tar.gz
        ... ensure that the object is located in us-west-2
    메시지에 원인이 있지만 60줄 traceback 끝에 묻혀 있어 리전 문제로 읽기 어렵다.
    또 role 권한 문제로 오해하기 쉽다(메시지가 sts:AssumeRole도 함께 언급하기 때문).

    동작:
      1) model_data 버킷이 이미 `region`이면 그대로 반환.
      2) 아니면 `region`에서 job_prefix 최신 Completed 잡의 아티팩트를 찾아 반환(무엇을 왜 바꿨는지 출력).
      3) 못 찾으면 명확한 메시지로 RuntimeError(무엇을 해야 하는지 안내).
    """
    if model_data and str(model_data).startswith("s3://"):
        bucket = str(model_data).replace("s3://", "").split("/", 1)[0]
        br = _bucket_region(bucket)
        if br == region:
            return str(model_data)
        print(f"⚠️  model_data가 다른 리전을 가리킵니다: 버킷 '{bucket}' = {br} ≠ AWS_REGION({region})")
        print("    (%store 값이 리전 변경 후에도 남아 있어서입니다. 같은 리전 최신 산출물로 교체합니다.)")
    else:
        print(f"⚠️  model_data가 없거나 S3 URI가 아닙니다 — {region}의 최신 학습 산출물을 찾습니다.")

    found = latest_model_artifact(region, job_prefix) if job_prefix else None
    if found:
        print(f"✅ 교체: {found}")
        return found
    raise RuntimeError(
        f"{region}에 배포할 학습 산출물을 찾지 못했습니다(job_prefix='{job_prefix}').\n"
        f"  → 이 리전에서 02_train_sft_sagemaker를 실행해 학습하거나,\n"
        f"  → 다른 리전의 아티팩트를 쓰려면 그 리전으로 AWS_REGION을 맞추거나 "
        f"aws s3 cp 로 {region} 버킷에 복사한 뒤 model_data를 직접 지정하세요."
    )


# ---------------------------------------------------------------------------
# 4) 비용 가드
# ---------------------------------------------------------------------------
COST_WARNING = """
🔴 비용 경고:
  - SageMaker real-time endpoint 는 삭제 전까지 시간당 계속 과금됩니다 (GPU 인스턴스).
    → 실습 후 반드시 99_cleanup 노트북 또는 predictor.delete_endpoint() 실행.
  - Training Job 은 잡 종료 시 과금이 멈추지만, Managed Spot 미사용 시 on-demand 요금.
  - Bedrock Converse 는 호출량(토큰) 기준 과금 — 상시 리소스는 없으나 대량 합성 시 비용 발생.
  - AgentCore Runtime 배포 시 Runtime 리소스 과금 → 미사용 시 정리.
"""


def print_cost_warning() -> None:
    print(COST_WARNING)
