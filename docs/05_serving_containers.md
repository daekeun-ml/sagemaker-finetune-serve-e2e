# 05 · 서빙 컨테이너 선택 — DJL LMI vs 단독 vLLM vs HF TGI

> **대상 독자**: SageMaker에서 파인튜닝한 Gemma SLM/LLM을 real-time endpoint로 올리려는데
> "컨테이너를 무엇으로 골라야 할지" 몰라 막힌 분을 위한 글입니다. vLLM은 들어봤지만 "LMI"가
> 무엇인지, 그리고 이 둘이 왜 따로 등장하는지 헷갈리는 분에게 특히 도움이 됩니다.
> **⚠️ 주의**: 이미지 태그·SDK 버전·백엔드 지원 여부는 🔴 **빠르게 바뀝니다**. 본문의 태그/버전 예시는
> 전부 "실행 전 재확인" 대상입니다(정확한 태그는 `available_images` 페이지에서 확인해 env로 주입하세요).
> **라이브 검증 2026-07** — 세부 내용은 문서 끝의 근거표를 참조하세요.

---

## §0. TL;DR

**한 줄 요약**: `vLLM`은 **엔진**이고, `DJL LMI`는 그 엔진을 **감싸는 AWS 관리형 SageMaker 컨테이너**입니다.
따라서 둘은 경쟁 관계가 아니라 **레이어가 다를** 뿐입니다. 이 킷은 기본값으로 **DJL LMI**(관리가 편합니다),
옵션으로 **단독 vLLM**(최신 기능을 유연하게 쓸 수 있습니다)을 `03_deploy_endpoint`에서 모두 노출합니다.

1. **레이어 멘탈 모델**: `엔진(vLLM/TensorRT-LLM)` ⊂ `서빙 컨테이너(DJL LMI / TGI / 단독 vLLM 서버)` ⊂
   `SageMaker endpoint`의 관계입니다. LMI를 쓴다고 해서 vLLM을 "쓰지 않는" 것이 아닙니다. LMI 안에서 vLLM이 돌 수 있기 때문입니다.
2. **기본(default)은 DJL LMI입니다**: `OPTION_ROLLING_BATCH`로 내부 백엔드(vLLM/TensorRT-LLM)를 고르고,
   나머지는 `OPTION_*` env 또는 `serving.properties`로 튜닝합니다. AWS가 이미지와 통합을 관리해 주므로 초심자가 겪는 마찰이 가장 적습니다.
3. **옵션은 단독 vLLM입니다**: vLLM의 OpenAI 호환 서버를 직접 띄우거나(BYOC), `/invocations`+`/ping`을 구현해
   SageMaker 규약에 맞춥니다. vLLM **최신 릴리스**를 바로 쓰고 native flag를 그대로 활용할 수 있다는 것이 강점이지만,
   대신 이미지·규약·업데이트를 **직접 관리**해야 합니다.
4. **HF TGI도 있습니다**: HuggingFace가 만든 또 다른 서빙 컨테이너로, SageMaker HF DLC로 제공됩니다. `{"inputs","parameters"}`
   스키마가 익숙하고 HF 생태계와 잘 맞물리지만, 서빙 백엔드와 성능 특성은 LMI·vLLM과 다릅니다.
5. **선택 규칙(조건부)**: "관리형이면서 빠른 시작과 백엔드 스위칭"이 필요하면 **LMI**를, "vLLM 최신 기능이나 플래그가 당장 필요"하면
   **단독 vLLM**을, "이미 HF `{inputs}` 스키마나 파이프라인에 묶여 있다"면 **TGI**를 고르세요. 한 번 골랐다고 영원히 묶이는 것은 아닙니다.
   env와 이미지만 바꾸면 얼마든지 갈아탈 수 있습니다.

---

## §0.5. 기존 Pain Point

학습(`02_train_sft_sagemaker`)까지는 잘 끝냈습니다. 이제 배포할 차례인데, 검색해 보면 다음과 같은 정보들이 뒤섞여 나옵니다.

- "vLLM이 제일 빠르다"고 합니다. 그럼 vLLM을 써야 할까요?
- 그런데 AWS 예제는 하나같이 **DJL / LMI**를 씁니다. 그렇다면 LMI는 vLLM과 **다른 것**일까요? 더 느릴까요?
- HuggingFace 예제는 **TGI**를 씁니다. 이건 또 다른 선택지일까요?
- 셋 다 "LLM 서빙"이라고 하는데, **무엇이 무엇을 감싸는지** 도무지 그림이 그려지지 않습니다.

핵심 혼동은 하나입니다. 바로 **"엔진"과 "서빙 컨테이너"를 같은 레이어로 착각**하는 것입니다.
이 혼동만 풀면 나머지는 "언제 무엇을 고르는가"라는 단순한 선택 문제로 줄어듭니다.

---

## §1. 왜? — 레이어가 다르다 (엔진 ≠ 서빙 컨테이너)

### 쉽게 말하면
- **엔진**(`vLLM`, `TensorRT-LLM`)은 자동차의 **엔진**에 해당합니다. 토큰을 빠르게 뽑아내는 핵심 로직
  (PagedAttention, continuous batching 등)을 담당합니다.
- **서빙 컨테이너**(`DJL LMI`, `TGI`, `vLLM OpenAI 서버`)는 그 엔진을 얹은 **완성차**입니다. HTTP 서버,
  모델 로딩, 요청 배칭, health check(`/ping`), 추론 endpoint(`/invocations`)까지 하나로 묶은 것입니다.
- **SageMaker endpoint**는 그 완성차를 굴리는 **도로와 관제**에 해당합니다(오토스케일·IAM·CloudWatch·인스턴스 관리).

`DJL LMI`는 AWS가 만든 완성차인데, **엔진을 바꿔 끼울 수 있다**는 점이 특징입니다. 예를 들어 `OPTION_ROLLING_BATCH=vllm`으로 지정하면
그 안에서 vLLM 엔진이 돕니다. 즉 "LMI를 쓴다는 것이 곧 vLLM을 쓰지 않는다는 뜻"이 **아니라는** 것입니다.

```
┌──────────────────────────────────────────────────────────────┐
│ SageMaker real-time endpoint (오토스케일·IAM·CloudWatch)        │  ← 인프라 레이어
│  ┌────────────────────────────────────────────────────────┐  │
│  │  서빙 컨테이너 (HTTP + /ping + /invocations + 배칭)        │  │  ← 컨테이너 레이어
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │  │
│  │  │  DJL LMI      │  │  HF TGI       │  │ vLLM OpenAI  │ │  │
│  │  │ (AWS 관리)     │  │ (HF 관리)      │  │ 서버 / BYOC   │ │  │
│  │  │  ┌─────────┐  │  │  ┌─────────┐  │  │ ┌─────────┐  │ │  │
│  │  │  │ vLLM /  │  │  │  │ 자체     │  │  │ │  vLLM   │  │ │  │  ← 엔진 레이어
│  │  │  │ TRT-LLM │  │  │  │ 백엔드    │  │  │ │ (직접)   │  │ │  │
│  │  │  └─────────┘  │  │  └─────────┘  │  │ └─────────┘  │ │  │
│  │  └───────────────┘  └───────────────┘  └──────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
    ▲ 같은 endpoint 레이어. 컨테이너/엔진만 갈아끼우는 문제다.
```

### 대조 (요지)
| 관점 | DJL LMI | 단독 vLLM | HF TGI |
|---|---|---|---|
| 이게 뭐냐 | **컨테이너**(엔진 wrap) | **엔진**(서버까지 포함) 직접 | **컨테이너**(HF 백엔드) |
| 누가 관리 | AWS | 당신(BYOC) 또는 vLLM 커뮤니티 | HuggingFace + AWS(DLC) |
| vLLM과의 관계 | **감싼다** | **그 자체** | 별개 백엔드 |

### 기술적 차이 3가지
1. **엔진 스위칭 지점**: LMI는 컨테이너를 바꾸지 않고 `OPTION_ROLLING_BATCH`(env)만으로 vLLM과 TRT-LLM을 오갈 수 있습니다.
   반면 단독 vLLM은 엔진이 곧 컨테이너이므로 스위칭이라는 개념 자체가 없습니다. TGI는 자체 백엔드로 고정되어 있습니다.
2. **최신성과 안정성의 트레이드오프**: 단독 vLLM은 upstream 릴리스를 바로 당겨 쓸 수 있어 **최신 기능을 가장 빠르게** 반영합니다.
   반면 LMI/TGI는 AWS나 HF가 특정 버전을 검증해 이미지로 굽기 때문에 **한 박자 늦지만 그만큼 검증되어** 있습니다.
3. **SageMaker 규약을 누가 처리하는가**: LMI/TGI DLC는 `/ping`, `/invocations`, 모델 로딩을 **AWS와 HF가 이미 구현**해 두었습니다.
   반면 단독 vLLM을 BYOC로 올리면 그 규약을 **직접 맞춰야** 합니다(또는 vLLM OpenAI 서버 앞단에 adapter를 둡니다).

---

## §2. 3-way 대조표 (선택 결정용)

| 기준 | **DJL LMI** (기본) | **단독 vLLM** (옵션) | **HF TGI** |
|---|---|---|---|
| 정체 | AWS 관리형 서빙 컨테이너 | 엔진 직접 배포(OpenAI 서버/BYOC) | HF 서빙 컨테이너(SageMaker HF DLC) |
| 엔진 선택 | **vLLM / TensorRT-LLM** 선택 가능 | vLLM 고정(그 자체) | TGI 자체 백엔드 |
| 설정 방법 | `OPTION_*` env + `serving.properties` | **vLLM native flag / args** 직접 | `{"inputs","parameters"}` + HF env |
| 백엔드 전환 | `OPTION_ROLLING_BATCH`로 스위칭 | 없음(단일 엔진) | 없음 |
| 버전 최신성 | AWS 검증 후 반영(한 박자 늦음) | **upstream 최신 즉시** | HF 검증 후 반영 |
| 유지보수 부담 | **낮음**(AWS가 이미지·규약 관리) | **높음**(이미지·규약·패치 직접) | 낮음(HF/AWS 관리) |
| SageMaker 통합 | 네이티브(`image_uris.retrieve`, DLC) | BYOC 필요(`/ping`+`/invocations`) 또는 어댑터 | 네이티브(HF DLC) |
| 스트리밍 | 지원(SageMaker 응답 스트림) | 지원(구현/규약에 의존) | 지원 |
| 언제 고르나(조건부) | 관리형·빠른 시작·백엔드 스위칭 원할 때 | vLLM **최신 기능/플래그가 당장** 필요할 때 | 이미 HF `{inputs}` 파이프라인·스키마에 묶였을 때 |

> 🔴 각 컨테이너가 지원하는 정확한 엔진·옵션·스트리밍 방식은 버전마다 바뀝니다. 이 표는 "성향"을 기준으로 정리한 것이므로,
> **실행 전에** 각 프로젝트 문서(§근거표)에서 현행 지원 매트릭스를 재확인하세요.

### SageMaker 추론 4옵션과의 관계 (배포 형태)
서빙 **컨테이너** 선택과는 별개로, SageMaker 추론의 **배포 형태**는 다음 4가지로 나뉩니다.

| 형태 | LLM/SLM 적합성 |
|---|---|
| **Real-time** | ✅ 이 킷의 기본(상시 저지연). GPU 인스턴스. |
| **Serverless** | ⚠️ **GPU 없음** → LLM/SLM **부적합**(현시점). |
| **Asynchronous** | 대용량·긴 처리(장문 배치성 요청)에 유용. |
| **Batch Transform** | 상시 endpoint 없이 대량 오프라인 추론. |

> 위 세 컨테이너(LMI/vLLM/TGI)는 주로 **real-time**(및 async) 위에 올립니다.
> Serverless는 GPU가 없어서 SLM 서빙 대상이 아닙니다.

---

## §3. DLC 이미지 URI 패턴 (LMI)

### 쉽게 말하면
LMI 컨테이너도 결국은 **ECR에 올라간 도커 이미지**입니다. AWS DLC(Deep Learning Container) 레지스트리에서
가져다 씁니다.

- **레지스트리 계정**: `763104351884` (대부분의 리전에서 공용으로 씁니다)
- **URI 패턴**:
  ```
  763104351884.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
  ```
- **LMI 태그 형태(예시)**: `djl-inference:0.XX-lmiXX-cuXXX` 계열입니다.
  🔴 정확한 `repository`/`tag`는 버전마다 다르므로, **available_images 페이지에서 확인**한 뒤 env로 주입하세요.

이미지를 해석하는 경로는 두 가지입니다.
1. **SDK resolve**: `sagemaker.image_uris.retrieve(framework="djl-lmi"/"djl-deepspeed" 등, region, version=...)`
   를 호출하면 SDK가 계정·리전·태그를 조립해 줍니다. ⚠️ 다만 SDK 버전에 매인 태그 목록이라 **최신보다 늦을 수** 있다는 점에 유의하세요.
2. **직접 지정**: 위 패턴으로 URI를 **직접** 만들고 env로 오버라이드하는 방식입니다. 이렇게 하면 available_images가 갱신되어도
   코드를 고칠 필요 없이 태그만 교체하면 됩니다.

> 🔴 `image_uris.retrieve`에 넘기는 framework 문자열(예: `djl-lmi`)과 지원되는 `version` 값은
> SDK 버전과 시점마다 다릅니다. **실행 전에** SDK 문서와 available_images를 함께 재확인하세요.

### 이 킷에서의 해석 — `common/dlc.py`
이 킷은 **env를 우선하는 직접 지정** 전략을 택했으며, SDK resolve는 폴백으로 둡니다. 우선순위는 다음과 같습니다.

1. `INFER_DLC_IMAGE_URI` 또는 `DLC_IMAGE_URI`(완전한 URI)가 있으면 그대로 사용합니다(가장 확실한 방법입니다).
2. `INFER_DLC_REPOSITORY` + `INFER_DLC_TAG`(없으면 `DLC_*`)가 있으면 패턴으로 조립합니다.
3. 둘 다 없으면 `None`을 반환하고, 노트북이 버전 조합으로 폴백 resolve합니다.

```python
# common/dlc.py (요지 — 추론 이미지 해석)
def resolve_inference_image(region: str) -> str | None:
    full = os.environ.get("INFER_DLC_IMAGE_URI") or os.environ.get("DLC_IMAGE_URI")
    if full:
        return full
    repo = os.environ.get("INFER_DLC_REPOSITORY") or os.environ.get("DLC_REPOSITORY")
    tag  = os.environ.get("INFER_DLC_TAG")        or os.environ.get("DLC_TAG")
    if repo and tag:
        return build_dlc_image_uri(region, repo, tag)   # 763104351884.dkr.ecr.<region>...:<tag>
    return None
```

즉 **LMI 이미지를 쓰려면** `INFER_DLC_REPOSITORY`(예: `djl-inference`)와 `INFER_DLC_TAG`(예: LMI 태그)를
env로 주입하면 됩니다. 계정ID와 패턴은 코드에 안정적으로 담겨 있으므로, **바뀌는 태그만 env로** 관리하면 됩니다.
(참고로 `common/config.py`의 `HF_*_VERSION`은 **HF DLC(학습/TGI)** 조합을 고정하는 핀이며 LMI 태그와는 별개입니다.)

> 🔴 `common/dlc.py`의 기본 repository/tag 값은 **예시(TODO verify)**일 뿐입니다. 실행 전에 available_images에서
> 현행 값을 확인해 env로 덮어쓰세요. 계정ID와 URI 패턴은 하드코딩되어 있지만, 이는 AWS 공개 레지스트리 값이므로
> 시크릿이 아닙니다(반면 고객 계정ID·role·bucket은 여전히 하드코딩하면 안 됩니다).

---

## §4. 이 킷은 둘 다 노출한다 — `03_deploy_endpoint`

이 킷은 "하나만 쓰라"고 강요하지 않습니다. **DJL LMI를 기본으로**, **단독 vLLM을 옵션으로** 나란히 제공합니다.
공통점은 배포한 뒤의 호출 방식이 **동일**하다는 것입니다. 어떤 컨테이너를 쓰든 결국 `sagemaker-runtime`으로 부르기 때문입니다.

### (A) 기본 경로 — 관리형 컨테이너 배포
`03_deploy_endpoint`는 `common/dlc.py`로 이미지 URI를 해석한 뒤 SageMaker Model로 배포합니다.
`INFER_DLC_*` env가 LMI 태그를 가리키면 **LMI**로 배포되고, 비워 두면 버전 조합으로 폴백합니다.

```python
# 요지: env(INFER_DLC_*)로 LMI 이미지를 가리키면 LMI, 없으면 폴백
from common import dlc
infer_image = dlc.resolve_inference_image(config.AWS_REGION)   # None이면 버전 resolve
# LMI라면 컨테이너 env로 백엔드/모델 지정 (실행 전 옵션명 재확인):
#   OPTION_ROLLING_BATCH=vllm         ← 내부 엔진을 vLLM으로
#   OPTION_MODEL_ID / HF_MODEL_ID     ← 서빙할 모델(또는 S3 model_data)
#   OPTION_MAX_MODEL_LEN, OPTION_TENSOR_PARALLEL_DEGREE ...  ← 튜닝
```

> 🔴 정확한 `OPTION_*` 키 이름과 기본값은 LMI 버전마다 다릅니다. 위 예시는 성향을 보여 주는 것일 뿐이므로, 실행 전에
> LMI 문서(§근거표)에서 현행 키를 확인하세요.

### (B) 옵션 경로 — 단독 vLLM
vLLM을 직접 쓰려면 다음 두 방식 중 하나를 택하면 됩니다.
- **vLLM OpenAI 호환 서버**에 SageMaker 규약 어댑터를 붙이거나,
- **BYOC** 방식으로, 컨테이너가 `/invocations`(추론)와 `/ping`(health)을 구현하도록 이미지를 직접 빌드합니다.
- 배포할 때 `INFER_DLC_IMAGE_URI`에 **본인이 빌드해 푸시한 vLLM 이미지 URI**를 넣으면 킷의 배포 코드를 그대로 재사용할 수 있습니다.

### (C) 호출은 컨테이너와 무관하게 동일 — `common/aws_utils.py`
```python
# invoke_sagemaker_endpoint(): boto3 "sagemaker-runtime" invoke_endpoint()
#   payload/응답 스키마는 컨테이너에 맞춰 방어적 파싱({"inputs","parameters"} 가정, 조정 가능)
#   스트리밍은 invoke_endpoint_with_response_stream()
```
> 🔴 **서비스 경계에 주의하세요.** SageMaker endpoint 호출은 `sagemaker-runtime`으로, Bedrock Claude 호출은
> `bedrock-runtime`(Converse)으로 하며 **별개 서비스이고 별개 클라이언트**입니다. 따라서 "endpoint를 Bedrock API로 호출"하는 것은 잘못된 방법입니다.
> (`common/llm_gateway.py`는 LiteLLM으로 두 백엔드를 하나의 인터페이스로 묶지만, 내부적으로는 각자의
> 클라이언트를 씁니다.)

### 연결 노트북 (트랙 공통 `00~06,99`)
- `02_train_sft_sagemaker`에서 머지 가중치를 S3(`model_data`)로 올립니다.
- **`03_deploy_endpoint`**가 바로 본 문서의 컨테이너 선택이 적용되는 지점입니다.
- `04_evaluate`에서 held-out 평가로 성공기준을 수치화합니다.
- `05_agentic_strands` / `06_agentcore_deploy`에서는 endpoint를 tool로 쓰는 agentic loop를 구성합니다
  (Strands `BedrockModel`/`LiteLLMModel` → AgentCore Runtime, ARM64 `/invocations`+`/ping:8080`).
- `99_cleanup`에서 **endpoint를 삭제해 과금을 중단**합니다.

---

## §4.5. SDK v3 로컬 검증 모드 — `ModelBuilder(mode=...)` (IN_PROCESS / LOCAL_CONTAINER)

> `03_deploy_endpoint` 노트북에서 링크로 참조. gemma-4 배포 전 "로컬 검증"을 SDK 모드로 할지 판단하는 상세 근거.

SDK v3 `ModelBuilder`는 **같은 코드**를 3단계 대상에 배포할 수 있습니다(생성자/`build`에 `mode` 지정, 기본 `SAGEMAKER_ENDPOINT`):

| mode | 실행 위치 | 용도 | 요구 |
|---|---|---|---|
| `Mode.IN_PROCESS` | 현재 파이썬 프로세스 | 가장 빠른 로직 검증(초경량) | 없음(백엔드 제약 큼) |
| `Mode.LOCAL_CONTAINER` | 로컬 Docker 컨테이너 | endpoint와 동일 컨테이너 재현 | 로컬 Docker + GPU |
| `Mode.SAGEMAKER_ENDPOINT` | SageMaker(클라우드) | 실제 서빙(기본) | AWS 과금 |

import: `from sagemaker.serve.mode.function_pointers import Mode` (실측 — `sagemaker.serve`에 직접 `Mode` 없음).

### ❓ `IN_PROCESS`로 gemma를 초경량 검증하면 되지 않나?
**안 됩니다(생성형 LLM 미지원).** IN_PROCESS 서버는 `model=<HF id>`를 받으면 내부적으로 **`transformers.pipeline` 또는 `SentenceTransformer`(임베딩)로만** 로드를 시도합니다(소스 실측: `sagemaker/serve/model_server/in_process_model_server/app.py`). 즉 **분류·임베딩 같은 경량 모델 전용**입니다:
- gemma-4는 멀티모달(오디오 포함)이라 pipeline이 `AnyToAnyPipeline`으로 잡혀 `librosa` 등을 요구하고,
- 임베딩 모델도 아니라 `SentenceTransformer` 폴백도 실패(`UnboundLocalError`).
- LLM을 IN_PROCESS로 띄우려면 `InferenceSpec`(load/invoke)을 직접 구현해야 하는데, 이는 vLLM 엔진을 손으로 재구현하는 셈이라 실익이 없습니다.
- 부가: IN_PROCESS도 `ModelBuilder.__post_init__`이 `role_arn`을 해석하므로(IAM user면 `RoleValidationError`) 로컬 실행이라도 `role_arn=`을 넘겨야 합니다.

### ❓ `LOCAL_CONTAINER`는 gemma-4 E4B에 쓸 만한가?
**부적합합니다(실측 2026-07):**
- **vLLM DLC + LOCAL_CONTAINER** ❌ — `image_uri`만 주면 passthrough라 `model_server=None`이 되고, LOCAL_CONTAINER의 `create_server`엔 **VLLM 분기가 없어**(TRITON/DJL_SERVING/TGI/MMS/... 만 존재) `None.logs()`로 크래시.
- **DJL LMI + LOCAL_CONTAINER** ⚠️ — 컨테이너·마운트까지는 되지만(모델을 `model_path/code/`에 **실파일**로 둬야 마운트됨 — 심링크는 컨테이너 안에서 깨짐), 당시 실측에서 `weights not initialized: layers.24~41...k_norm` ValueError로 엔진 초기화 실패했습니다. **다만 이 실패의 원인은 LMI/vLLM이 아니라 우리가 넘긴 체크포인트였습니다** — 상세는 아래 §4.7. 지금은 학습 스크립트가 그 텐서를 복원해 저장하므로 이 에러는 재현되지 않습니다.
- docker-py 기본 타임아웃 60s는 큰 이미지에 부족 → `container_timeout_in_seconds` 상향. 그래도 `deploy()`가 `ReadTimeout`을 내도 컨테이너는 백그라운드로 기동 중일 수 있어 `docker logs`/`curl :8080/invocations`로 직접 확인.
- 참고: **HF PyTorch Inference DLC + `model_server=MMS` + LOCAL_CONTAINER**는 E4B에서 실측 성공했습니다(로드+`/invocations` 응답 확인). 다만 이 킷은 transformers 단건 서빙 경로를 **서빙 선택지에서 제외**했습니다(연속 배칭·스트리밍 없음) — 아래 §4.8 참고.

### ➡️ 결론 (이 킷의 gemma-4 경로)
- **로컬 검증**: SDK 로컬 모드(IN_PROCESS/LOCAL_CONTAINER)를 **기본 경로로 쓰지 않습니다.** 대신 **`02b_local_serve`**에서 `vllm serve`로 실제 엔진을 띄워 확인합니다.
- **클라우드 배포**: **vLLM DLC(기본)** · **SGLang DLC**(1-A) · **DJL LMI**(1-B) — E2B/E4B/12B/26B/31B 전 사이즈. 셋 다 vLLM 계열/연속 배칭이라 로컬 `vllm serve` 검증이 그대로 유효합니다.
- SDK 모드는 "배포 API 동일성"을 이해하는 개념으로만 소개하고, gemma-4 서빙 자체엔 위 경로를 씁니다.

---

## §4.7. 🔴 E2B/E4B를 vLLM으로 서빙하려면 — KV-shared dead weight 복원 (실측 2026-07-30)

> **결론부터: "E4B는 vLLM으로 못 띄운다"는 말은 사실이 아닙니다.** 원본 `google/gemma-4-E4B-it`은 vLLM에서
> 그대로 뜹니다. 못 뜨는 건 **transformers `save_pretrained`를 거친 체크포인트**입니다.

**무엇이 없어지나.** gemma-4 E계열은 뒤쪽 `num_kv_shared_layers`개 레이어가 앞 레이어의 KV를 재사용합니다
(E4B: 42층 중 24~41의 18층). transformers는 그 레이어에 `k_norm`/`k_proj`/`v_proj` 모듈을 **아예 만들지 않습니다**
(`modeling_gemma4.py`: *"Layers sharing kv states don't need any weight matrices"*). 그래서 파인튜닝 후
`save_pretrained`로 저장하면 원본에 있던 그 텐서가 **소실**됩니다 — 실측 정확히 **54개**
(18층 × `k_norm`/`k_proj`/`v_proj`).

**왜 vLLM만 죽나.** vLLM `Gemma4Attention`은 `k_norm`을 **전 레이어에 등록**합니다(사용은 `if not
self.is_kv_shared_layer`로 건너뛰지만 등록은 무조건). 등록된 파라미터가 체크포인트에 없으면 weight 검증이
`ValueError: Following weights were not initialized from checkpoint: ...layers.24~41...k_norm`으로 실패합니다.
transformers는 자기가 안 만든 모듈이니 아무 문제가 없습니다 — **엔진 간 기대치 차이**입니다.

**복원해도 정확도에 무해한 이유.** shared 레이어는 forward에서 앞 레이어의 KV를 그대로 재사용하므로
(`if self.is_kv_shared_layer: key_states, value_states = shared_kv_states[...]`) 이 텐서는 **연산에 쓰이지 않습니다**.
LoRA(q/k/v/o_proj 타깃)도 그 레이어엔 모듈이 없어 학습되지 않습니다. 즉 base 값을 그대로 되살리는 것은
**vLLM의 weight 검증만 통과시키는 목적**이며 출력에 영향이 없습니다.

**이 킷의 처리.** `scripts/train.py`·`train_grpo.py`의 `_revive_kv_shared_from_base()`가 저장 직전에 base
체크포인트에서 그 54개를 읽어 `save_pretrained(state_dict=...)`로 함께 저장합니다. 모델 객체엔 해당 모듈이
없으므로 **명시 `state_dict` 전달이 유일한 방법**입니다. `num_kv_shared_layers=0`인 12B/26B/31B는 자동으로
건너뜁니다(복원 0개).

**실측 검증(L40S 46GB, vLLM 0.25.1, E4B bf16):**

| 체크포인트 | 저장 키 | vLLM 로드 |
|---|---|---|
| 복원 전(`save_pretrained` 그대로) | 665 | ❌ `weights not initialized ...k_norm` |
| 복원 후(이 킷) | 719 = 원본과 동일 | ✅ 로드 성공 + 정상 생성 |

**참고: vLLM issue [#44788](https://github.com/vllm-project/vllm/issues/44788)** (as of 2026-07-30 OPEN)은
바로 이 현상입니다. 이슈 제목이 "Gemma 4 models with KV sharing"이라 "E계열은 vLLM 불가"로 읽히기 쉬우나,
두 번째 코멘트가 `save_pretrained` 왕복 후에만 재현됨을 보여줍니다. 원본 체크포인트는 그 54개를 모두 갖고 있습니다
(safetensors 헤더 직접 확인). FP8 변형(`leon-se/gemma-4-E4B-it-FP8-Dynamic`)도 원본 유래라 동일하게 정상입니다.

**AWS DLC 이미지 (763104351884, us-west-2 ECR 실조회 2026-07-30):**
```
vllm:0.25.1-gpu-py312-cu130-ubuntu22.04-sagemaker                  # push 2026-07-22
sglang:0.5.15-gpu-py312-cu130-ubuntu24.04-sagemaker                # push 2026-07-23  (⚠️ ubuntu24.04)
djl-inference:0.36.0-lmi27.0.0-cu130-v1.1                          # push 2026-07-16  (LMI 최신)
# (HF Inference DLC는 서빙 선택지에서 제외 — 단건·스트리밍 불가. 필요 시 dlc.resolve_hf_inference_image())
```
⚠️ 태그는 자주 갱신됩니다 — 배포 전 재확인하세요:
`aws ecr describe-images --registry-id 763104351884 --repository-name vllm --region <region>`

---

## §4.8. 서빙 엔진·이미지를 env로 고르기

**한 줄**: `SERVING_ENGINE`으로 엔진을 고르고, 이미지는 버전 env(권장) 또는 완전 URI로 지정합니다. **리전은 절대 하드코딩하지 않습니다** — 코드가 `AWS_REGION`으로 채웁니다.

| `SERVING_ENGINE` | 컨테이너 | 특징 | 노트북 절 |
|---|---|---|---|
| `vllm` (기본) | vLLM DLC | 최신 vLLM, 가장 널리 검증됨 | 1-A |
| `sglang` | SGLang DLC | vLLM 대안(RadixAttention). gemma4 지원 | 1-A (같은 셀) |
| `lmi` | DJL LMI | AWS 관리형(내부 백엔드 vLLM), `OPTION_*` env | 1-B |

**셋 다 연속 배칭 + OpenAI 호환(`messages`) + 스트리밍을 지원합니다** — 그래서 엔진을 바꿔도 호출 코드(04 평가·05 agentic)가 그대로 돕니다.

> ❓ **transformers(HF Inference DLC) 경로는 왜 없나요?** `code/inference.py` 핸들러로 서빙하면 **단건 처리**라 연속 배칭이 없고, HF Inference Toolkit이 응답을 완성본으로 버퍼링해 **토큰 스트리밍도 불가**합니다(§4.6). E4B가 vLLM으로 못 뜬다고 알려졌을 때의 우회로였는데, 그 원인이 체크포인트였음이 밝혀져(§4.7) 더는 필요하지 않습니다. `resolve_hf_inference_image()`는 `common/dlc.py`에 남아 있으니 직접 쓸 수는 있습니다.

이 킷의 `.env`는 **이미지를 리전 포함 완전 URI로 하드코딩**해 둡니다(ECR 실조회 검증값). 무엇이 쓰이는지 한눈에 보이고, SDK의 `image_uris.retrieve` 추측을 우회합니다:

```bash
# .env
SERVING_ENGINE=vllm
VLLM_IMAGE_URI=763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.25.1-gpu-py312-cu130-ubuntu22.04-sagemaker
SGLANG_IMAGE_URI=763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:0.5.15-gpu-py312-cu130-ubuntu24.04-sagemaker
LMI_IMAGE_URI=763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi27.0.0-cu130-v1.1
```

리전을 옮길 때는 `AWS_REGION`과 위 URI의 리전을 함께 바꿉니다(이미지는 리전별 ECR에서만 pull됨). 리전을 자주 옮긴다면 해당 줄을 **주석 처리**하세요 — 그러면 코드가 `AWS_REGION`으로 URI를 자동 조립합니다(`VLLM_DLC_VERSION` 등으로 버전만 지정).

해석 결과는 `dlc.serving_image_table(region)`으로 4개 엔진을 한 번에 확인할 수 있고, `03_deploy_endpoint` 노트북이 이를 출력합니다.

### 🔴 서빙 env는 CLI 플래그로 기계적으로 변환됩니다
두 DLC의 `sagemaker_entrypoint.sh`(aws/deep-learning-containers 소스 확인 2026-07-30)는 **접두사를 떼고 소문자화 + `_`→`-`** 해서 그대로 엔진 CLI 플래그로 넘깁니다:

| env | → CLI | 엔진 |
|---|---|---|
| `SM_VLLM_MAX_MODEL_LEN=2048` | `--max-model-len 2048` | `vllm.entrypoints.openai.api_server` |
| `SM_SGLANG_TP_SIZE=1` | `--tp-size 1` | `sglang.launch_server` |
| `SM_SGLANG_MODEL_PATH=/opt/ml/model` | `--model-path /opt/ml/model` | (미지정 시 이 값이 기본) |

즉 **화이트리스트가 없습니다** — 엔진이 아는 플래그면 무엇이든 이 규칙으로 전달할 수 있습니다. LMI만 규약이 달라 `OPTION_*`(예 `OPTION_ROLLING_BATCH=vllm`)를 씁니다.

---

## §4.9. 🔴 24GB GPU에서 endpoint가 `Failed` — CUDA OOM은 `max_num_seqs` 기본값 때문 (실측 2026-07-31)

### 한 줄

**모델이 커서가 아닙니다.** vLLM 기본 `max_num_seqs=256`이 실습 규모에 과하게 잡혀 샘플러 버퍼가 GPU를 넘깁니다. 이 킷은 **32**로 낮춰 두었고, GPU를 바꿀 필요는 없습니다.

### 증상 — 원인이 안 보입니다

`describe-endpoint`가 알려주는 건 이것뿐입니다:

```
FailureReason: The primary container for production variant AllTraffic
               did not pass the ping health check.
```

진짜 원인은 **CloudWatch endpoint 로그 안에만** 있습니다:

```
Available KV cache memory: 4.69 GiB
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
  GPU 0 has a total capacity of 21.96 GiB of which 147.12 MiB is free.
  ... flashinfer_sample -> top_k_mask_logits -> torch.empty_like(logits)
```

### 왜 하필 `256.00 MiB`인가 — 결정적 단서

우연이 아니라 **정확한 산술**입니다:

```
max_num_seqs × vocab_size × 4B(fp32) = 256 × 262,144 × 4 = 256 MiB
```

gemma-4의 vocab이 **262,144**로 크기 때문에, 동시 시퀀스 기본값 256이면 샘플러 logits 버퍼 하나가 256 MiB입니다. flashinfer가 `torch.empty_like`로 사본을 하나 더 만들므로 실제로는 **512 MiB**가 필요합니다.

### 메모리 예산 (실측, ml.g6.2xlarge = L4 22.9GB)

한도 = `21.96 × 0.92 = 20.21 GiB`

| 항목 | 멀티모달(05) | 텍스트(02) |
|---|---|---|
| 가중치 | 15.18 GiB (vision 포함) | 14.23 GiB |
| KV 캐시 (vLLM 자동 배정) | 4.69 GiB | 0.47 GiB |
| 활성 + 비torch + CUDAGraph | 1.12 GiB 필요 | 6.28 GiB |
| **결과** | **남은 여유 0.34 GiB → 0.78 GiB 부족, OOM** | 간신히 통과 |

- **텍스트 트랙도 안전지대가 아닙니다** — KV 여유가 0.47 GiB뿐이었습니다. 멀티모달은 vision tower로 가중치가 ~1 GiB 크고, 그 차이가 그대로 실패로 이어졌습니다.
- vLLM 자신도 로그에서 `--kv-cache-memory=3.76 GiB`를 권고합니다 → **KV를 4.69로 과대 배정한 것**입니다.

### ❓ GPU 타입을 바꿔야 하나?

**아닙니다.** 로컬 L40S를 `gpu_memory_utilization=0.441`로 제한해 **L4와 같은 절대 예산(20.2 GiB)** 을 만든 뒤 실측한 결과:

| 설정 | 결과 |
|---|---|
| `max_num_seqs=256`(기본), vLLM 0.26.0 컨테이너 | ❌ CUDA OOM |
| `max_num_seqs=32`, 같은 예산 | ✅ 로드 + **이미지 추론 성공** (KV 3.36 GiB, 여유 1.54 GiB) |

즉 **설정 문제이지 하드웨어 문제가 아닙니다**. 다만 다음 경우엔 `ml.g6e.2xlarge`(L40S 45GB)가 여유롭습니다:
- 동시 요청이 많아 `max_num_seqs`를 크게 두어야 할 때
- `max_model_len`을 4096 이상으로 늘릴 때 (KV 캐시가 길이에 비례)

### 대응 — 엔진별 설정 (플래그명 라이브 검증)

| 엔진 | 동시 시퀀스 | 메모리 비율 |
|---|---|---|
| vLLM | `SM_VLLM_MAX_NUM_SEQS=32` | `SM_VLLM_GPU_MEMORY_UTILIZATION=0.90` |
| SGLang | `SM_SGLANG_MAX_RUNNING_REQUESTS=32` | `SM_SGLANG_MEM_FRACTION_STATIC=0.90` |
| LMI | `OPTION_MAX_ROLLING_BATCH_SIZE=32` | `OPTION_GPU_MEMORY_UTILIZATION=0.90` |

로컬 검증(`scripts/serve_local_vllm.sh`)도 같은 값을 기본으로 쓰며 `MAX_NUM_SEQS`/`GPU_MEM_UTIL` env로 덮어쓸 수 있습니다.

🔴 **키 이름을 직접 쓰지 마세요 — `dlc.serving_env()`가 한 곳에서 관리합니다.** 위 표처럼 같은 설정이 엔진마다 다른 키를 쓰기 때문에, 노트북마다 dict를 손으로 쓰면 값을 하나 바꿀 때 한 엔진을 빼먹습니다(실제로 이 킷에서 `max_num_seqs`를 vLLM 분기에만 넣고 LMI 분기를 놓쳐 OOM이 재발할 뻔했습니다). 그래서 "의미 → 엔진별 키" 매핑을 `common/dlc.py` 한 곳에 두고 노트북은 의미만 넘깁니다:

```python
serve_env = dlc.serving_env(
    ENGINE,                       # 'vllm' | 'sglang' | 'lmi'
    max_model_len=4096,           # 서빙 길이(학습 길이 아님)
    max_num_seqs=32,              # OOM 방지
    gpu_memory_utilization='0.90',
    mm_limit=json.dumps({'image': 1}),        # 멀티모달만. SGLang은 기본 허용이라 무시됨
    hf_token=config.get_serving_hf_token(),   # gated 모델일 때만 채워짐
)
```

세 엔진 중 무엇을 골라도 같은 호출로 알맞은 키가 나옵니다(LMI는 `OPTION_ROLLING_BATCH=vllm`과 `tp='max'` 관용구까지 자동).

> ⚠️ **버전에 기대지 마세요** — 같은 절대 예산에서 vLLM 0.25.1은 KV를 3.36 GiB로, 0.26.0은 4.69 GiB로 잡았습니다(실측). 컨테이너 태그를 올리면 여유가 사라질 수 있으므로 명시적으로 낮춰 둡니다.

---

## §4.10. 🔴 `max_tokens`가 작으면 응답이 조용히 잘립니다 — `finish_reason`으로 확인 (실측 2026-07-31)

### 한 줄

응답이 짧게 끝났을 때 **모델이 요약을 잘한 것인지 잘린 것인지 구분하려면 `finish_reason`을 보세요.** `length`면 잘린 것이고, `stop`이면 모델이 스스로 끝낸 것입니다.

### 실측 (요약 트랙 endpoint, 입력 5,996자)

| `max_tokens` | `finish_reason` | `completion_tokens` | 응답 길이 |
|---|---|---|---|
| **256** | **`length`** ⛔ | 256 (한도에 붙음) | 902자 |
| 512 | `stop` ✅ | 397 | 1,446자 |
| 1024 | `stop` ✅ | 571 | 1,829자 |

256으로는 요약이 **문장 중간에서 끊겼고**, 에러도 경고도 나지 않았습니다. 512부터 모델이 스스로 종료합니다.

### 왜 놓치기 쉬운가

- **예외가 없습니다.** 잘린 응답도 정상 200 응답이라 코드가 그냥 통과합니다.
- 노트북이 `print(pred[:400])` 처럼 출력까지 자르면 **이중으로 가려집니다** — 이 킷이 실제로 그랬습니다(응답 1,262자 중 400자만 표시). 그래서 `common/display_utils.show_inference()`로 전체를 렌더링하도록 바꿨습니다.
- 평가 지표에서는 더 위험합니다 — 정답이 `max_tokens`보다 길면 예측이 구조적으로 잘려 **ROUGE/정확도가 실제보다 낮게** 나옵니다(모델 탓이 아닌데 모델을 의심하게 됩니다).

### 대응

트랙별 `gen_max_tokens`(spec)를 정답 길이 분포에서 정하고, 배포·평가·에이전트 셀이 **모두 같은 값**을 쓰게 했습니다:

| 트랙 | `gen_max_tokens` | 근거(정답 토큰 분포) |
|---|---|---|
| 추출 / 분류 | 256 | JSON·라벨은 짧음 |
| 요약 | 512 | median 209 / p90 475 (max 964는 미포함 — 필요하면 1024) |
| 도메인 QA | 512 | 256이면 13건(8.7%)이 잘림 |

확인 방법:

```python
r = client.invoke_endpoint(...)          # OpenAI 호환 응답
ch = json.loads(r['Body'].read())['choices'][0]
assert ch['finish_reason'] != 'length', '응답이 잘렸습니다 — max_tokens를 올리세요'
```

---

## §4.11. 🔴 노트북에 응답을 찍을 때 — 입력 텍스트가 마크다운을 깨뜨립니다 (실측 2026-07-31)

### 증상

요약 트랙 실시간 추론에서 **첫 번째 결과의 응답이 화면에 아예 안 보였습니다**(두 번째는 정상). 호출은 성공했고 응답도 1,596자가 정상 수신된 상태였습니다.

### 원인 — 시드 데이터의 백틱

billsum 법안 원문은 구식 인용부호로 **이중 백틱**을 씁니다:

```
(A) by inserting ``and'' at the end of paragraph (6)
```

문제가 된 holdout의 입력에는 **백틱이 79개** 있었습니다. 입력 미리보기를 `<sub>...</sub>`로 넣었더니 그 안의 텍스트가 **마크다운으로 해석**되면서 인라인 코드스팬이 열리고, 뒤따르는 `**PREDICTION**` 블록까지 삼켜 버렸습니다.

`html.escape()`로는 막을 수 없습니다 — `<`, `>`, `&`만 변환하고 **백틱·`*`·`_`는 그대로 통과**시킵니다:

```python
html.escape("``and''")   # → "``and''"   (그대로!)
```

### 대응

사용자 데이터를 노트북 마크다운에 넣을 때는 **`<pre>`로 감쌉니다**(그 안은 마크다운이 비활성). `common/display_utils.py`가 입력 미리보기·전문·평문 예측 모두 이 방식으로 렌더합니다.

같은 이유로 **평문 예측을 `> 인용문`으로 감싸는 것도 위험합니다** — 요약이 원문의 `` ``인용'' ``을 그대로 옮기거나 `>`를 포함하면 깨집니다. JSON 예측만 코드펜스를 쓰고(그건 `json.dumps` 출력이라 안전), 평문은 `<pre>`로 둡니다.

> 💡 일반화하면: **모델 입출력은 신뢰할 수 없는 텍스트**입니다. 마크다운으로 렌더할 땐 항상 `<pre>`나 코드펜스 안에 두세요. 조용히 사라지는 버그라 발견이 늦습니다.

---

## §4.12. 🔴 "추론 셀이 73초" — 대부분 추론이 아닙니다 (실측 2026-07-31)

멀티모달 추론 셀 한 번이 **73초**였습니다. 단계별로 재보니 범인이 추론이 아니었습니다:

| 단계 | 시간 | 비중 |
|---|---|---|
| **`load_seed_examples(1)`** | **50.0초** | **68%** ⛔ |
| endpoint 추론 | 21.0초 | 29% |
| PNG 인코딩 + base64 | 0.35초 | <1% |

### 원인 1 — `streaming=True`가 매번 다시 받습니다 (50초)

`load_dataset(..., streaming=True)`는 **로컬 디스크에 캐시하지 않습니다.** cord-v2는 이미지가 parquet에 내장돼 있어 **첫 row 하나를 꺼내는 데 23초**가 걸리고, 셀을 다시 실행하면 **또** 그만큼 듭니다(재호출 24초 실측).

| 방식 | 첫 실행 | 재실행 |
|---|---|---|
| `streaming=True` | 24초 | **24초** (캐시 없음) |
| `split="train[:n]"` | 36초 (전량 준비) | **0.15초** ✅ |

노트북은 같은 셀을 여러 번 돌리므로 **split 슬라이스가 맞습니다.** 첫 회 36초는 캐시를 만드는 일회성 비용입니다. → 실측 결과 셀 전체가 **73초 → 22초**가 됐습니다(시드 로드 1.1초).

**더 나은 방법 — 검증용 이미지는 리포에 둡니다.** 배포 스모크는 이미지 1~2장이면 충분한데, 그걸 위해 매번 데이터셋을 건드릴 이유가 없습니다. `tracks/05_multimodal_extraction/samples/`에 영수증 2장 + 정답 JSON을 넣고 `track_data.load_sample_receipts()`로 읽습니다(**0.03초**). cord-v2는 CC BY 4.0이라 출처 표기 시 재배포가 가능합니다.

샘플을 고를 때도 **생성 토큰 수를 봤습니다** — 원본 `train[0]`은 메뉴가 22개(592토큰)라 추론만 24초입니다. 메뉴 3개(80토큰)인 것을 기본으로 두니 **추론 5.3초**가 됐습니다.

| | 시드 로드 | 추론 | 합계 |
|---|---|---|---|
| 처음 (streaming + `train[0]`) | 50.0초 | 21.0초 | **73초** |
| 지금 (samples + 짧은 영수증) | **0.03초** | **5.3초** | **5.3초** |

> ⚠️ 반대로 **학습 컨테이너**처럼 "한 번만 읽고 버리는" 환경에서는 streaming이 맞습니다(디스크·시간 절약). 캐시가 재사용되는지로 판단하세요.

### 원인 2 — 남은 21초는 정상입니다 (그리고 잘리고 있었습니다)

payload 크기는 무관했습니다 — 1,853KB(PNG) → 64KB(축소 JPEG)로 **29배 줄여도 21.0s → 20.6s**로 그대로입니다. 병목은 **생성 토큰 수**입니다(실측 **약 40ms/토큰**, L4):

| `max_tokens` | 소요 | `finish_reason` |
|---|---|---|
| 128 | 5.3초 | `length` ⛔ |
| 512 | 20.6초 | `length` ⛔ |
| 768 | 21.3초 | `stop` ✅ |

즉 **512로는 이 영수증이 잘리고 있었습니다.** cord-v2 정답 JSON 길이를 100건 재보니 median은 56토큰인데 **최대 592토큰**이고, 하필 노트북이 데모로 쓰는 첫 시드가 그 긴 케이스였습니다.

| `max_tokens` | 잘리는 비율(100건) |
|---|---|
| 256 / 512 | 2.0% |
| **768** | **0%** ✅ |

→ 멀티모달 셀은 `max_tokens=768`로 올렸습니다. 이미지도 JPEG(q85)로 보내 payload를 8배 줄였습니다(속도는 같지만 전송이 가벼움).

> 💡 **교훈**: "느리다"를 추론 탓으로 단정하지 말고 **단계별로 재세요**. 여기서는 68%가 데이터 로드였고, 게다가 진짜 문제(응답 절단)는 시간과 무관하게 숨어 있었습니다.

---

## §4.13. 🔴 `%store`는 트랙을 넘어 공유됩니다 — 엉뚱한 엔드포인트 호출 (실측 2026-07-31)

### 증상

요약 트랙에서 추론했는데 이런 400 에러가 났습니다:

```
ModelError: ... "This model's maximum context length is 2048 tokens.
However, you requested 512 output tokens and your prompt contains at least
1537 input tokens, for a total of at least 2049 tokens."
```

**그런데 요약 엔드포인트의 `max_model_len`은 4096입니다** — 2049는 문제될 값이 아닙니다.

### 원인 — 다른 트랙이 `endpoint_name`을 덮어씀

에러 메시지에 붙어 오는 CloudWatch 링크가 결정적 단서입니다:

```
.../Endpoints/gemma-mm-extraction-vllm-1785498368   ← 멀티모달 엔드포인트!
```

`%store`는 IPython의 **전역** 저장소로, 트랙·커널·리전을 넘어 값이 유지됩니다. 멀티모달 트랙(`max_model_len=2048`)을 배포한 뒤 요약 노트북을 열면 `%store -r endpoint_name`이 **멀티모달 엔드포인트 이름**을 복구해 옵니다.

| 엔드포인트 | `max_model_len` | 2049 토큰 |
|---|---|---|
| 요약 (의도한 것) | 4096 | ✅ 여유 |
| 멀티모달 (실제 호출된 것) | 2048 | ⛔ 1토큰 초과 |

**진단이 어려운 이유**: 에러가 "context length"를 말하니 `max_tokens`나 `serve_max_model_len` 설정을 의심하게 됩니다. 실제 문제는 **엔드포인트를 잘못 골랐다**는 것입니다.

### 대응 — 트랙별로 키를 분리

트랙마다 고유 키(`ep_<track_key>`)로도 저장하고, 복구할 때 **그 키를 우선**합니다:

```python
# 저장 (배포 직후)
ep_summarization = endpoint_name
%store endpoint_name          # 하위호환용 전역
%store ep_summarization       # 트랙 전용 — 충돌 불가

# 복구
%store -r ep_summarization
%store -r endpoint_name
endpoint_name = globals().get('ep_summarization') or globals().get('endpoint_name')
assert endpoint_name, 'endpoint_name 이 없습니다 — 03의 배포 셀을 먼저 실행하세요.'
print('사용할 endpoint:', endpoint_name)     # 무엇을 부르는지 항상 눈에 보이게
```

키가 `ep_extraction` / `ep_classification` / `ep_summarization` / `ep_domain_qa` / `ep_mm_extraction`로 갈리므로 **여러 트랙을 병행해도 섞이지 않습니다.** 실제 커널에서 전역을 멀티모달로 오염시킨 뒤 요약 복구를 실행해 올바른 엔드포인트가 선택됨을 확인했습니다.

> 💡 같은 이유로 `train_path`는 `%store`를 **아예 쓰지 않고** 트랙 로컬 파일(`data/train.jsonl`)을 직접 씁니다. `%store`에 담아야 할 값은 "이 트랙 것"임을 이름에 새기세요.

---

## §4.6. 응답 스트리밍 — vLLM 경로에서는 **됩니다** (실측 2026-07-31, E4B 포함)

> ⚠️ **이 절은 갱신됐습니다.** 이전 버전은 "E4B는 스트리밍 불가"라고 썼는데, 그건 **HF PyTorch Inference DLC를 쓰던 시절**의 결론입니다. 이 킷의 서빙 경로가 vLLM/SGLang/LMI 셋으로 바뀐 뒤 **E4B에서 토큰 스트리밍이 정상 동작함을 실측 확인**했습니다.

**결론: vLLM DLC로 서빙하는 E4B에서 SSE 토큰 스트리밍이 됩니다.**

### 실측 (요약 트랙 endpoint, vLLM 0.26.0, ml.g6.2xlarge, 입력 5,996자)

| 방식 | 첫 응답 | 완료 | 조각 수 |
|---|---|---|---|
| `invoke_endpoint_with_response_stream` (`stream: true`) | **0.42초** | 15.9초 | 391 |
| `invoke_endpoint` (완성 대기) | 16.16초 | 16.2초 | 1 |

**첫 응답 체감 38배**. 요약처럼 응답이 긴 트랙에서 차이가 큽니다.

### 🔴 청크 경계는 SSE 줄 경계와 일치하지 않습니다 — 파싱 함정

가장 걸리기 쉬운 함정입니다. `PayloadPart` 하나가 **JSON 중간에서 끊겨서** 옵니다(실측):

```
b'...,"finish_reason":"length",...,"system_finger'      ← 여기서 끊김
b'print":"vllm-0.26.0-67353da1"}\n\n'                   ← 나머지가 다음 청크
```

그래서 청크를 받는 즉시 `json.loads`하면 `JSONDecodeError`가 납니다. **버퍼에 모아 `\n\n`(SSE 이벤트 구분자) 단위로만 잘라 파싱**해야 합니다 — `common/aws_utils.stream_sagemaker_chat()`이 이 처리를 담당합니다:

```python
for piece in aws_utils.stream_sagemaker_chat(endpoint_name, msgs, region=REGION):
    print(piece, end='', flush=True)
```

노트북에서는 `display_utils.stream_inference()`가 `display_id`로 같은 출력 셀을 갱신해 제자리에서 흘려 보여 줍니다.

### 태스크별로 켜고 끄기

- **켤 만한 것**: 요약·도메인 QA 같은 긴 자유서술 → 이 킷은 두 트랙에서 `STREAM = True`가 기본입니다.
- **끄는 게 맞는 것**: 추출(JSON)·분류(라벨) → 응답이 **완성돼야 파싱/사용 가능**하고 애초에 짧습니다. agentic tool도 완성값을 반환해야 Claude가 소비합니다. → `STREAM = False` 기본.

### 🔴 스트리밍이 개선하지 않는 것

**전체 생성 시간과 throughput(동시 처리량)은 그대로입니다** — 첫 토큰 체감만 줄입니다. 위 실측에서도 완료 시각은 15.9s vs 16.2s로 사실상 같습니다. 동시 처리량은 `max_num_seqs`와 연속 배칭이 결정합니다(§4.9).

---

## §5. ❓ 오개념 노트 (흔한 착각 교정)

> **❓ "LMI는 vLLM과 경쟁하는 것 아닌가요? LMI를 쓰면 vLLM은 못 쓰는 것 아닌가요?"**
> 아닙니다. **LMI는 vLLM을 감싸는 컨테이너**입니다. `OPTION_ROLLING_BATCH=vllm`으로 지정하면 LMI 안에서 vLLM 엔진이
> 돕니다. 둘은 레이어가 다릅니다(엔진 vs 컨테이너). "vLLM을 쓰고 싶다"의 답이 종종 "LMI로 쓴다"가 되는 이유가 여기에 있습니다.

> **❓ "한 번 고르면 영원히 그 컨테이너에 묶이는 것 아닌가요?"**
> 아닙니다. 이 킷은 이미지 URI를 **env(`INFER_DLC_*`)로** 해석하고, 호출은 `sagemaker-runtime`으로 통일해 두었습니다.
> 따라서 LMI, 단독 vLLM, TGI 사이의 전환은 **이미지 URI(+env)와 payload 스키마를 조정하는** 문제일 뿐, 처음부터 다시 작성하는 일이 아닙니다.

> **❓ "vLLM이 제일 빠르다니까 무조건 단독 vLLM이 정답 아닌가요?"**
> 엔진 성능과 **운영 총비용**은 서로 다른 축입니다. 단독 vLLM은 최신 기능을 유연하게 쓸 수 있지만, 그 대신 이미지·SageMaker 규약·보안 패치를
> **직접** 책임져야 합니다. 관리 마찰을 줄이고 싶다면 LMI(내부 vLLM 백엔드)가 대체로 더 낫습니다. 같은 엔진을 관리형으로 쓰는 셈이기 때문입니다.

> **❓ "Serverless로 싸게 LLM을 서빙하면 되지 않나요?"**
> 현시점의 SageMaker Serverless Inference에는 **GPU가 없습니다.** 따라서 LLM/SLM에는 부적합하며, 이 킷의 기본은 real-time(GPU)입니다.

> **❓ "DLC는 관리형 잡 전용 아닌가요? DLAMI와 같은 것 아닌가요?"**
> DLC는 **워크로드 컨테이너**로 EC2/ECS/EKS 등 어디서나 실행되며, 관리형 잡 전용이 아닙니다.
> 또한 **DLAMI**(노드 호스트 이미지)와도 다른 레이어입니다. 본 문서의 LMI/TGI는 모두 DLC로 배포되는 컨테이너입니다.

> **❓ "SageMaker 배포 가드레일(blue/green·canary·rolling)이 컨테이너 기능 아닌가요?"**
> 아닙니다. 그 배포 가드레일은 **SageMaker classic endpoint의 배포 기능**이며 컨테이너 선택과는 무관합니다
> (그리고 HyperPod의 기능도 아닙니다). 컨테이너는 "무엇을 서빙하는가"의 문제이고, 가드레일은 "어떻게 롤아웃하는가"의 문제입니다.

---

## §5.5. Speculative decoding (EAGLE3 / P-EAGLE) — self-managed endpoint에서

추론 속도를 더 끌어올리고 싶다면 **speculative decoding**을 고려할 수 있습니다. 작은 draft가 여러 토큰을 미리 제안하고
target 모델이 한 번에 검증하는 기법으로, 같은 출력 품질에서 throughput을 높입니다. **EAGLE3**는 별도의 draft LLM
대신 target 모델의 hidden-state를 재사용하는 경량 draft head를 쓰는 방식이며, **P-EAGLE**는 AWS가 여기에
parallel drafting(여러 draft 토큰을 단일 forward pass에서 동시에 예측)을 더해 upstream vLLM에 기여한 확장입니다.

> ⚠️ 아래는 라이브 검증 2026-07 기준이며, vLLM 버전·config 키·지원 head는 빠르게 바뀝니다. 배포 전 재확인하세요.

**핵심 사실 — JumpStart 전용이 아닙니다.** AWS 블로그의 "P-EAGLE on SageMaker"는 **JumpStart 원클릭** 경험을
소개하지만, speculative decoding을 켜는 **config 자체는 container-level 기능**이라 이 킷처럼 JumpStart를 쓰지 않는
self-managed endpoint(DJL LMI · vLLM DLC)에서도 설정할 수 있습니다.

| 컨테이너 | speculative decoding 설정 키 | 비고 |
|---|---|---|
| **DJL LMI** (vLLM backend) | `OPTION_SPECULATIVE_CONFIG` (JSON) 또는 `serving.properties`의 `option.speculative_config` | 공식 문서화, EAGLE3 예제 존재 |
| **vLLM DLC** | `SM_VLLM_SPECULATIVE_CONFIG` → `--speculative-config`로 매핑 | `SM_VLLM_*` env는 일반 컨테이너 규칙 |
| **HF TGI DLC** | ❌ 이 키 미지원 | TGI는 다른 엔진(자체 메커니즘) |

설정 예시(EAGLE3 / P-EAGLE):
```json
{"method": "eagle3", "model": "<eagle3-draft-head>", "num_speculative_tokens": 3, "parallel_drafting": true}
```
`parallel_drafting: true`가 P-EAGLE 경로를 켭니다(upstream vLLM `SpeculativeConfig.parallel_drafting` 필드).

> ### 🔴 가장 큰 실무 장벽 — draft head가 있어야 합니다
> speculative decoding은 **config 키만 넣는다고 동작하지 않습니다.** target 모델에 맞춰 **학습된 draft head 체크포인트**가
> 반드시 있어야 하며, `parallel_drafting`은 그 목적에 맞게 학습된 head를 추가로 요구합니다.
>
> - **AWS가 공개한 P-EAGLE head**: GPT-OSS-120B/20B, Qwen3-Coder-30B 뿐입니다. **Gemma용은 공개돼 있지 않습니다**
>   (JumpStart의 Gemma-4-31B-IT용 head는 배포 시 내부 번들되는 아티팩트로 공개 다운로드 대상이 아닙니다).
> - **커뮤니티 Gemma EAGLE3 head**는 존재합니다 — 예: `RedHatAI/gemma-4-31B-it-speculator.eagle3`,
>   `BCCard/MoAI-gemma-4-12B-it-speculator.eagle3`, `planethunter98/eagle3-head-gemma3-12b-it`(모두 실행 전 model card로 검증).
>   다만 이들은 **base/instruct 모델용**입니다.
> - 🔴 **이 킷은 Gemma를 fine-tune합니다.** EAGLE3 head는 target의 hidden-state에 맞춰 학습되므로, base용 head를
>   fine-tuned 모델에 그대로 쓰면 acceptance rate가 떨어질 수 있습니다. 따라서 (a) 커뮤니티 head를 쓰되 반드시
>   자체 벤치마크로 acceptance rate를 실측하거나, (b) 자신의 fine-tuned Gemma에 맞는 EAGLE3 head를 직접 학습해야 합니다.

**이 킷에서의 위치**: `03_deploy_endpoint`의 vLLM(B) 경로에서 `SM_VLLM_SPECULATIVE_CONFIG`(또는 LMI의 `OPTION_SPECULATIVE_CONFIG`)로
켤 수 있도록 자리를 만들어 두었습니다. 다만 위 head 요건 때문에 **기본값은 비활성(`RUN_VLLM=False`, `EAGLE3_SPECULATOR=''`)**이며,
head를 확보해 정합성을 검증한 뒤에 켜는 것을 권장합니다.

---

## §6. 🔴 비용 / cleanup 주의
- **real-time endpoint는 삭제하기 전까지 시간당 GPU 요금이 부과됩니다.** 어떤 컨테이너를 골랐든 마찬가지이므로,
  실습이 끝나면 반드시 `99_cleanup`(또는 `predictor.delete_endpoint()`)을 실행하세요.
- 단독 vLLM을 **BYOC**로 쓰면 ECR 이미지 저장 비용과 빌드/푸시 시간(운영 부담)이 추가로 든다는 점도 감안하세요.
- 배포하거나 호출한 직후에 `common/aws_utils.cw_links()`가 CloudWatch/콘솔 다이렉트 링크를 출력해 주므로,
  로그를 보면서 컨테이너 기동, OOM, 백엔드 로딩 상태를 확인할 수 있습니다.

---

## §7. 근거 (라이브 검증 2026-07)

> 🔴 아래는 특정 시점의 스냅샷입니다. 이미지 태그, `OPTION_*` 키, 지원 엔진, SDK framework 문자열은 자주 바뀌므로
> **실행 전에 재확인**하세요. AWS 마케팅 수치("처리량 몇 배" 등)는 원문에서 "AWS claim" 표현으로 확인하시기 바랍니다.

| 주제 | URL |
|---|---|
| SageMaker 추론 개요(4옵션: real-time/serverless/async/batch) | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html |
| Serverless Inference(현시점 GPU 미지원 확인용) | https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html |
| LMI(Large Model Inference) 컨테이너 문서 | https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html |
| LMI 구성/`OPTION_*`·`serving.properties` 튜닝 | https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html |
| DLC available_images (계정 763104351884·현행 태그) | https://aws.github.io/deep-learning-containers/reference/available_images/ |
| SDK 이미지 resolve(`image_uris.retrieve`) | https://sagemaker.readthedocs.io/en/stable/api/utility/image_uris.html |
| BYOC 규약(`/invocations`+`/ping`) | https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html |
| 스트리밍 호출(`invoke_endpoint_with_response_stream`) | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointWithResponseStream.html |
| DJL Serving 공식 저장소(GitHub) | https://github.com/deepjavalibrary/djl-serving |
| AWS Deep Learning Containers 저장소(GitHub) | https://github.com/aws/deep-learning-containers |
| vLLM 엔진/OpenAI 호환 서버(GitHub) | https://github.com/vllm-project/vllm |
| §4.9 LMI `OPTION_MAX_ROLLING_BATCH_SIZE`/`OPTION_GPU_MEMORY_UTILIZATION` + pass-through 근거 | https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html |
| §4.9 SGLang `--max-running-requests`/`--mem-fraction-static` 플래그 확인(server_args.py) | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py |
| HuggingFace TGI(GitHub) | https://github.com/huggingface/text-generation-inference |
| SageMaker HuggingFace Inference Toolkit(GitHub) | https://github.com/aws/sagemaker-huggingface-inference-toolkit |

**킷 내 참조 파일**: `common/dlc.py`(이미지 URI 해석·env 우선) · `common/config.py`(HF DLC 버전 핀·인스턴스) ·
`common/aws_utils.py`(`invoke_sagemaker_endpoint`·서비스 경계) · `common/llm_gateway.py`(LiteLLM 통합) ·
`tracks/*/03_deploy_endpoint.ipynb`(배포 지점) · `tracks/*/99_cleanup.ipynb`(과금 중단).

**내비게이션**: (로컬에 있으면) ← `04_.md` · `README.md` / → `06_.md`
*(문서 세트가 로컬에 함께 있을 때만 유효한 상대 링크)*
