# 01 · SageMaker 추론(Inference) 완전 가이드 — 이 킷의 앵커 문서

> **독자**: SageMaker로 파인튜닝한 SLM(Gemma)을 "어떻게 서빙하나"가 궁금한 초심자~중급자.
> HyperPod/EC2는 몰라도 됩니다. 필요한 대비만 표로 짚습니다.
> **⚠️ 주의**: 🔴 표시된 값(모델 ID·DLC 이미지 태그·SDK 버전·리전·서비스 한도)은 **빠르게 바뀝니다**.
> 이 문서의 구체 수치·태그는 전부 "실행 전 재확인" 대상입니다. 계정 ID/시크릿/절대경로는 하드코딩하지 마세요.
> **라이브 검증 2026-07** (근거 표는 문서 맨 끝에 있습니다). fact-critical 항목은 aws-fact-checker에 검증을 의뢰했습니다(핸드오프 노트를 참조하세요).

이 문서는 이 킷의 **추론 앵커 문서**입니다. 다른 가이드(학습·agentic·평가)는 "endpoint가 무엇인지"를 설명할 때 이 문서로 링크를 겁니다.

---

## §0. TL;DR (한 줄)

**SageMaker 추론에는 4가지 옵션(Real-time / Serverless / Asynchronous / Batch Transform)이 있고, LLM/SLM 서빙에는 GPU가 붙는 Real-time이 사실상 유일한 선택입니다. 이 킷은 Real-time endpoint + (HF TGI 또는 DJL LMI) 서빙 컨테이너로 배포하며, 호출은 `sagemaker-runtime`(Bedrock과 별개 서비스), 정리는 반드시 `99_cleanup`입니다.**

정리하면 다음과 같습니다.
1. **⚠️ Serverless Inference에는 GPU가 없습니다.** 따라서 LLM/SLM에는 부적합하며, 이것이 이 킷이 Real-time을 선택한 근본 이유입니다.
2. **endpoint 호출은 Bedrock 호출과 다릅니다.** endpoint는 `boto3.client("sagemaker-runtime").invoke_endpoint()`로 부르고, Bedrock Claude는 `bedrock-runtime.converse()`로 부릅니다. 즉 **별개 서비스이고 별개 클라이언트**입니다.
3. **서빙 컨테이너는 "하나로 모든 상황을 해결"할 수 있는 것이 아닙니다.** HF TGI, DJL LMI(내부 vLLM·TensorRT-LLM 백엔드), vLLM 단독 중에서 상황에 맞게 골라야 합니다.
4. **Real-time은 삭제하기 전까지 시간당(GPU) 요금이 계속 부과됩니다.** 그러므로 실습이 끝나면 반드시 `99_cleanup.ipynb`를 실행하세요.

---

## §0.5. 기존 Pain Point (지금 겪는 혼란)

파인튜닝까지 끝낸 초심자가 배포 단계에서 실제로 자주 막히는 지점은 다음과 같습니다.

- "endpoint 종류가 4개나 되는데 **무엇을 골라야 할까요?**" — 문서마다 이름만 나열할 뿐, *언제 무엇을* 써야 하는지는 알려주지 않습니다.
- "Bedrock은 그냥 API로 부르던데, **내 endpoint도 Bedrock으로 부르는 걸까요?**" — 아닙니다. 완전히 다른 서비스입니다.
- "**Serverless가 제일 싸 보이는데** 왜 쓰지 않을까요?" — GPU가 없어서 LLM이 돌아가지 않기 때문입니다. 이 사실을 모르고 골랐다가 배포에 실패하는 경우가 많습니다.
- "서빙 컨테이너가 TGI, LMI, vLLM으로 여러 개인데, **어느 것을 model_data에 물려야 할까요?**"
- "테스트만 했을 뿐인데 **다음 날 청구서가 날아왔어요.**" — Real-time endpoint를 지우지 않았기 때문입니다.

이 문서는 위 다섯 가지 고민을 순서대로 해소해 드립니다.

---

## §1. 왜(Why) Real-time인가 — 4옵션 대조표 + 비유 + 기술적 차이 3가지

### 쉽게 말하면
추론 옵션 4개는 결국 **"손님이 어떤 방식으로 찾아오는가"**에 대한 답이라고 이해하시면 됩니다.
- 손님이 **실시간 채팅**처럼 찾아온다면, 항상 켜 두는 창구(**Real-time**)가 필요합니다.
- 손님이 **가끔 뜸하게** 찾아온다면, 부를 때만 여는 창구(**Serverless**)를 쓸 수 있습니다. 다만 이 창구에는 **GPU 조리기구가 없다**는 점에 유의하세요.
- 손님이 **큰 서류뭉치**를 맡기고 나중에 찾아간다면, 접수한 뒤 처리하는 방식(**Asynchronous**)이 적합합니다.
- 손님이 **하룻밤에 수만 건**을 한 번에 맡긴다면, 야간 일괄처리(**Batch Transform**)로 소화합니다.

LLM/SLM은 **GPU 조리기구가 반드시** 필요한데, 그것을 갖춘 창구는 Real-time, Async, Batch뿐입니다. 이 가운데 **대화형 실시간 응답**이 목적이라면 Real-time이 정답입니다.

### 대조표 — SageMaker 추론 4옵션

| 축 | Real-time | Serverless | Asynchronous | Batch Transform |
|---|---|---|---|---|
| **용도** | 상시 실시간 응답(챗·API) | 간헐적/예측불가 트래픽 | 대용량 payload·긴 처리 | 데이터셋 일괄 추론 |
| **상주 리소스** | 있음(항상 켜짐) | 없음(요청 시 기동) | 있음(오토스케일→0 가능) | 잡 단위(끝나면 종료) |
| **레이턴시** | 최저(ms~수초) | cold start 지연 有 | 비동기(큐→S3 결과) | 배치 완료까지 |
| **Cold start** | 없음(웜) | 🔴 **있음** | 스케일업 시 有 | N/A(잡) |
| **GPU 지원** | ✅ 있음 | ⚠️ **없음(CPU 전용)** | ✅ 있음 | ✅ 있음 |
| **비용 모델** | 인스턴스 시간당(삭제 전까지) | 요청+실행시간(유휴 0) | 인스턴스 시간(0까지 축소 가능) | 잡 실행 시간만 |
| **payload/timeout** | 상대적으로 작음·짧음 | 작음·짧음 | 큰 payload·긴 timeout 허용 | 대용량 파일 |
| **LLM/SLM 적합** | ✅ **적합(이 킷)** | ❌ **부적합(GPU 없음)** | △ 긴 생성·오프라인 | △ 대량 오프라인 채점 |

> 🔴 payload 크기, timeout, cold-start 시간, 오토스케일 축소 최솟값 같은 **구체적인 수치는 서비스 한도**에 해당하므로 자주 바뀝니다. 실제 값은 아래 근거 표의 공식 문서에서 **실행 전에 반드시 재확인**하세요. (이 표는 각 옵션의 "성격"을 잡기 위한 것입니다.)

### 기술적 차이 3가지 (왜 이렇게 나뉘는가)
1. **인스턴스 상주 여부**: Real-time과 Async는 프로비저닝된 인스턴스가 붙어 있지만, Serverless는 요청이 들어오는 순간에 용량을 할당합니다. 이 때문에 cold start가 생기고, **가속기(GPU)를 상주 형태로 붙일 수 없어서** GPU를 지원하지 못합니다.
2. **요청-응답 채널**: Real-time은 동기 응답(HTTP)을 주고, Async는 **S3 입력 → 큐 → S3 출력** 방식으로 비동기 처리하며, Batch는 S3 데이터셋을 잡 단위로 훑습니다. 이렇게 채널이 다르기 때문에 payload/timeout 한도도 서로 다릅니다.
3. **스케일 바닥값**: Async는 인스턴스를 0까지 축소할 수 있어 유휴 비용을 줄일 수 있지만, Real-time은 (오토스케일을 걸더라도) 통상 1대 이상을 유지합니다. 이것이 바로 **상시 과금**이 발생하는 원인입니다.

> ### ❓ 오개념 노트 — "Serverless가 제일 싸니까 LLM도 Serverless로 하면 되지 않나요?"
> **아닙니다.** SageMaker **Serverless Inference에는 GPU가 없습니다(현재 기준, CPU 전용).** Gemma 같은 SLM/LLM은 GPU 없이는 사실상 돌아가지 않거나, 돌아가더라도 실용 불가 수준으로 느립니다. "간헐적인 트래픽이니 Serverless가 저렴하겠다"는 판단은 CPU 모델(작은 임베딩이나 전통 ML)에나 맞는 이야기입니다. 요컨대 **LLM/SLM에는 Real-time(또는 GPU가 붙는 Async/Batch)을 쓰세요.** 🔴 다만 "Serverless GPU 미지원"은 정책성 항목이라 언젠가 바뀔 수 있으니, 배포 전에 재확인하시기 바랍니다.

**언제 무엇을 고르면 되는가 (조건부 권고)**
- 대화형·실시간 API가 목표이고 트래픽이 꾸준하다면 → **Real-time**을 선택하세요 (이 킷의 선택입니다).
- CPU로 충분한 작은 모델이면서 트래픽이 뜸하고 cold start를 감수할 수 있다면 → **Serverless**가 적합합니다.
- 입력이 크고(수십 MB) 응답까지 오래 걸려도 되며 비동기로 받고 싶다면 → **Asynchronous**를 쓰세요.
- 상시 endpoint 없이 대량 데이터셋을 한 번에 채점하거나 생성하려면 → **Batch Transform**을 고르세요 (예: 평가셋 오프라인 추론).

---

## §2. Endpoint 해부학 — Model → EndpointConfig → Endpoint

### 쉽게 말하면
endpoint 하나가 뜨기까지는 **3층짜리 레고**가 차곡차곡 쌓인다고 생각하시면 됩니다.
- **Model**: "어떤 가중치(model_data, S3)를 + 어떤 컨테이너(이미지)로 로드할 것인가"를 정의합니다.
- **EndpointConfig**: "그 Model을 + 어떤 인스턴스로 + 몇 대(variant)로 + 트래픽 비율은 어떻게" 배치할지 담은 설계도입니다.
- **Endpoint**: 위 설계도를 바탕으로 실제로 **떠 있는 상시 서버**이며, 곧 과금이 시작되는 지점입니다.

```
[S3: model_data tar.gz]      [ECR: 서빙 DLC 이미지]
   (파인튜닝 merged 가중치)        (763104351884.dkr.ecr...)
            \                       /
             v                     v
        ┌──────────── Model ────────────┐   ← "무엇을 어떤 컨테이너로"
        └───────────────┬───────────────┘
                        v
        ┌──────── EndpointConfig ────────┐   ← "어떤 인스턴스 · 몇 대 · variant 트래픽"
        └───────────────┬───────────────┘
                        v
        ┌────────────  Endpoint  ────────┐   ← 🔴 상시 서버 = 과금 시작
        └───────────────┬───────────────┘
                        v
   client.invoke_endpoint(EndpointName=..., Body=...)   ← sagemaker-runtime
```

> **왜 3층으로 나뉘어 있을까요?** 이렇게 분리되어 있는 덕분에 **무중단 배포**(새 EndpointConfig로 교체), **A/B 테스트(production variant)**, **오토스케일**이 가능해집니다. 다만 이런 배포 가드레일(blue/green·canary·rolling)은 **SageMaker "클래식" endpoint의 기능**이지 HyperPod의 기능이 아니라는 점에 유의하세요(§오개념 노트 참조).

### invoke_endpoint는 어떻게 동작하는가
호출은 **별도의 런타임 서비스**(`sagemaker-runtime`)로 전달됩니다. 이 킷의 `common/aws_utils.invoke_sagemaker_endpoint()`가 그 얇은 래퍼 역할을 합니다.

```python
# common/aws_utils.py (요지) — 실제 payload 스키마는 서빙 컨테이너에 맞춰 조정
client = boto3.client("sagemaker-runtime", region_name=region)
resp = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps({"inputs": prompt, "parameters": {...}}),
)
```
- `{"inputs", "parameters"}`는 HF TGI/DJL의 **일반 텍스트 생성 스키마**를 가정한 것입니다. 컨테이너가 OpenAI 호환(messages) 스키마를 쓴다면 payload가 달라지므로, 배포한 컨테이너 문서에 맞춰 조정하세요.
- 응답 파싱도 방어적으로 해야 합니다. TGI는 `[{"generated_text": ...}]`, DJL은 `{"generated_text": ...}` 형태로, 컨테이너마다 응답 구조가 다르기 때문입니다.

### 스트리밍 — invoke_endpoint_with_response_stream
토큰을 한 번에 받지 않고 **흘려받고 싶다면**(챗 UX에 유용합니다) 동일한 `sagemaker-runtime`의 **`invoke_endpoint_with_response_stream()`**를 사용하세요. 이때 컨테이너(TGI/LMI)가 스트리밍 응답을 지원해야 하며, 응답은 이벤트 스트림 형태로 청크 단위로 도착합니다.

> 🔴 스트리밍 payload 필드(예: `"stream": true`)와 이벤트 스트림 파싱 방식은 **컨테이너·SDK 버전에 따라 달라집니다.** 실행 전에 해당 DLC 문서에서 재확인하세요.

### 🔴 서비스 경계 — endpoint vs Bedrock (혼동 금지)

| | SageMaker Endpoint (내 파인튜닝 SLM) | Bedrock (Claude 등 관리형 LLM) |
|---|---|---|
| boto3 클라이언트 | `sagemaker-runtime` | `bedrock-runtime` |
| 호출 API | `invoke_endpoint` / `invoke_endpoint_with_response_stream` | `converse` / `converse_stream` |
| 리소스 | 내가 띄운 상시 endpoint | 관리형(상주 리소스 없음) |
| 가중치 | 내 model_data(S3) | AWS가 호스팅 |
| 모델 지정 | `EndpointName` | `modelId`(inference-profile prefix 필요) |
| 이 킷의 위치 | 파인튜닝 Gemma 서빙 | agentic reasoning·합성 데이터 생성 |

> ### ❓ 오개념 노트 — "내 endpoint도 Bedrock API로 부르면 되지 않나요?"
> **아닙니다.** endpoint는 `sagemaker-runtime.invoke_endpoint()`로, Bedrock은 `bedrock-runtime.converse()`로 호출하며 **완전히 다른 서비스**입니다. 이 킷은 `common/llm_gateway.py`(LiteLLM)를 통해 **두 백엔드를 하나의 인터페이스처럼** 사용하지만, 이는 내부적으로 각각 올바른 서비스로 라우팅하는 것일 뿐 같은 API를 쓰는 것이 아닙니다. (Bedrock Claude는 `converse`를 쓰며, inference-profile prefix `us./eu./apac./global.`가 필요하고, 모델 ID는 🔴 env로 주입해야 하며 하드코딩하면 안 됩니다 — `common/config.BEDROCK_CLAUDE_MODEL_ID`를 참고하세요.)

---

## §3. 서빙 컨테이너 — HF TGI vs DJL LMI vs vLLM 단독

### 쉽게 말하면
endpoint는 "빈 GPU 서버"에 **서빙 컨테이너(요리사)**를 넣어야 비로소 모델을 돌릴 수 있습니다. 요리사 후보는 다음과 같습니다.
- **HF TGI** (Text Generation Inference): HuggingFace가 만든 텍스트 생성 전용 서버입니다. `HuggingFaceModel`로 손쉽게 붙일 수 있습니다.
- **DJL LMI** (Deep Java Library — Large Model Inference): **AWS가 만든** 서빙 컨테이너입니다. **내부 백엔드를 vLLM 또는 TensorRT-LLM 중에서 선택**할 수 있으며(환경변수 `OPTION_ROLLING_BATCH` 등으로 지정), 대형 모델과 고처리량 상황에 강합니다.
- **vLLM 단독**: vLLM 엔진을 직접 띄우는 방식으로, BYOC 커스텀 컨테이너나 vLLM OpenAI 호환 서버로 구성합니다. 최신 vLLM 기능을 바로 쓰고 싶을 때 적합합니다.

### 대조표 — 서빙 컨테이너 선택

| | HF TGI | DJL LMI (AWS) | vLLM 단독 |
|---|---|---|---|
| 제공 주체 | HuggingFace | **AWS** | vLLM 커뮤니티 |
| 내부 백엔드 | TGI 엔진 | 🔴 **vLLM / TensorRT-LLM 선택**(`OPTION_ROLLING_BATCH`) | vLLM |
| SageMaker 연동 | `HuggingFaceModel`로 간단 | LMI 설정(serving.properties/env) | BYOC 또는 vLLM OpenAI 서버 |
| 강점 | 빠른 시작·HF 친화 | 대형·고처리량·백엔드 유연 | 최신 vLLM 기능 즉시 |
| 페이로드 스키마 | `{inputs,parameters}` | 컨테이너 설정 따라 | OpenAI 호환(messages) 흔함 |
| 언제 | SLM 빠른 배포 | 처리량·대형·튜닝 필요 | vLLM 특정 기능 필수 |

> 🔴 "DJL LMI가 어떤 vLLM/TRT-LLM 버전을 내장하는가", "TGI가 특정 모델 아키텍처를 지원하는가"는 모두 **이미지 태그 버전에 달려 있습니다.** available_images에서 태그를 확인한 뒤 재검증하세요.

### model_data는 어떻게 로드되는가
- 학습(`tracks/*/scripts/train.py`, TRL `SFTTrainer` + PEFT LoRA)이 끝나면 **merged 가중치**를 `tar.gz`로 묶어 S3에 올립니다. 이것이 바로 **`model_data`**입니다.
- 배포할 때 Model이 이 S3 아티팩트를 컨테이너 안에 풀어 놓으면, 서빙 엔진이 이를 로드합니다.
- 이 킷의 `03_deploy_endpoint` 노트북은 앞 단계에서 `%store`로 저장해 둔 `model_data`를 받아 `HuggingFaceModel(model_data=..., image_uri=...)`에 물립니다.

> Gemma를 서빙할 때 주의할 점입니다(라이브 검증 항목이며 🔴 실행 전 재확인이 필요합니다). chat template은 `apply_chat_template`에 위임합니다(이 킷의 `common/gemma_format.py`). **Gemma는 system role을 거부**하므로 첫 user 턴으로 fold해야 합니다. dtype는 **bf16**를 쓰고 fp16은 사용하지 마세요. LoRA는 all-linear에 `modules_to_save=[lm_head, embed_tokens]`를 함께 지정합니다. 학습에 관한 상세 내용은 `tracks/*/scripts/train.py`를 참고하세요.

### DLC 이미지 URI 패턴 (이 킷의 `common/dlc.py`)
SageMaker 서빙/학습 컨테이너(DLC) 이미지는 AWS ECR에 올라가 있으며, **URI 패턴이 고정**되어 있습니다.

```
763104351884.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
└── ECR 계정(대부분 리전 공용)                    └─🔴 태그는 자주 바뀜
```
- `common/dlc.py`는 이 패턴으로 URI를 조립하되, **env(`DLC_IMAGE_URI` 또는 `DLC_REPOSITORY`+`DLC_TAG`, 추론은 `INFER_*`를 우선 적용)**로 오버라이드할 수 있게 설계했습니다. 덕분에 available_images 페이지가 갱신되어도 **코드를 고칠 필요 없이 env만** 바꾸면 됩니다.
- env가 지정되지 않으면 `None`을 반환하고, 노트북은 `transformers_version`/`pytorch_version` 조합으로 폴백합니다. 다만 이 경우 구식 리스트로 resolve될 위험이 있으므로 **태그를 직접 지정하시길 권장**합니다.

> 🔴 **repository/tag는 실행 전에 [available_images](https://aws.github.io/deep-learning-containers/reference/available_images/)에서 반드시 재확인**하세요. `common/dlc.py`와 `common/config.py`의 기본 태그 문자열은 "예시(`# TODO verify`)"일 뿐이며 현행 값이 아닐 수 있습니다.

> ### ❓ 오개념 노트 — "서빙 컨테이너 하나면 다 되는 것 아닌가요?"
> **아닙니다.** 컨테이너마다 **지원하는 모델, payload 스키마, 스트리밍 방식, 백엔드**가 다릅니다. 같은 Gemma라도 TGI에 물릴 때와 DJL LMI에 물릴 때 **payload가 서로 다릅니다.** `invoke_sagemaker_endpoint()`가 응답을 **방어적으로 파싱**하는 이유가 바로 여기에 있습니다. "한 컨테이너로 모든 것을 해결한다"가 아니라, **모델·처리량·기능 요구에 맞춰 선택**해야 한다는 점을 기억하세요.
>
> 참고로 **DLAMI와 DLC는 다릅니다.** DLAMI는 노드(호스트) 머신 이미지이고, DLC는 그 위에서 도는 **워크로드 컨테이너**입니다. 그리고 DLC는 관리형 잡 전용이 아니라 **EC2/ECS/EKS(HyperPod-EKS 포함)** 어디서나 실행됩니다.

---

## §4. 이 킷은 어떻게 배포하나 (03_deploy_endpoint)

### 쉽게 말하면
`03_deploy_endpoint` 노트북은 **"S3의 학습 결과 → 상시 endpoint → 바로 호출 테스트"**까지를 한 번에 처리해 줍니다.

```
02_train_sft_sagemaker (%store model_data)
        │
        v
03_deploy_endpoint:
   dlc.resolve_inference_image(region)  ──env 있으면 그 태그, 없으면 버전 폴백
        │
        v
   HuggingFaceModel(model_data, image_uri=...).deploy(
        instance_type=config.INFER_INSTANCE_TYPE,  # 기본 ml.g5.2xlarge (GPU)
        initial_instance_count=1)
        │
        v
   aws_utils.invoke_sagemaker_endpoint(EP, prompt, ...)   # 즉시 호출 테스트
   aws_utils.cw_links(region, endpoint_name=EP)           # CloudWatch 링크 출력
```

- **컨테이너 선택**: 노트북의 기본 경로는 `HuggingFaceModel`(HF TGI DLC)에 `dlc.resolve_inference_image` 오버라이드를 결합한 것입니다. **DJL LMI로 vLLM/TensorRT-LLM 백엔드**를 쓰고 싶다면, LMI 이미지를 `INFER_DLC_IMAGE_URI`로 주입하고 LMI 설정(`OPTION_ROLLING_BATCH` 등)을 붙이는 식으로 확장하면 됩니다. **vLLM 단독**은 BYOC 경로로 구성합니다.
  🔴 이 킷의 "기본 컨테이너"는 코드 작성 시점을 기준으로 한 것이며, 처리량 요구에 따라 LMI나 vLLM으로 바꿀 수 있습니다. 실제 이미지는 노트북과 `common/dlc.py`에서 확인하세요.
- **인스턴스**: `config.INFER_INSTANCE_TYPE`을 사용합니다(기본값은 GPU 인스턴스인 `ml.g5.2xlarge`입니다). SLM LoRA 서빙에 합리적인 기본값이며, 모델 크기에 따라 조정하세요.
- **호출**: 동기 응답은 `aws_utils.invoke_sagemaker_endpoint()`로 받고, 통합 인터페이스는 `common/llm_gateway.endpoint_chat()`을 씁니다(LiteLLM의 `sagemaker_chat/<ep>` 또는 `sagemaker/<ep>`+`hf_model_name` 형태).
- **관측**: `aws_utils.cw_links()`가 SageMaker 콘솔과 CloudWatch Logs(`/aws/sagemaker/Endpoints`)로 가는 바로가기 HTML을 출력해 줍니다.

> 🔴 `HuggingFaceModel`에 넘기는 `transformers_version`/`pytorch_version`/`py_version`(= `config.HF_*_VERSION`)은 반드시 **AWS가 게시한 DLC 태그 조합**이어야 합니다(임의의 최신 버전을 넣으면 안 됩니다). available_images에서 재확인하세요.

---

## §5. 비용 & Cleanup (필수) · 오토스케일 · CloudWatch

### 🔴 비용 경고 (`common/aws_utils.COST_WARNING`)
- **Real-time endpoint는 삭제하기 전까지 시간당(GPU 인스턴스) 요금이 계속 부과**됩니다. 호출이 전혀 없어도 켜져 있는 한 비용이 발생합니다. 그러므로 실습이 끝나면 **반드시 `99_cleanup.ipynb`**를 실행하거나 `predictor.delete_endpoint()`를 호출하세요.
- Training Job은 잡이 종료되면 과금이 멈춥니다(단, Managed Spot을 쓰지 않으면 on-demand 요금이 적용됩니다).
- Bedrock Converse는 **토큰(호출량) 기준으로 과금**됩니다. 상주 리소스는 없지만 대량으로 합성할 때는 비용이 누적됩니다.
- AgentCore Runtime을 배포하면 Runtime 리소스에 요금이 부과되므로, 사용하지 않을 때는 정리하세요.

### Cleanup이 실제로 지우는 것 (`99_cleanup`)
```
sm.delete_endpoint(EndpointName=...)          # 상시 서버(과금원) 제거
sm.delete_endpoint_config(EndpointConfigName=...)  # 설계도 제거
# (AgentCore 배포 시 Runtime + ECR 이미지도 별도 정리)
sm.list_endpoints()  # → "남은 endpoint: 없음 ✅" 확인
```
> **순서에 주의하세요.** Endpoint를 먼저 지워야 EndpointConfig를 지울 수 있습니다. Model 오브젝트는 과금 대상이 아니지만 함께 정리해도 무방합니다.

### 오토스케일 (선택)
- Real-time은 **application auto scaling**을 이용해 인스턴스 수를 트래픽에 맞춰 조절할 수 있습니다(예: `InvocationsPerInstance`를 타깃으로 지정). 다만 통상 **최소 1대 이상**을 유지하므로 완전히 0으로 축소되지는 않습니다. 결국 "쓰지 않으면 삭제한다"가 비용 관리의 핵심입니다.
- 🔴 오토스케일 정책, 메트릭, 축소 최솟값은 재확인이 필요한 항목입니다.

### CloudWatch
- endpoint 로그는 **CloudWatch Logs `/aws/sagemaker/Endpoints`**에 쌓이며, 지표(Invocations, ModelLatency, 4XX/5XX 등)는 SageMaker 네임스페이스에서 확인할 수 있습니다.
- `aws_utils.cw_links()`가 노트북에서 클릭할 수 있는 콘솔/로그 링크를 바로 출력해 줍니다(콘솔 URL은 "현재 기준"이므로, AWS가 형식을 바꾸면 갱신이 필요합니다).

---

## §6. 다른 가이드에서 이 문서로 오는 링크(앵커)

- 학습에서 배포로 넘어갈 때: `tracks/*/scripts/train.py`와 `02_train_sft_sagemaker`를 거친 다음 이 문서의 §4로 오세요.
- 통합 호출/게이트웨이가 필요할 때: `common/llm_gateway.py`(Bedrock과 endpoint를 하나의 인터페이스로 묶습니다).
- 이미지 태그를 해석할 때: `common/dlc.py`를 참고하세요.
- 비용과 정리에 관해서는: `common/aws_utils.py`(`COST_WARNING`, `cw_links`)와 `99_cleanup.ipynb`를 보세요.
- agentic에서 endpoint를 tool로 호출할 때: `05_agentic_strands`의 `call_slm`이 이 endpoint를 부릅니다.

---

## 근거 (라이브 검증 2026-07) — 실행 전 재확인 대상

| 주제 | URL |
|---|---|
| SageMaker 추론 옵션 개요(4종) | https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html |
| Real-time endpoint 호스팅 | https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html |
| Serverless Inference (⚠️ GPU 미지원 확인) | https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html |
| Asynchronous Inference | https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html |
| Batch Transform | https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html |
| InvokeEndpoint (sagemaker-runtime) API | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html |
| InvokeEndpointWithResponseStream (스트리밍) | https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointWithResponseStream.html |
| Bedrock Converse API (별개 서비스) | https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html |
| SageMaker 오토스케일 | https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html |
| endpoint CloudWatch 지표 | https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html |
| DLC available images (이미지 태그) | https://aws.github.io/deep-learning-containers/reference/available_images/ |
| Deep Learning Containers (GitHub) | https://github.com/aws/deep-learning-containers |
| DJL LMI (AWS 서빙 컨테이너, GitHub) | https://github.com/deepjavalibrary/djl-serving |
| DJL LMI 문서(SageMaker) | https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html |
| HF TGI (GitHub) | https://github.com/huggingface/text-generation-inference |
| SageMaker Python SDK — HuggingFaceModel (GitHub) | https://github.com/aws/sagemaker-python-sdk |
| vLLM (GitHub) | https://github.com/vllm-project/vllm |
| LiteLLM (게이트웨이, GitHub) | https://github.com/BerriAI/litellm |

> **🔴 재확인이 반드시 필요한 항목**: Serverless GPU 미지원 여부, payload/timeout/cold-start 구체 수치, DLC repository/tag, SDK/컨테이너 버전, 리전 가용성, 오토스케일 최솟값입니다. 이들은 분기마다 바뀌므로 실행 전에 꼭 확인하세요.

**내비게이션**: `README.md` · `docs/` (학습·agentic·평가 가이드) · `common/` 소스.
