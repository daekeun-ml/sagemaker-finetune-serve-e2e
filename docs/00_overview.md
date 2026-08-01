# 00 · 전체 지도 — kit 구조와 실행 순서

!!! info "Scope"
    이 kit을 처음 여는 ML 엔지니어 / 데이터 과학자를 위한 지도입니다.
    SageMaker·Bedrock을 몰라도 읽을 수 있습니다.

    - **선행 조건** — 없습니다. 이 문서가 kit의 **진입점(index)**입니다
    - **여기서 다루는 것** — 무엇이 어디에 있고 어떤 순서로 도는지 · 노트북 ↔ 문서 매핑 ·
      모델·엔진 기본값 · 비용과 정리
    - **여기서 다루지 않는 것** — 개념 상세는 각 가이드 01~06으로 연결합니다

개념 설명은 각 상세 문서로 넘기고, 여기서는 "무엇이 어디에 있고 어떤 순서로 도는가"만 확정합니다.

!!! warning "빠르게 바뀌는 값"
    모델 ID·DLC 이미지 태그·SDK 버전·리전 가용성·GA 상태는 분기마다 바뀝니다. 이 문서의 태그와 버전은 전부 **실행 직전 재확인** 대상이며,
    확인처는 각 절에 인라인으로 링크한 공식 문서입니다(이미지 태그는 [available_images 페이지](https://aws.github.io/deep-learning-containers/reference/available_images/)).
    계정 ID·시크릿·절대경로는 하드코딩하지 말고 전부 env로 주입하세요.

---

## TL;DR

**태스크별로 독립 완결되는 실습 코스 5개가 얇은 `common/` 레이어를 공유하는 step-by-step 실습 kit입니다.**
각 코스는 task → 오픈 시드 데이터 → grounded 합성 → SageMaker 학습(PyTorch DLC + TRL/PEFT) → real-time endpoint(vLLM DLC 기본) → agentic(Strands → AgentCore) → held-out 평가로 이어집니다.

정리하면 다음과 같습니다.

1. **텍스트 코스 파이프라인은 7단계**입니다. 노트북 `00→06` + `99_cleanup`이 그대로 각 단계에 대응합니다([E2E 파이프라인](#e2e-파이프라인-텍스트-코스-7단계)).
2. **코스는 5개**(추출→JSON / 분류 / 요약 / 도메인-QA / 멀티모달 추출)이고 서로 독립된 E2E입니다. 공통 로직만 분리했습니다([5개 독립 코스와 공통 레이어](#5개-독립-코스와-공통-레이어)).
3. **기본 모델은 `google/gemma-4-E4B-it`**(apache-2.0 · ungated · HF 토큰 불필요)이고, `MODEL_SIZE`로 `E2B` / `E4B` / `12B` / `26B-A4B` / `31B`를 고릅니다([모델 선택](#모델-선택-gemma-4-프리셋-5종)).
4. **학습은 PyTorch DLC + TRL/PEFT(JumpStart 아님), 서빙 기본은 vLLM DLC**이며 `SERVING_ENGINE`으로 `sglang`·`lmi`로 바꿀 수 있습니다([왜 이 구조인가](#왜-이-구조인가)).
5. **실행 규율은 `DRY_RUN=1` 먼저**입니다. [파이프라인을 dry-run으로 확인](#실행-방법과-dry_run-규율)한 뒤 실제 실행으로 넘어가고, 끝나면 반드시 정리하세요([비용과 cleanup](#비용과-cleanup)).

!!! tip "어디서부터 읽을까"
    - **바로 손을 대고 싶다면** → [시작하기](getting_started.md)(설치 → 스모크 → dry-run → 노트북).
    - **순서대로 완주하려면** → [실행 runbook](RUN_E2E.md)에 단계별 핸드오프·비용·체크리스트가 있습니다.
    - **개념부터 잡으려면** → 배포/추론이 이 kit의 핵심이므로 [SageMaker 추론](04_sagemaker_inference.md)을 먼저 보세요. 학습은 [파인튜닝](03_finetuning.md), 데이터는 [합성 데이터](02_synthetic_data.md)입니다.

---

## 기존 Pain Point

이 kit이 없을 때 실제로 겪는 것들입니다.

- "Gemma를 SageMaker에서 파인튜닝→서빙"하는 예제는 조각조각 흩어져 있고, **버전·이미지 태그가 스치듯 오래된 경우가 많습니다**. 그대로 복붙하면 컨테이너 resolve부터 실패합니다.
- "AWS 예제는 죄다 DJL LMI를 쓰는데 vLLM이 제일 빠르다고 하고, 어떤 문서는 HF DLC를 쓴다"처럼 **컨테이너 선택지가 정리되지 않아** 무엇에 `model_data`를 물릴지 판단이 서지 않습니다.
- **tier/서비스 혼동**도 흔합니다. "endpoint를 Bedrock으로 부른다", "Serverless로 LLM 띄운다", "학습은 JumpStart로" 같은 오해가 실습을 무너뜨립니다.
- Gemma 특유의 함정(chat template의 system role 거부, fp16 NaN, packing cross-contamination)을 모르고 시작하면 **조용히 망가진 학습**을 얻게 됩니다.
- 합성 데이터를 teacher 그대로 만들고 **그걸로 평가**하면 성능을 과대평가하게 됩니다.

이 kit은 위 함정을 코드 주석·노트북·본 문서에 오개념 노트로 박아 두어, 초심자가 밟지 않도록 돕습니다.

---

## 왜 이 구조인가

!!! abstract "쉽게 말하면"
    이 kit은 "하나의 큰 튜토리얼"이 아니라 **"같은 부품을 공유하는 5개의 작은 완결 튜토리얼"**입니다.
    부품(합성·학습·서빙·평가)은 `common/`에 한 번만 작성해 두고, 코스는 데이터와 프롬프트만 갈아끼웁니다.
    멀티모달 코스만 이미지 입력이라 구조가 조금 다릅니다.

### 인프라 비용은 TCO의 한 칸일 뿐입니다

"관리형이 EC2보다 비싸다"는 비교는 **총 소유비용(TCO) 중 인프라 비용 한 칸만** 놓고 이루어집니다. 아래 축별 선택보다 한 층 위에 있는 전제, 즉 **왜 관리형 티어인가**가 여기서 갈립니다.

티어를 고를 때 사람들이 실제로 보는 축은 하나뿐입니다: **시간당 단가**. 그리고 그 비교에서 관리형이 지는 것은 사실입니다.
이 kit이 쓰는 `ml.g6.2xlarge`를 SageMaker endpoint로 띄우면 같은 세대의 `g6.2xlarge` EC2 인스턴스보다 시간당 단가가 높습니다.

??? info "단가는 직접 비교하세요"
    정확한 배율은 리전·시점에 따라 다릅니다. [SageMaker AI 요금](https://aws.amazon.com/sagemaker-ai/pricing/)과 [EC2 온디맨드 요금](https://aws.amazon.com/ec2/pricing/on-demand/)에서 같은 인스턴스 계열을 직접 대조하세요.

[![관리형 배포와 자체 배포의 총 소유비용 비교 도해. 왼쪽에 대표적 오해 세 가지("EC2나 온프렘으로 서빙하면 간단한데?", "그냥 vLLM/SGLang 띄우는 게 더 낫네!", "SageMaker AI 쓰려니까 EC2보다 인프라 비용이 비싼데?")와 TCO 관점의 반론(숙련된 팀·인력이 구성되지 않음, 유지보수 및 보안에 따른 숨겨진 시간과 비용)이 있고, 오른쪽 위에는 누적 막대 두 개가 있다. 관리형 배포(SageMaker AI) 막대는 짧고, 자체 배포(EC2/EKS) 막대는 훨씬 길며 인프라 비용·운영 비용·규정 준수 비용 세 구간으로 나뉜다. 오른쪽 아래 육각형 세 개는 판단해야 할 축을 비용(모델 호스팅 비용·운영 오버헤드·배포 및 관리해야 할 모델 수), 성능(지연 시간·처리량·가용성), 복잡성(엔지니어링 공수·모델 크기와 테스트와 업그레이드·페이로드 크기·추론 워크플로)으로 나눈다](images/why_sagemaker.png)](images/why_sagemaker.png)

*자체 배포의 막대가 긴 이유는 단가가 아니라 칸 수입니다. 인프라 비용 위에 운영 비용과 규정 준수 비용이 더 얹힙니다.*

그림 왼쪽의 세 문장은 이 kit을 처음 볼 때 실제로 하는 질문과 거의 같습니다. 세 개 모두 같은 대답을 갖습니다: **빠진 두 칸을 채우고 다시 비교하세요.**

| 비용 칸 | 무엇이 들어가나 | 자체 배포(EC2/EKS)에서는 | 관리형(SageMaker AI)에서는 |
|---|---|---|---|
| **인프라 비용** | 인스턴스·스토리지·네트워크 | 시간당 단가가 낮음 | 시간당 단가가 높음. **여기까지만 보면 자체 배포가 이깁니다** |
| **운영 비용** | GPU 드라이버·CUDA 업그레이드, `/ping` 상당의 health check와 로드밸런서 구성, 재시작·롤백, 관측 스택, 당직 | 전부 내 몫. 그리고 이것은 **인건비라서 청구서에 안 보입니다** | 컨트롤 플레인이 AWS 몫. 내가 쓰는 것은 `.env` 값과 노트북 몇 줄 |
| **규정 준수 비용** | guest OS 패치 적용, 감사 증적, 격리·암호화 구성 | [공동 책임 모델](https://aws.amazon.com/compliance/shared-responsibility-model/) 기준으로 guest OS와 그 위의 소프트웨어 패치는 **고객 책임** | 호스트와 관리형 런타임은 AWS 책임. 내 몫은 이미지 태그를 올리는 것([운영 관점 비교](01_sagemaker_basics.md#운영-관점-비교)의 「보안 패치·규정 준수」 행) |

그림 오른쪽 아래의 육각형 세 개는 그렇게 비교할 때 실제로 봐야 하는 축입니다. 이 kit의 실습에서 각각이 어디서 드러나는지 붙여 보면 추상적인 목록이 아니게 됩니다.

- **비용** — 모델 호스팅 비용, 운영 오버헤드, **배포·관리해야 할 모델 수**.

    마지막 항목이 가장 자주 빠집니다. 모델이 하나면 EC2 한 대에 vLLM을 띄우는 것으로 충분합니다.
    그런데 코스마다 다른 모델이 붙기 시작하면(이 kit만 해도 추출·분류·요약·멀티모달 코스가 각자 어댑터를 만듭니다) 대수가 아니라 **배포 파이프라인 수**가 늘어납니다.

- **성능** — 지연 시간, 처리량, 가용성.

    지연과 처리량은 서빙 엔진의 몫이라 자체 배포로도 같은 값을 낼 수 있습니다(vLLM은 같은 vLLM입니다). 하지만 **가용성**은 엔진이 주지 않습니다.

    `/ping` health check로 기동 실패를 걸러 내고 인스턴스를 **여러 AZ에 분산**해 주는 것은 endpoint 층의 기능입니다(AWS는 production endpoint에 [인스턴스 여러 대를 두라고 권고](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-best-practices.html)합니다. 이 kit은 실습이라 `initial_instance_count=1` 고정입니다).

    모델을 교체할 때의 canary·롤백은 [배포 가드레일](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)이 담당합니다(endpoint **업데이트** 전용 기능입니다).

- **복잡성** — 엔지니어링 공수, 모델 크기·테스트·업그레이드, **페이로드 크기**, 추론 워크플로.

    페이로드 크기와 처리 시간이 커지면 Real-time이 아니라 Asynchronous나 Batch Transform이 답입니다.

    자체 배포에서는 그 세 가지를 각각 큐·워커·Job 러너로 내가 만들어야 합니다([추론 4옵션 비교](04_sagemaker_inference.md#왜-real-time인가--추론-4옵션-비교)).

??? tip "그래도 자체 배포가 이기는 경우가 있습니다"
    이 그림은 관리형을 항상 고르라는 주장이 아닙니다. 운영 비용 칸이 **이미 지불된** 팀이 있습니다. 사내에 Kubernetes 플랫폼 팀과 관측 스택이 서 있고 GPU 노드가 이미 그 위에서 돌고 있다면, 세 번째 칸의 증분은 거의 0이고 남는 것은 인프라 단가 차이뿐입니다. 그때는 자체 배포가 정확히 더 싼 선택입니다.
    판단의 기준은 "관리형이냐 자체냐"가 아니라 **"운영 비용과 규정 준수 비용을 내가 이미 내고 있는가"**입니다. 티어별 조건 정리는 [언제 무엇을 쓰나](01_sagemaker_basics.md#언제-무엇을-쓰나)에, 티어 간 운영 축 대조는 [운영 관점 비교](01_sagemaker_basics.md#운영-관점-비교)에 있습니다.

티어를 관리형으로 정한 뒤, 그 안에서 축마다 무엇을 골랐는지가 다음 표입니다.

### 설계 축별 선택

| 축 | 이 kit의 선택 | 대안 | 왜 이걸 골랐나 (조건부) |
|---|---|---|---|
| 학습 경로 | **PyTorch DLC + [TRL `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) + [PEFT](https://huggingface.co/docs/peft/index) LoRA/QLoRA** | JumpStart 원클릭 / HF DLC | 커스텀 chat template·LoRA 타깃·bf16 등을 **직접 제어**해야 Gemma가 제대로 학습됩니다. 베이스가 순수 PyTorch DLC라 `scripts/requirements.txt`로 최신 `transformers`를 컨테이너 안에서 맞출 수 있습니다. 세밀한 제어가 필요 없다면 JumpStart도 유효한 선택입니다. |
| [추론 옵션](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) | **SageMaker real-time endpoint** | Serverless / Async / Batch | 대화형 SLM은 상시 저지연이 필요하므로 real-time이 적합합니다. **Serverless는 GPU가 없어 LLM에 부적합합니다.** 배치성 작업이라면 Async/Batch를 고려하세요. |
| 서빙 컨테이너 | **[vLLM](https://github.com/vllm-project/vllm) DLC (`SERVING_ENGINE=vllm`, 기본)** | [SGLang](https://github.com/sgl-project/sglang) DLC / [DJL LMI](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) | gemma-4 서빙에는 vLLM >= 0.19가 필요하고, AWS 독립 vLLM DLC가 그 최신을 가장 빨리 따라갑니다. 관리형 컨테이너와 `OPTION_*` 규약이 익숙하면 `lmi`, RadixAttention 등이 필요하면 `sglang`으로 env만 바꾸면 됩니다. 셋 다 연속 배칭 + OpenAI 호환(messages)이라 **호출 코드가 동일합니다**. |
| reasoning 모델 | **Bedrock Claude ([Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html))** | 자체 대형 모델 호스팅 | agentic 오케스트레이션은 대형 LLM이 유리하고, Bedrock은 상시 리소스 없이 토큰 단위로 과금됩니다. |
| 에이전트 | **[Strands](https://github.com/strands-agents/sdk-python) → [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)** | LangGraph 등 | Strands는 Bedrock/SageMaker와 정합하고, AgentCore는 관리형 런타임입니다. LangGraph 옵션은 `pyproject.toml` extras로 열려 있습니다. |
| 데이터 | **grounded 합성 (Bedrock Converse + critique/refine)** | 대형 오픈 합성셋 그대로 | 시드 도메인에 grounded 시켜 **task에 적합하고 라이선스도 안전**하게 만듭니다. |

### 기술적 차이 3가지

1. **서비스 경계를 코드로 분리했습니다.** SageMaker endpoint 호출과 Bedrock 호출은 **별개 서비스·별개 클라이언트**입니다(`common/aws_utils.py`).

    - endpoint → `boto3 sagemaker-runtime.invoke_endpoint()` (스트리밍은 `invoke_endpoint_with_response_stream`)
    - Bedrock → `boto3 bedrock-runtime.converse()`

    호출 스키마와 API 문서 링크는 [서비스 경계](04_sagemaker_inference.md#서비스-경계--endpoint--bedrock)에 있습니다.

2. **DLC 이미지 URI를 env로 분리했습니다.** 태그는 자주 바뀌므로 코드에 박지 않습니다. `common/dlc.py`가 다음 순서로 해석합니다.

    1. 완전 URI env — `DLC_IMAGE_URI` / `VLLM_IMAGE_URI` / `SGLANG_IMAGE_URI` / `LMI_IMAGE_URI`
    2. `DLC_REPOSITORY` + `DLC_TAG`
    3. SDK `image_uris.retrieve`

    현행 계정·리포지토리·태그 조합은 [deep-learning-containers의 available_images.md](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)가 원본입니다. 로컬 `transformers`와 컨테이너 `transformers`도 서로 다르므로 구분합니다.

3. **Gemma 함정 방어를 기본값으로 넣었습니다.** 아래가 모두 `tracks/*/scripts/train.py`에 내장되어 있습니다.

    - `attn=eager`를 안전 기본값으로 둡니다.
    - `bf16`을 강제합니다(**fp16 금지** — Gemma에서 NaN을 유발합니다).
    - packing은 flash-attn이 아니면 자동으로 끕니다.
    - E2B/E4B는 저장 직전 KV-shared dead weight를 base에서 복원합니다(`_revive_kv_shared_from_base`).
    - 텍스트 코스는 머지 후 language 서브모듈만 `*ForCausalLM`으로 re-export합니다.

---

## E2E 파이프라인 (텍스트 코스 7단계)

!!! abstract "쉽게 말하면"
    문제를 정하고 → 데이터를 만들고 → 학습하고 → 띄우고 → 에이전트로 감싸고 → 채점하는 흐름입니다.
    각 단계가 노트북 하나에 대응합니다. 아래는 텍스트 코스(01~04) 기준이고,
    멀티모달 코스 05는 더 짧은 별도 파이프라인입니다.

```
 task 정의
   │
   ▼
 [00_setup]      env·role·bucket·의존성 확인 (DRY_RUN 권장)
   │
   ▼
 [01_data_and_synthetic]  오픈 시드 로드 + grounded 합성(Bedrock Converse + critique/refine)
   │                       └ common/synth/bedrock_synth.py  →  messages JSONL
   ▼
 [02_train_sft_sagemaker]     PyTorch DLC + TRL SFTTrainer(LoRA/QLoRA)  ← JumpStart 아님
   │                       └ tracks/*/scripts/train.py (로컬 dry-run ↔ SageMaker .fit() 겸용)
   │
   ├┈┈▶ (선택) [02a_train_grpo_sagemaker]  SFT→GRPO 정련(RLHF)  ← 추출·분류 코스만
   ├┈┈▶ (선택) [02b_local_serve]           배포 전 로컬 vLLM 프리플라이트
   ▼
 [03_deploy_endpoint]     real-time endpoint (vLLM DLC 기본 · SGLang/DJL LMI 선택)
   │                       └ 호출: sagemaker-runtime.invoke_endpoint (별개: Bedrock)
   ▼
 [04_evaluate]            held-out 세트로 성공기준 수치화 (로컬·빠름·저렴)
   │                       └ common/eval_utils.py
   ▼
 [05_agentic_strands]     Strands Agent (reasoning=Bedrock Claude, tool=call_slm→endpoint)
   │
   ▼
 [06_agentcore_deploy]    AgentCore Runtime (ARM64, /invocations + /ping :8080)
   │                       └ agentcore/app.py
   ▼
 [99_cleanup]             endpoint·리소스 삭제 (과금 중단)
```

- **(선택) `02a_train_grpo_sagemaker`**: SFT 결과를 GRPO(RLHF)로 정련합니다(`scripts/train_grpo.py`).
- **(선택) `02b_local_serve`**: SageMaker 배포 전 로컬 vLLM으로 프리플라이트합니다(`scripts/serve_local_vllm.sh`, `scripts/bench_local_vllm.sh`).
- 위 두 노트북은 코스마다 있고 없고가 갈립니다. 어느 코스에 붙는지는 [텍스트 코스의 공통 노트북 세트](#텍스트-코스의-공통-노트북-세트)에 있습니다.
- **`04_evaluate`**: endpoint를 held-out 세트로 직접 호출해 코스별 지표를 계산합니다. 로컬에서 돌기 때문에 빠르고 저렴합니다.

!!! danger "합성·학습셋으로 평가하지 마세요"
    증강 이전 seed에서 **held-out을 먼저 분리**한 뒤 나머지만 합성으로 늘립니다.
    합성 데이터나 학습셋으로 채점하면 지표가 조용히 부풀고, 그 수치를 근거로 배포 결정을 내리게 됩니다.
    분리 규율은 [held-out 규율](02_synthetic_data.md#held-out-규율--합성으로-평가-금지)에 있습니다.

### 멀티모달 코스 05의 별도 파이프라인

`tracks/05_multimodal_extraction`은 이미지 → 구조화 JSON 추출이라 합성 데이터 단계가 없고 노트북 세트가 다릅니다(5단계).

```
 [00_setup] ─▶ [01_data_explore] ─▶ [02_train_mm_sagemaker] ─▶ [03_deploy_mm_endpoint] ─▶ [99_cleanup]
              cord-v2 영수증           vision tower 동결 +          멀티모달 endpoint
              이미지+JSON 탐색          language LoRA               (이미지 입력 허용)
              (합성 단계 없음)          scripts/train_mm.py
```

서빙은 **이미지 입력을 받는 멀티모달 endpoint**입니다(텍스트 전용으로 re-export 하지 않고 vision tower를 유지합니다). agentic/agentcore 단계는 텍스트 코스 전용이라 05에는 없습니다.

---

## 5개 독립 코스와 공통 레이어

**텍스트 4개 코스는 데이터셋과 프롬프트만 다르고 파이프라인은 동일합니다.** 그래서 공통 부품은 `common/`에 한 번만 두면 됩니다. 멀티모달 코스(05)만 이미지 입력이라 노트북 세트가 다릅니다.

아래 표는 **어느 코스를 고를지** 정하는 데 필요한 것만 담았습니다. 주 지표를 고른 이유와 보조 지표, 시드 데이터셋의 함정은 각 코스 문서에 있습니다. 코스 이름을 누르면 갑니다.

| 코스 | task | 시드 데이터셋 (라이선스) | 주 지표 |
|---|---|---|---|
| [추출 → JSON](courses/extraction.md) (**플래그십**) | 텍스트 → 구조화 JSON | `glaiveai/glaive-function-calling-v2` (apache-2.0) | `arg_f1` |
| [분류](courses/classification.md) | 은행 고객 문의 intent 77종 분류 | `mteb/banking77` (mit) | macro-F1 |
| [요약](courses/summarization.md) | 법안 원문 요약 | `FiscalNote/billsum` (cc0-1.0) | ROUGE-L |
| [도메인 QA](courses/domain_qa.md) | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) | LLM-judge |
| [멀티모달 추출](courses/multimodal.md) | **이미지** → 구조화 JSON (영수증) | `naver-clova-ix/cord-v2` (cc-by-4.0) | valid JSON + 필드 정확도 — **육안 대조**(`04_evaluate` 없음) |

지표가 코스 선택을 가르는 지점이 하나 있습니다.

- **추출·분류** — **정답과 규칙으로 대조**할 수 있어 채점이 순수 파이썬이고 비용이 0입니다.
- **요약·도메인 QA** — 정답이 하나가 아니라 LLM-judge를 붙여야 합니다(Bedrock 호출 과금). 같은 이유로 GRPO(`02a`)가 없습니다.

디렉터리는 표 순서대로 `tracks/01_extraction_to_json` · `02_classification` · `03_summarization` · `04_domain_qa` · `05_multimodal_extraction`입니다.

코드 쪽 이름(`tracks/` 디렉터리, `TRACKS` 레지스트리, `--track` 인자)은 초기 명칭을 그대로 유지하고 있습니다. 본문의 "코스"와 같은 것을 가리킨다고 읽으시면 됩니다.

### 텍스트 코스의 공통 노트북 세트

코스 01~04는 **항상 있는 노트북 8개**를 같은 순서로 갖습니다([E2E 파이프라인](#e2e-파이프라인-텍스트-코스-7단계)의 7단계 + `99_cleanup`).

```
00_setup  →  01_data_and_synthetic  →  02_train_sft_sagemaker
          →  03_deploy_endpoint     →  04_evaluate
          →  05_agentic_strands     →  06_agentcore_deploy     →  99_cleanup
```

여기에 조건부 노트북이 두 개 더 붙습니다(무엇을 하는 단계인지는 위 [E2E 파이프라인](#e2e-파이프라인-텍스트-코스-7단계)에 있습니다).

- **`02a_train_grpo_sagemaker` — 추출·분류 코스에만 있습니다.** 리워드를 프로그램으로 채점할 수 있는 두 코스만 대상입니다([왜 추출·분류 코스에만 GRPO가 있나](03_finetuning.md#왜-추출분류-코스에만-grpo가-있나)).
    그래서 이 두 코스는 노트북이 10개, 요약·도메인 QA는 9개입니다.
- **`02b_local_serve` — 4개 코스 모두에 있지만 선택입니다.** 로컬 GPU가 없으면 건너뛰어도 됩니다.

따라서 최소 경로는 `00 → 01 → 02 → 03 → 04 → 99`이고, 노트북별 산출물은 [노트북 단계와 산출물](#노트북-단계와-산출물) 표에 있습니다.

코스별로 달라지는 값은 두 곳에만 있습니다.

- `tracks/*/track_data.py` — 시드 데이터셋 로드와 `messages` 어댑터(원본 row → 학습 형태).
- `common/config.py`의 `TRACKS` 레지스트리 — `seed_dataset`·`max_seq_length`·epoch 등 코스 상수.

### 학습 길이와 서빙 길이는 다른 값입니다

코스 페이지의 설정 표에 `max_seq_length`(학습)와 `serve_max_model_len`(서빙)이 따로 나오는 이유입니다.

- **학습** — "입력+정답"이 `max_seq_length`에 들어가도록 자릅니다.
- **서빙** — "입력 + **앞으로 생성할** 토큰"이 한 컨텍스트에 함께 들어가야 합니다.

두 값을 하나로 묶으면 입력이 긴 코스에서 `(프롬프트 + max_tokens) > 컨텍스트`가 되어 vLLM이 400(`context length exceeded`)으로 거부합니다(요약 코스: 프롬프트 max 2,006 + 생성 256 > 2048).

`TrackSpec.serve_max_model_len`을 지정하지 않은 코스는 `tracks/_shared_build._serve_len()`이 `max_seq_length × 2`를 씁니다(입력만큼 생성 여유를 둔다는 뜻).

생성 상한 `gen_max_tokens`는 또 별개이며 코스별 정답 길이 분포에서 정합니다.
코스별 값과 절단 확인법은 [max_tokens 절단과 finish_reason](05_serving_containers.md#max_tokens-절단과-finish_reason)에 있습니다.

### 멀티모달 코스의 다른 점

[멀티모달 추출 코스](courses/multimodal.md)는 노트북이 5개뿐이고 합성·agentic 단계가 없습니다. 단계 도해와 이유는 위의 [멀티모달 코스 05의 별도 파이프라인](#멀티모달-코스-05의-별도-파이프라인)에 있습니다.

### 공유되는 common 부품

| 파일 | 역할 |
|---|---|
| `common/config.py` | gemma-4 프리셋(`MODEL_SIZE`)·`MODEL_ID`·`SERVING_ENGINE`·region/role/bucket + env, `DRY_RUN`, `TRACKS` 레지스트리 |
| `common/gemma_format.py` | 표준 `messages` 어댑터 (`apply_chat_template`에 위임, 수동 마커 금지) |
| `common/aws_utils.py` | `invoke_endpoint`(sagemaker-runtime) · `converse`(bedrock-runtime) · CloudWatch 링크 · 비용 가드 |
| `common/dlc.py` | DLC 이미지 URI 해석(계정 `763104351884`, 태그는 env 주입) + 엔진별 서빙 env 조립(`serving_env`) |
| `common/model_inspect.py` | 체크포인트 점검(KV-sharing 여부·서빙 가능 엔진 판정) |
| `common/llm_gateway.py` | (LiteLLM) Bedrock + SageMaker endpoint 단일 인터페이스 |
| `common/synth/bedrock_synth.py` | grounded 합성 (Converse + critique/refine, boto3만·무의존성) |
| `common/eval_utils.py` | 코스별 메트릭 (추출/분류/요약/QA) + Bedrock LLM-judge |
| `tracks/*/scripts/train.py` · `train_grpo.py` | self-contained 학습 (로컬 dry-run ↔ SageMaker 겸용) |
| `agentcore/app.py` | AgentCore Runtime 엔트리포인트 ([bedrock-agentcore SDK](https://github.com/aws/bedrock-agentcore-sdk-python)로 Strands 에이전트 호스팅) |

??? question "오개념 — “코스끼리 뭔가 공유하니 순서대로 해야 하나?”"
    **그렇지 않습니다.** 5개 코스는 **완전히 독립된 E2E**입니다. 관심 있는 코스 하나만 `00→99`로 돌려도 완결됩니다.
    `common/`은 코드 중복을 제거하기 위한 것일 뿐, 실행 의존성이 아닙니다.

---

## 모델 선택 (gemma-4 프리셋 5종)

기본값은 `MODEL_SIZE=E4B`입니다. **gemma-4 전 사이즈가 apache-2.0 + ungated라서 HF 토큰이 필요 없습니다**(HF raw `config.json` 확인).

라이선스 배너와 chat template은 [gemma-4-E4B-it 모델 카드](https://huggingface.co/google/gemma-4-E4B-it)가 원본입니다.

| 프리셋 (`MODEL_SIZE`) | 모델 ID | 성격 | 프리셋 인스턴스 | transformers 요건 |
|---|---|---|---|---|
| `E2B` | `google/gemma-4-E2B-it` | effective 2.3B(on-disk 5.12B, PLE가 46.7%). KV-sharing 있음. 계열 최소라 스모크 테스트에 적합 | `ml.g5.2xlarge` | >= 5.5.0 |
| **`E4B`** (기본) | `google/gemma-4-E4B-it` | effective 4.5B(PLE 포함 ~8B). KV-sharing 있음. 단일 L4 24GB QLoRA 여유 | `ml.g5.2xlarge` | >= 5.5.0 |
| `12B` | `google/gemma-4-12B-it` | 11.95B dense, unified arch | `ml.g5.12xlarge` | >= 5.10.1 |
| `26B-A4B` | `google/gemma-4-26B-A4B-it` | MoE total 25.2B / active 3.8B, 128 experts. audio 미지원(vision만) | `ml.g5.12xlarge` | >= 5.5.0 |
| `31B` | `google/gemma-4-31B-it` | 31.27B dense, 계열 최대. audio 미지원(vision만) | `ml.g6e.12xlarge` | >= 5.5.0 |

- **`31B`만 프리셋 인스턴스가 `ml.g6e.12xlarge`(L40S, nominal 48GB · 가용 44GiB)입니다.** 4bit로도 base가 24GB 카드(L4·A10G, 가용 22GiB)를 넘길 수 있어서입니다.
    `params/2` 어림값이 통하지 않는 지점입니다.
- 이 kit의 `.env`는 용량 대기가 짧은 `ml.g6.2xlarge`(L4 24GB GPU + 32GB RAM)로 학습·서빙 인스턴스를 override해 둡니다.
    크기를 올릴 때는 `TRAIN_INSTANCE_TYPE`/`INFER_INSTANCE_TYPE`도 함께 조정하세요.
- **인스턴스는 GPU만 보지 말고 호스트 RAM도 보세요.** QLoRA 학습 자체는 GPU에 들어가지만, 학습 후 merge/re-export가 base 모델을 bf16 full로 CPU에 로드하므로 RAM이 병목입니다.
    `train.py`는 merge 전 학습 모델을 해제하고 base를 `low_cpu_mem_usage`로 로드해 사본을 최소화합니다. E4B의 peak RAM 실측값은 약 17.5GB입니다.
- **gemma-4는 전 사이즈가 멀티모달입니다(텍스트 전용 공식 체크포인트가 없습니다).** 그래서 텍스트 코스는 머지 후 language 서브모듈만 텍스트 arch(`*ForCausalLM`, `model_type=*_text`)로 re-export합니다.
    이 과정을 건너뛰면 서빙 컨테이너가 image processor를 찾다가 `Can't load image processor`로 죽습니다.

??? info "31B가 24GB 카드를 넘기는 내역"
    - quantizable linear 29.29B → NF4로 14.6GB(+double-quant 상수 0.46GB)
    - `embed_tokens` 1.41B와 vision tower 0.58B는 4bit로 내려가지 않아 bf16으로 남음
    - 합계 base만 약 19.1GB. 여기에 activation·optimizer를 얹으면 22GiB에서는 sharding이 강제됩니다.
    - 호스트 RAM은 병목이 아닙니다(merge peak 약 68GB vs 384GiB).

??? question "오개념 — “Gemma는 gated니까 HF 토큰부터 받아야 하지 않나요?”"
    **gemma-4는 아닙니다.** 라이선스는 **모델 계열**을 따릅니다. Gemma 3/2/3n은 gated + Gemma Terms(서빙 시 use-restriction 전파 의무)이지만,
    **Gemma 4는 apache-2.0 + ungated**여서 토큰·약관 수락이 없습니다. `MODEL_IS_GATED` 기본값이 `0`인 것도 이 때문입니다.
    gated 모델을 `MODEL_ID`로 지정할 때만 `MODEL_IS_GATED=1` + 토큰이 필요합니다. gated + Gemma Terms의 대표 예는 [`gemma-3-4b-it` 모델 카드](https://huggingface.co/google/gemma-3-4b-it)입니다. **재배포/서빙 전 그 페이지의 라이선스 배너를 재확인**하세요.

### Gemma 학습 관용구

`tracks/*/scripts/train.py`에 기본으로 들어가 있는 결정들입니다.

- chat template은 `-it` 토크나이저에 내장되어 있으므로 `apply_chat_template`에 위임합니다(수동으로 `<start_of_turn>`를 조립하지 마세요). system role이 거부되면 첫 user 턴에 fold합니다.
- LoRA는 텍스트 코스에서 `target_modules="all-linear"` + `modules_to_save=["lm_head","embed_tokens"]`입니다.
    멀티모달 학습은 vision/audio proj가 `ClippableLinear`라 매칭되면 크래시하므로 `language_model` 한정 regex를 씁니다.
- `bf16`은 필수입니다(**fp16 금지** — Gemma에서 오버플로/NaN을 유발합니다). `attn=eager`가 안전 기본값입니다(soft-cap/sliding-window 정합성).
- packing은 `flash_attention_2`일 때만 켜집니다. eager/sdpa에서는 **샘플 간 cross-contamination을 방지하기 위해 자동으로 꺼집니다**.
- E2B/E4B는 KV-shared 레이어의 텐서(`k_norm`/`k_proj`/`v_proj`)를 transformers가 아예 만들지 않아 `save_pretrained` 시 소실됩니다(E4B 실측 54개).
    그러면 vLLM이 `weights not initialized ...k_norm`으로 엔진 초기화에 실패합니다([vLLM 이슈 #44788](https://github.com/vllm-project/vllm/issues/44788)).
    `train.py`는 저장 직전 그 텐서를 base에서 복원합니다(복원 전 665키 → 복원 후 719키 = 원본과 동일). 연산에 쓰이지 않는 dead weight라 정확도에는 무해합니다.

---

## 실행 방법과 DRY_RUN 규율

**(1) `uv`로 설치하고, (2) env를 주입한 뒤, (3) `DRY_RUN=1`로 파이프라인을 확인하고 나서 실제 실행으로 넘어가시면 됩니다.**

### uv로 설치

```bash
# uv 미설치 시:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml     # 또는 재현 설치:  uv sync
# 최신화:  uv lock --upgrade  /  uv lock --upgrade-package transformers
```

pip만 쓰신다면 `pip install -r requirements.txt`를 실행하세요. 버전은 `>=` floor로만 고정되어 있고, floor 값 자체는 **실행 전 재확인** 대상입니다.

| 무엇 | 원본 파일 | 현재 floor |
|---|---|---|
| 로컬 env | `requirements.txt` | transformers 5.14.1 / trl 1.8.0 / peft 0.19.1 / datasets 5.0.0 |
| 컨테이너 안 | `tracks/*/scripts/requirements.txt` | transformers 5.14.1 / trl 1.9.0 / peft 0.19.1 |

### 환경변수 주입

시크릿·계정 ID·절대경로는 하드코딩하지 않습니다. 리포의 `.env`는 **설정만 담기 때문에 커밋됩니다**. 개인 값은 `.env.local`(gitignore)에 두세요.

```bash
export AWS_REGION=us-west-2                              # config 기본값. 리전 재확인 후 사용
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5   # config 기본값. inference-profile prefix 필수
export MODEL_SIZE=E4B                                    # E2B | E4B(기본) | 12B | 26B-A4B | 31B
export SERVING_ENGINE=vllm                               # vllm(기본) | sglang | lmi
export DRY_RUN=1                                         # 먼저 파이프라인 검증, 실제 실행 시 0
# gated 모델(gemma-3 계열)을 쓸 때만:  export MODEL_IS_GATED=1  +  hf auth login
```

`BEDROCK_CLAUDE_MODEL_ID`에 `global.`/`us.`/`eu.`/`apac.` prefix가 붙는 것은 Bedrock이 모델을 [inference profile](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)로 노출하기 때문입니다.
prefix 없는 raw 모델 ID를 넘기면 호출이 거부됩니다.

`AWS_REGION` 기본값이 `us-west-2`(오레곤)인 이유는 GPU 용량 여유가 큰 편이기 때문입니다. `InsufficientInstanceCapacity`로 막히면 이 값만 바꿔 재시도하세요.

HF 토큰은 env보다 `hf auth login`(파일 저장)을 권장합니다. `config.get_hf_token()`이 env와 저장 토큰을 모두 읽습니다.

### DRY_RUN 우선

`DRY_RUN=1`로 두면 **파이프라인만** 검증합니다. 학습은 **1 epoch · `max_seq_length` <= 512 · 앞 32건**으로, 합성은 소량으로 돕니다.
관련 코드는 `common/config.py:is_dry_run`과 `train.py`의 dry-run 오버라이드입니다.

**GPU dry-run은 L40S에서 검증되었습니다.** 다른 GPU/메모리에서는 배치·seq 길이를 재조정해야 할 수 있습니다. 파이프라인이 확인되면 `DRY_RUN=0`으로 실제 실행하세요.

??? question "오개념 — “로컬 `transformers`와 SageMaker가 같은 버전이겠지?”"
    **그렇지 않습니다.** 로컬 env의 `transformers`는 데이터 준비/dry-run용이고, **SageMaker 컨테이너 버전은 DLC 이미지 태그**가 결정합니다.
    컨테이너 안에서 상위 버전이 필요하면 `tracks/*/scripts/requirements.txt`가 이를 업그레이드합니다.
    이 kit의 학습 베이스가 순수 PyTorch DLC인 것도 같은 이유입니다. baked-in `transformers`에 묶이지 않습니다.

---

## 문서와 노트북 매핑

### 상세 문서 지도

| 문서 | 다루는 것 | 대응 노트북 단계 | 주요 참조 코드 |
|---|---|---|---|
| [01 SageMaker 기초](01_sagemaker_basics.md) | Training Job vs Endpoint, 경로 규약, 수명과 과금 (선행 개념) | 전 단계 | `common/aws_utils.py`, `common/config.py` |
| [02 합성 데이터](02_synthetic_data.md) | grounded 합성(Converse + critique/refine), 라이선스·PII, held-out 규율 | `01_data_and_synthetic` | `common/synth/bedrock_synth.py`, `tracks/*/track_data.py` |
| [03 파인튜닝](03_finetuning.md) | PyTorch DLC + TRL SFTTrainer, LoRA/QLoRA, Gemma 관용구, SFT→GRPO | `02_train_sft_sagemaker`, `02a_train_grpo_sagemaker` | `tracks/*/scripts/train.py`, `common/gemma_format.py` |
| [04 SageMaker 추론](04_sagemaker_inference.md) `[CORE]` | 추론 4옵션, endpoint 3층 구조와 호출(sagemaker-runtime), 서빙 컨테이너·DLC 이미지 | `03_deploy_endpoint` | `common/aws_utils.py`, `common/dlc.py` |
| [05 서빙 컨테이너](05_serving_containers.md) | vLLM / SGLang / DJL LMI 비교, KV-shared 복원, OOM·절단 대응 | `03_deploy_endpoint`, `02b_local_serve` | `common/dlc.py`, `common/model_inspect.py` |
| [06 Agentic loop](06_agentic.md) | Strands(Bedrock reasoning + SLM tool) → AgentCore Runtime | `05_agentic_strands`, `06_agentcore_deploy` | `agentcore/app.py`, `common/llm_gateway.py` |
| [실행 runbook](RUN_E2E.md) | 단계별 핸드오프·비용 가드·완료 기준 | 전 단계 | — |
| [SDK V3](sdk_v3/index.md) | V2→V3 매핑, 메타패키지 4레이어, 마이그레이션 함정 | 전 단계 | `sagemaker` 3.16.0 |

위 표의 가이드는 **주제별**(데이터·학습·배포·에이전트)이라 5개 코스에 공통으로 적용됩니다. **코스별**로 무엇이 다른지는 코스 문서 5개가 따로 다루며, [5개 독립 코스와 공통 레이어](#5개-독립-코스와-공통-레이어)의 표에서 연결됩니다.

평가는 별도 문서 없이 `common/eval_utils.py`와 각 코스의 `04_evaluate` 노트북에 담겨 있습니다(held-out 평가, 코스별 메트릭, LLM-judge).

### 노트북 단계와 산출물

| 노트북 | 산출물 | 비고 |
|---|---|---|
| `00_setup` | env/role/bucket 확인 | `DRY_RUN` 권장 |
| `01_data_and_synthetic` | `messages` JSONL(합성) | S3 업로드 |
| `02_train_sft_sagemaker` | 학습 Job → 모델 아티팩트(S3) | LoRA 머지 + 텍스트 re-export 산출물 포함 |
| (선택) `02a_train_grpo_sagemaker` | GRPO 정련 모델 | 추출·분류 코스만 |
| (선택) `02b_local_serve` | 로컬 vLLM 프리플라이트 결과 | 과금 없음(로컬 GPU) |
| `03_deploy_endpoint` | real-time endpoint | 과금 시작 |
| `04_evaluate` | 메트릭 리포트 | held-out만 (로컬·빠름·저렴) |
| `05_agentic_strands` | 로컬 에이전트 루프 | endpoint + Bedrock |
| `06_agentcore_deploy` | AgentCore Runtime | Runtime 과금 |
| `99_cleanup` | 리소스 삭제 | 과금 중단 |

---

## 자주 나오는 오개념

앞에서 다루지 않은, kit 전체를 볼 때 자주 나오는 착각들입니다.

??? question "오개념 — “AWS 예제는 다 DJL LMI인데, 이 kit은 왜 vLLM이 기본인가요?”"
    **둘 다 씁니다. 기본값만 vLLM DLC입니다.** gemma-4 서빙에는 vLLM >= 0.19가 필요하고, AWS 독립 vLLM DLC가 그 최신을 가장 빨리 따라갑니다.
    LMI도 됩니다. 단 **번들 vLLM 버전을 결정하는 것은 태그의 `lmi<NN>` 부분**입니다. 이 kit이 고정한 `djl-inference:0.36.0-lmi27.0.0-cu130-v1.1`은 LMI 27.0.0 = vLLM 0.23.1이라 조건을 충족합니다(ECR 실조회로 확인). 앞의 `0.36.0`은 djl-serving 버전이라 판단 기준이 아니므로, `LMI_IMAGE_URI`를 비워 SDK 폴백에 맡기면 같은 `0.36.0` 키로 더 낮은 `-lmi<NN>` 태그가 잡힐 수 있습니다. 그 태그의 번들 vLLM이 0.19 미만이면 gemma-4가 로드되지 않으니 배포 전 확인하세요.
    `SERVING_ENGINE=lmi`로 두면 [DJL LMI](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html)가 `OPTION_ROLLING_BATCH=vllm`으로 뜨고, `sglang`도 같은 방식으로 고를 수 있습니다.
    세 엔진 모두 연속 배칭 + OpenAI 호환(messages)이라 **호출 코드는 바뀌지 않습니다**. 선택 기준은 [서빙 컨테이너](05_serving_containers.md)에 있습니다.

컨테이너 이야기는 학습 쪽에서도 같은 형태로 반복됩니다.

??? question "오개념 — “학습은 HF DLC를 써야 하는 거 아닌가요?”"
    **꼭 그렇지 않습니다.** 이 kit은 순수 **PyTorch DLC**(`pytorch-training`)를 베이스로 쓰고 `scripts/requirements.txt`로 `transformers`/`trl`/`peft`를 직접 설치합니다.
    [HF DLC](https://huggingface.co/docs/sagemaker/index)의 baked-in `transformers`는 gemma-4에 필요한 버전보다 낮을 수 있는데, 베이스를 PyTorch DLC로 두면 컨테이너 안에서 최신으로 맞출 수 있습니다.
    학습 이미지는 **리전별 private ECR**(`763104351884.dkr.ecr.<region>...`)만 허용됩니다. `public.ecr.aws` URI를 주면 `CreateTrainingJob`이 거부합니다.

학습·서빙을 지나면 평가 단계에서 tier를 헷갈리게 됩니다.

??? question "오개념 — “SageMaker 관리형 evaluator로 채점하면 되지 않나요?”"
    **이 kit의 산출물에는 쓸 수 없습니다.** SDK v3의 `BenchMarkEvaluator`/`LLMAsJudgeEvaluator`/`CustomScorerEvaluator`는 **SageMaker Public Hub에 평가 레시피가 등록된 모델**(Amazon Nova·일부 JumpStart) 전용입니다.
    gemma-4 커스텀 파인튜닝 산출물(S3 체크포인트)은 Hub 레시피가 없어 `DescribeHubContent ... does not exist`로 실패했습니다.
    그래서 평가 경로는 `04_evaluate`의 **로컬 메트릭 평가**(`common/eval_utils.py`)입니다(빠르고 저렴하다는 부수 효과도 있습니다).

마지막은 이 kit에서 가장 비싼 착각인 과금에 관한 것입니다.

??? question "오개념 — “endpoint를 안 부르면 공짜겠지?”"
    **그렇지 않습니다.** real-time endpoint는 **호출 여부와 무관하게 provisioned 인스턴스가 시간당 과금**됩니다.
    쓰지 않는다면 삭제하는 것이 정답입니다([비용과 cleanup](#비용과-cleanup)).

위 항목들의 근거를 원본에서 직접 확인하고 싶다면 다음이 출발점입니다.

??? info "더 읽을 거리"
    LMI 컨테이너가 실제로 어떻게 빌드되는지는 [djl-serving의 `lmi.Dockerfile`](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docker/lmi.Dockerfile)에서 Dockerfile 수준으로 확인할 수 있습니다(vLLM·SGLang DLC는 [deep-learning-containers의 `vllm`](https://github.com/aws/deep-learning-containers/tree/master/vllm)·[`sglang`](https://github.com/aws/deep-learning-containers/tree/master/sglang) 디렉터리).
    엔진 자체의 버전·기능 지원 여부는 문서보다 저장소(위 [설계 축별 선택](#설계-축별-선택)의 vLLM·SGLang 링크)가 빠릅니다.

---

## 비용과 cleanup

!!! danger "비용과 cleanup"
    **real-time endpoint는 삭제 전까지 시간당(GPU 인스턴스) 과금됩니다.** 실습 후에는 반드시 `99_cleanup.ipynb`를 실행하거나 endpoint를 직접 삭제하세요.
    **AgentCore Runtime**도 배포하면 Runtime 리소스가 과금되므로 사용하지 않을 때는 정리해 주세요.
    여러 코스·여러 리전에 띄운 적이 있다면 각 코스 prefix와 리전별 endpoint 목록을 모두 확인해야 합니다.

각 노트북은 학습/배포 직후 **CloudWatch 다이렉트 링크**를 출력합니다(`common/aws_utils.cw_links`).

| 소스 | 과금 방식 | 정리 방법 |
|---|---|---|
| SageMaker real-time endpoint | 인스턴스 시간당, 삭제 전까지 계속 | `99_cleanup` → `delete_endpoint` → `delete_endpoint_config` → `delete_model` |
| SageMaker Training Job | Job 실행 시간만(종료 시 과금 중단) | 자동 종료 |
| Bedrock Converse | 호출 토큰량 기준, 상주 리소스 없음 | teardown 불필요. 대량 합성/평가 시 비용 누적 주의 |
| AgentCore Runtime | Runtime 리소스 과금(배포한 경우) | `bash agentcore/cleanup_agent.sh --aws`(Runtime + ECR) |
| 로컬 `local_model/`·vLLM 프로세스 | 과금 없음(디스크·GPU 점유) | `bash scripts/cleanup_local.sh --yes` |

### 라이선스 요약

- **Gemma 4는 apache-2.0 + ungated**로 마찰이 가장 적습니다(이 kit의 기본 경로입니다).
- Gemma 3/2/3n은 **Gemma Terms + gated**입니다. HF 토큰·약관 수락이 필요하고, 서빙 시 use-restriction 전파 의무가 따릅니다(예: [`gemma-3-4b-it` 모델 카드](https://huggingface.co/google/gemma-3-4b-it)).
- 시드 데이터셋은 전부 permissive한 것만 선별했으나, share-alike(dolly의 cc-by-sa-3.0 등) 파생물은 주의하시기 바랍니다.
- 재배포/서빙 전에 각 모델·데이터셋의 **live 라이선스 배너를 재확인**하세요.
