# SageMaker AI Fine-tuning & Serving E2E

Gemma 4를 Amazon SageMaker AI에서 **합성 데이터 → 파인튜닝 → 서빙 → 평가 → agentic loop**까지 잇는 한국어 핸즈온 가이드입니다.
**태스크별 실습 코스** 5개가 각각 독립된 E2E로 동작하므로, 필요한 태스크 하나만 골라 처음부터 끝까지 돌릴 수 있습니다.

!!! tip "어디서부터 읽을까"
    - **설치부터 첫 학습까지** → [시작하기](getting_started.md)
    - **SageMaker AI가 처음이라면** → [SageMaker AI 기초](01_sagemaker_basics.md)를 먼저 읽으세요. Training Job과 Endpoint의 수명·과금 차이를 잡아 두면 아래 가이드가 전제하는 개념이 채워집니다.
    - **전체를 완주한다면** → [실행 runbook](RUN_E2E.md)에 단계별 핸드오프와 비용이 있습니다.
    - **구조부터 보고 싶다면** → [전체 지도](00_overview.md)에 문서·노트북 매핑이 있습니다.
    - **어느 코스를 고를지 정해야 한다면** → 아래 [코스](#코스) 표에서 고르세요.
    - **특정 주제만** 필요하면 아래 가이드에서 골라 보세요.

## 왜 이 kit인가

**"SageMaker AI가 Gemma 4를 지원하지 않나?"** — 지원합니다. 다만 **안 되는 조합이 꽤 있습니다.**

가장 눈에 띄는 것: **JumpStart로는 gemma-4를 파인튜닝할 수 없습니다.** 5종 전부 `training_supported=False`이고, 배포만 됩니다. 관리형 파인튜닝 경로도 지원 모델·기법·리전 목록이 정해져 있어, 원하는 조합이 빠져 있으면 목록에 오를 때까지 기다려야 합니다.

| 방식 | 파인튜닝 | 성격 |
|---|---|---|
| **SageMaker JumpStart** | 불가 (배포만) | 클릭 몇 번으로 endpoint. gemma-4는 5종 모두 `training_supported=False` |
| **SageMaker AI model customization** | 가능 | 관리형 SFT/DPO/RFT. 지원 모델·기법·리전이 한정 |
| **SageMaker Recipes** | 가능 | 검증된 레시피로 SFT/DPO/GRPO. 지원 모델 목록에 의존 |
| **이 kit** (DLC + 커스텀 스크립트) | 가능 | 모델·기법·하이퍼파라미터를 직접 제어. 대신 코드를 관리해야 함 |

컨테이너를 직접 가져가면 AWS가 대신 처리해 주던 문제가 그대로 넘어옵니다. **이 문서들이 다루는 것이 대부분 그 문제들입니다.**

!!! abstract "쉽게 말하면"
    지원 목록에 있는 조합이면 운영 부담이 적으므로 관리형이 낫습니다.
    **목록에 없거나**, 학습 코드를 열어 고쳐야 하거나, 최신 모델을 바로 써야 할 때 이 경로를 택합니다.

??? question "오해 — “관리형이 안 되면 그냥 직접 하면 되는 거 아닌가요?”"
    되지만, 관리형이 조용히 처리해 주던 것들이 그대로 넘어옵니다. 이 kit이 실제로 부딪혀 고친 것들입니다.

    - **`save_pretrained`로 저장한 gemma-4 E2B/E4B는 vLLM·SGLang·LMI에서 로드가 실패합니다.** KV-sharing 레이어의 텐서가 저장 과정에서 빠지기 때문입니다 → [서빙 컨테이너](05_serving_containers.md)
    - **학습을 다 마친 Job이 머지 도중 죽습니다.** SDK가 `StoppingCondition`을 생략하면 1시간을 넣는데, 그 창이 머지·업로드까지 덮습니다 → [파인튜닝](03_finetuning.md#maxruntimeexceeded--학습-뒤-머지에서-잘리는-함정)
    - **24GB GPU에서 서빙이 OOM으로 뜨지 않습니다.** vLLM 기본 `max_num_seqs=256`이 실습 규모에 과합니다 → [서빙 컨테이너](05_serving_containers.md)
    - **응답이 조용히 잘립니다.** 예외도 없고 HTTP 200이라, `finish_reason`을 봐야 압니다 → [SageMaker AI 추론](04_sagemaker_inference.md)

    각 문서에 증상 → 원인 → 대응 순으로 정리해 두었습니다.

!!! warning "경로 선택은 실행 직전에 재확인"
    지원 모델·기법·리전은 빠르게 바뀝니다. 위 표는 방향을 잡기 위한 것이고, 실제로 고르기 전에 최신 문서와 콘솔에서 다시 확인하세요.

## 가이드

파일명 번호가 곧 읽는 순서입니다. 처음이라면 위에서부터, 특정 단계만 필요하면 해당 항목으로 가세요.
각 문서가 무엇을 다루고 어느 노트북·참조 코드에 대응하는지는 [전체 지도의 상세 문서 지도](00_overview.md#상세-문서-지도)에 한 표로 정리돼 있습니다.

| 단계 | 문서 |
|---|---|
| 설치 | [시작하기](getting_started.md) |
| 지도 | [00 전체 지도](00_overview.md) |
| 개념 | [01 SageMaker AI 기초](01_sagemaker_basics.md) |
| 데이터 | [02 합성 데이터](02_synthetic_data.md) |
| 학습 | [03 파인튜닝](03_finetuning.md) |
| 배포 | [04 SageMaker AI 추론](04_sagemaker_inference.md) · [05 서빙 컨테이너](05_serving_containers.md) |
| 활용 | [06 Agentic loop](06_agentic.md) |
| 측정 | [속도 측정](benchmark.md) — TTFT/TPOT/ITL. SageMaker AI Endpoint에는 `vllm bench serve`가 없습니다 |
| 완주 | [실행 runbook](RUN_E2E.md) |
| 참조 | [SDK V3](sdk_v3/index.md) — V2에서 바뀐 것 · [학습](sdk_v3/training.md) · [배포](sdk_v3/serving.md) |

## 파이프라인

```
데이터 준비 → 학습(SFT/GRPO) → 배포(endpoint) → 평가(held-out) → agentic loop → 정리
```

| 단계 | 내용 |
|---|---|
| 데이터 | 공개 permissive 시드 + Bedrock Converse로 grounded 합성 |
| 학습 | PyTorch DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA. 추출·분류는 SFT→GRPO 선택 가능 |
| 서빙 | real-time endpoint — vLLM(기본) / SGLang / DJL LMI (셋 다 OpenAI 호환) |
| 평가 | held-out 세트로 endpoint를 직접 호출해 코스별 지표 산출 |
| Agent | Strands Agent + Bedrock Claude, AgentCore Runtime 배포까지 |

기본 모델은 `google/gemma-4-E4B-it`(apache-2.0, ungated)입니다. `MODEL_SIZE`로 `E2B` / `E4B` / `12B` / `26B-A4B` / `31B`를 고르거나 `MODEL_ID`로 임의 모델을 지정할 수 있습니다.

## 코스

| 코스 | 태스크 | 시드 데이터셋 |
|---|---|---|
| [`01_extraction_to_json`](courses/extraction.md) | 텍스트 → 구조화 JSON 추출 | `glaiveai/glaive-function-calling-v2` |
| [`02_classification`](courses/classification.md) | 의도 분류 | `mteb/banking77` |
| [`03_summarization`](courses/summarization.md) | 문서 요약 | `FiscalNote/billsum` |
| [`04_domain_qa`](courses/domain_qa.md) | 도메인 QA / instruction | `databricks/databricks-dolly-15k` |
| [`05_multimodal_extraction`](courses/multimodal.md) | 이미지 → 구조화 JSON (영수증) | `naver-clova-ix/cord-v2` |

코스 이름을 누르면 해당 코스 페이지로 갑니다. 각 페이지에 그 코스가 푸는 문제, 시드 데이터의 변환 전후, 성공 기준 지표, 노트북 순서, 코스별 설정값이 있습니다.

노트북과 학습 스크립트는 [GitHub 리포지토리](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e)에 있습니다. 설치와 실행 방법은 [시작하기](getting_started.md)와 [실행 runbook](RUN_E2E.md)에 있습니다.

!!! warning "비용"
    real-time endpoint는 **삭제할 때까지 시간당 과금**됩니다. 실습을 마치면 각 코스의 `99_cleanup.ipynb`를 반드시 실행하세요.

버전 정보에 대해서도 같은 규율이 필요합니다.

!!! warning "빠르게 바뀌는 값"
    모델 ID·DLC 이미지 태그·SDK 버전·리전 가용성·요금은 빠르게 바뀝니다. 이 문서의 수치는 실측 스냅샷이므로 **실행 직전에 각 소스에서 재확인**하세요. 확인처는 각 문서의 해당 절에 인라인으로 링크되어 있습니다.

## Disclaimer

이 가이드는 저자의 개인 견해와 실측 경험을 정리한 것으로, 저자가 재직 중인 회사의 공식 문서나
입장을 대변하지 않습니다. 내용이 공식 문서와 다를 경우 **공식 문서가 우선합니다.**

문서의 수치와 동작은 특정 시점에 관측한 결과입니다. 실행 전에 각 절에 링크된 원문에서 다시
확인하세요.

## 저자

**김대근 (Daekeun Kim)** / AWS Principal AI Specialist Solutions Architect

[LinkedIn](https://www.linkedin.com/in/daekeun-kim) |
[GitHub](https://github.com/daekeun-ml) |
[Hugging Face](https://huggingface.co/daekeun-ml) |
[기술 블로그](https://housekdk.gitbook.io/)

## License

코드와 문서는 [MIT](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e/blob/master/LICENSE)입니다.
모델과 데이터셋 라이선스는 별개이니 재배포·서빙 전에 각 모델 카드에서 확인하세요 —
gemma-4는 apache-2.0 + ungated지만, gemma-3/2는 커스텀 Gemma Terms + gated이고 use-restriction이
파인튜닝 산출물까지 전파됩니다.
