# SageMaker Fine-tuning & Serving E2E

Gemma 4를 Amazon SageMaker에서 **파인튜닝 → 서빙 → 평가 → agentic loop**까지 잇는 한국어 핸즈온 가이드입니다.
5개 트랙이 각각 독립된 E2E로 동작하므로, 필요한 태스크 하나만 골라 처음부터 끝까지 돌릴 수 있습니다.

!!! tip "어디서부터 읽을까"
    - **설치부터 첫 학습까지** → [시작하기](getting_started.md)
    - **SageMaker가 처음이라면** → [SageMaker 기초](01_sagemaker_basics.md)를 먼저 읽으세요. Training Job과 Endpoint의 수명·과금 차이를 잡아 두면 아래 가이드가 전제하는 개념이 채워집니다.
    - **전체를 완주한다면** → [실행 런북](RUN_E2E.md)에 단계별 핸드오프와 비용이 있습니다.
    - **구조부터 보고 싶다면** → [전체 지도](00_overview.md)에 문서·노트북 매핑이 있습니다.
    - **특정 주제만** 필요하면 아래 가이드에서 골라 보세요.

## 가이드

파일명 번호가 곧 읽는 순서입니다. 처음이라면 위에서부터, 특정 단계만 필요하면 해당 항목으로 가세요.
각 문서가 무엇을 다루고 어느 노트북·참조 코드에 대응하는지는 [전체 지도의 상세 문서 지도](00_overview.md#상세-문서-지도)에 한 표로 정리돼 있습니다.

| 단계 | 문서 |
|---|---|
| 설치 | [시작하기](getting_started.md) |
| 지도 | [00 전체 지도](00_overview.md) |
| 개념 | [01 SageMaker 기초](01_sagemaker_basics.md) |
| 데이터 | [02 합성 데이터](02_synthetic_data.md) |
| 학습 | [03 파인튜닝](03_finetuning.md) |
| 배포 | [04 SageMaker 추론](04_sagemaker_inference.md) · [05 서빙 컨테이너](05_serving_containers.md) |
| 활용 | [06 Agentic loop](06_agentic.md) |
| 완주 | [실행 런북](RUN_E2E.md) |

## 파이프라인

```
데이터 준비 → 학습(SFT/GRPO) → 배포(endpoint) → 평가(held-out) → agentic loop → 정리
```

| 단계 | 내용 |
|---|---|
| 데이터 | 공개 permissive 시드 + Bedrock Converse로 grounded 합성 |
| 학습 | PyTorch DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA. 추출·분류는 SFT→GRPO 선택 가능 |
| 서빙 | real-time endpoint — vLLM(기본) / SGLang / DJL LMI (셋 다 OpenAI 호환) |
| 평가 | held-out 세트로 endpoint를 직접 호출해 트랙별 지표 산출 |
| Agent | Strands Agent + Bedrock Claude, AgentCore Runtime 배포까지 |

기본 모델은 `google/gemma-4-E4B-it`(apache-2.0, ungated)입니다. `MODEL_SIZE`로 `E4B` / `12B` / `26B-A4B`를 고르거나 `MODEL_ID`로 임의 모델을 지정할 수 있습니다.

## 트랙

| 트랙 | 태스크 | 시드 데이터셋 |
|---|---|---|
| `01_extraction_to_json` | 텍스트 → 구조화 JSON 추출 | `glaiveai/glaive-function-calling-v2` |
| `02_classification` | 의도 분류 | `mteb/banking77` |
| `03_summarization` | 문서 요약 | `FiscalNote/billsum` |
| `04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` |
| `05_multimodal_extraction` | 이미지 → 구조화 JSON (영수증) | `naver-clova-ix/cord-v2` |

노트북과 학습 스크립트는 [GitHub 리포지토리](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e)에 있습니다. 설치와 실행 방법은 [시작하기](getting_started.md)와 [실행 런북](RUN_E2E.md)에 있습니다.

!!! warning "비용"
    real-time endpoint는 **삭제할 때까지 시간당 과금**됩니다. 실습을 마치면 각 트랙의 `99_cleanup.ipynb`를 반드시 실행하세요.

버전 정보에 대해서도 같은 규율이 필요합니다.

!!! warning "빠르게 바뀌는 값"
    모델 ID·DLC 이미지 태그·SDK 버전·리전 가용성은 빠르게 바뀝니다. 각 문서의 수치에는 관측일이 붙어 있으니, 그 날짜를 보고 **실행 직전에 각 소스에서 재확인**하세요. 확인처는 각 문서의 해당 절에 인라인으로 링크되어 있습니다.
