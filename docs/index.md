# SageMaker AI Fine-tuning & Serving E2E

Gemma 4를 Amazon SageMaker AI에서 **합성 데이터 → 파인튜닝 → 서빙 → 평가 → agentic loop**까지 연결하는 한국어 hands-on 가이드입니다.
5가지 **유스케이스별 실습 코스**가 각각 독립된 E2E로 구성되어 있어, 필요한 유스케이스만 선택해 실행할 수 있습니다.

!!! tip "어디서부터 읽을까"
    - **설치부터 첫 학습까지** → [시작하기](getting_started.md)
    - **SageMaker AI가 처음이라면** → [SageMaker AI 기초](01_sagemaker_basics.md)를 먼저 읽으세요. Training Job과 Endpoint의 수명과 과금 차이를 알아 두면 나머지 가이드를 이해하기 쉽습니다.
    - **전체 과정을 실행한다면** → [E2E 실행 가이드](RUN_E2E.md)에 단계별 핸드오프와 비용이 있습니다.
    - **구조부터 보고 싶다면** → [전체 지도](00_overview.md)에 문서와 노트북 매핑이 있습니다.
    - **어떤 유스케이스를 실행할지 정해야 한다면** → 아래 [유스케이스별 실습](#5가지-유스케이스별-실습) 표에서 고르세요.
    - **특정 주제만** 필요하면 아래 가이드에서 골라 보세요.

## 개발 배경

SageMaker AI에는 JumpStart, Model Customization, HyperPod Recipes와 커스텀 학습 작업 등 여러 파인튜닝 경로가 있습니다. 지원되는 모델과 학습 방식으로 빠르게 시작하려면 관리형 경로가 적합합니다. 학습과 서빙 코드를 직접 수정하거나 여러 유스케이스에 같은 E2E 구조를 적용하려면 커스텀 경로가 필요합니다.

| 경로 | 적합한 경우 | 고려 사항 |
|---|---|---|
| [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use-studio-updated-fine-tune.html) | 모델 카드에서 지원하는 배포와 파인튜닝을 빠르게 실행 | 모델마다 지원 기능과 하이퍼파라미터가 다름 |
| [SageMaker Model Customization](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight.html) | 지원 모델을 관리형 SFT, DPO와 강화학습 방식으로 파인튜닝 | 지원 모델, 학습 방식과 리전 범위 안에서 사용 |
| [SageMaker HyperPod Recipes](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html) | 사전 구성된 분산 학습 스택과 레시피를 사용 | 지원 레시피, 모델과 인스턴스 조합을 확인해야 함 |
| 이 에셋 | 모델, 데이터 변환, 학습, 서빙과 평가 코드를 직접 제어 | 컨테이너와 모델 호환성까지 직접 관리 |

이 에셋은 관리형 기능을 대체하려는 것이 아닙니다. 하나의 코드 구조로 5가지 유스케이스를 실행하고, 데이터 준비부터 파인튜닝, 배포, 평가와 실험 추적까지 직접 수정할 수 있는 참조 구현을 제공하는 것이 목적입니다.

??? info "커스텀 경로에서 직접 처리할 문제"
    이 에셋을 구현하고 검증하면서 다음 문제를 코드와 설정에서 직접 처리했습니다.

    - **[모델 저장](05_serving_containers.md):** Gemma 4 E2B/E4B를 `save_pretrained`로 저장할 때 KV-sharing 텐서가 누락되면 vLLM, SGLang과 DJL LMI가 모델을 불러오지 못합니다.
    - **[작업 제한 시간](03_finetuning.md#maxruntimeexceeded-학습-뒤-머지에서-잘리는-함정):** SDK의 기본 1시간에는 학습, 모델 병합과 S3 업로드가 모두 포함됩니다. `StoppingCondition`을 충분히 길게 지정해야 합니다.
    - **[GPU 메모리](05_serving_containers.md):** vLLM의 기본 `max_num_seqs=256`은 24GB GPU에서 CUDA OOM을 일으킬 수 있습니다. 인스턴스에 맞게 값을 낮춰야 합니다.
    - **[응답 잘림](04_sagemaker_inference.md):** HTTP 200이 반환되어도 출력이 길이 제한에 걸릴 수 있습니다. `finish_reason`을 확인해야 합니다.

!!! note "지원 범위 확인"
    지원 모델, 학습 방식과 리전은 변경될 수 있습니다. 실행 전에 각 경로의 공식 문서와 모델 카드를 확인하세요.

## 가이드

처음이라면 시작하기와 기본 개념을 먼저 읽고, 이후 목적에 맞는 가이드를 선택하세요. `02~06`은 데이터 준비부터 Agentic loop까지 이어지는 기술 가이드이며, 실행과 실험 관리 문서는 별도로 구성되어 있습니다.

| 구분 | 가이드 | 내용 |
|---|---|---|
| 시작 | [시작하기](getting_started.md) | 설치, 설정과 첫 실행 |
| 기본 개념 | [00 전체 지도](00_overview.md), [01 SageMaker AI 기초](01_sagemaker_basics.md) | 저장소 구조, Training Job과 Endpoint |
| 모델 구축 | [02 합성 데이터](02_synthetic_data.md), [03 파인튜닝](03_finetuning.md) | 학습 데이터 준비와 모델 파인튜닝 |
| 모델 배포 | [04 SageMaker AI 추론](04_sagemaker_inference.md), [05 서빙 컨테이너](05_serving_containers.md) | Endpoint 배포, 호출과 서빙 엔진 선택 |
| 활용 | [06 Agentic loop](06_agentic.md) | 파인튜닝 모델과 Bedrock을 연결한 에이전트 구성 |
| 직접 실행 | [노트북 E2E](RUN_E2E.md), [Python 파이프라인](pipelines.md) | 단계별 실습 또는 무인 실행 |
| 실험 관리 | [SageMaker Managed MLflow](mlflow.md), [속도 측정](benchmark.md) | 실험 기록 비교와 TTFT, TPOT, ITL 측정 |
| SDK 참조 | [SDK V3 개요](sdk_v3/index.md), [학습](sdk_v3/training.md), [배포](sdk_v3/serving.md) | V2와의 차이와 V3 API 사용법 |

문서와 노트북, 참조 코드의 대응 관계는 [상세 문서 지도](00_overview.md#상세-문서-지도)에서 확인할 수 있습니다.

## 5가지 유스케이스별 실습

이 저장소는 하나의 예제만 제공하지 않습니다. 같은 데이터 준비, 학습, 배포와 평가 구조를 사용해 아래 5가지 유스케이스를 각각 처음부터 끝까지 실행할 수 있습니다.

| 유스케이스 | 실습 코스 | 시드 데이터셋 |
|---|---|---|
| 텍스트 구조화 추출 | [`01_extraction_to_json`](courses/extraction.md) | `glaiveai/glaive-function-calling-v2` |
| 의도 분류 | [`02_classification`](courses/classification.md) | `mteb/banking77` |
| 문서 요약 | [`03_summarization`](courses/summarization.md) | `FiscalNote/billsum` |
| 도메인 질의응답 | [`04_domain_qa`](courses/domain_qa.md) | `databricks/databricks-dolly-15k` |
| 이미지 구조화 추출 | [`05_multimodal_extraction`](courses/multimodal.md) | `naver-clova-ix/cord-v2` |

실습 코스 이름을 누르면 해당 유스케이스의 문제 정의, 데이터 변환 전후, 성공 기준, 노트북 순서와 설정값을 확인할 수 있습니다.

노트북과 학습 스크립트는 [GitHub 저장소](https://github.com/daekeun-ml/sagemaker-finetune-serve-e2e)에 있습니다. 설치와 실행 방법은 [시작하기](getting_started.md)와 [E2E 실행 가이드](RUN_E2E.md)에 있습니다.

!!! warning "비용"
    real-time endpoint는 **삭제할 때까지 시간당 과금**됩니다. 실습을 마치면 각 코스의 `99_cleanup.ipynb`를 반드시 실행하세요.

!!! warning "빠르게 바뀌는 값"
    모델 ID, DLC 이미지 태그, SDK 버전, 리전 가용성, 요금은 빠르게 바뀝니다. 이 문서의 수치는 특정 시점의 측정값이므로 **실행 직전에 연결된 공식 문서에서 다시 확인**하세요.

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
모델과 데이터셋 라이선스는 별개이니 재배포와 서빙 전에 각 모델 카드에서 확인하세요:
gemma-4는 apache-2.0 + ungated지만, gemma-3/2는 커스텀 Gemma Terms + gated이고 use-restriction이
fine-tuning artifact까지 전파됩니다.
