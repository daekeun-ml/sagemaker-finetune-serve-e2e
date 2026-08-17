# 노트북 실행법

!!! note "흔한 오해: SageMaker 노트북은 필수가 아닙니다"
    SageMaker Notebook Instance나 Studio를 별도로 만들 필요는 없습니다. 필요한 Python 패키지, AWS 자격증명과 IAM 권한이 있으면 랩탑, EC2, 사내 서버 등 어떤 개발 환경에서도 실행할 수 있습니다.

    노트북은 SageMaker 제어 영역(control plane) API를 호출하는 클라이언트입니다. 실제 학습과 추론은 개발 환경과 분리된 AWS 관리형 Training Job과 Endpoint에서 실행됩니다.

    control plane과 학습 데이터의 네트워크 경로는 [SageMaker AI 보안과 네트워크](../concepts/03_sagemaker_security_network.md)에서 구분해 설명합니다.

!!! info "Scope"
    코스 하나를 노트북으로 실행하는 순서와 단계별 결과 전달을 간단히 설명합니다.

    설치와 환경 설정은 [시작하기](../getting_started.md), Python 자동 실행은 [Python 스크립트 실행법](run_pipeline.md)에서 확인하세요.

## 코스 선택

| 유스케이스 | 노트북 디렉터리 |
|---|---|
| [텍스트 구조화 추출](../courses/extraction.md) | `tracks/01_extraction_to_json/` |
| [의도 분류](../courses/classification.md) | `tracks/02_classification/` |
| [문서 요약](../courses/summarization.md) | `tracks/03_summarization/` |
| [도메인 질의응답](../courses/domain_qa.md) | `tracks/04_domain_qa/` |
| [이미지 구조화 추출](../courses/multimodal.md) | `tracks/05_multimodal_extraction/` |

코스는 서로 독립적입니다. 처음에는 `01_extraction_to_json` 하나를 실행하고 cleanup까지 마친 뒤 다른 코스로 이동하는 편이 안전합니다.

## 실행 전 확인

- [ ] `.venv` 설치와 스모크 테스트를 완료했습니다.
- [ ] AWS 자격증명과 SageMaker AI 실행 역할을 준비했습니다.
- [ ] 사용할 리전의 학습 및 추론 인스턴스 쿼터를 확인했습니다.
- [ ] 텍스트 코스에서 합성 데이터나 에이전트를 사용한다면 Bedrock 모델 접근 권한이 있습니다.
- [ ] `.env`와 `config.yaml`의 모델, 인스턴스와 서빙 엔진을 확인했습니다.

## 노트북 DRY_RUN { #두-가지-검증-모드-구분 }

`DRY_RUN=1`은 데이터, 평가와 로컬 학습 규모를 줄입니다. SageMaker AI Training Job과 Endpoint 셀을 실행하면 실제 AWS 리소스와 비용이 발생합니다.

AWS 리소스를 만들지 않고 전체 흐름만 확인하려면 Python 파이프라인의 `--dry-run`을 사용하세요.

```bash
python pipelines/run_extraction.py --stages all --dry-run
```

## 단계별 실행과 데이터 핸드오프

같은 코스 디렉터리에서 번호 순서대로 노트북을 실행합니다.

| 단계 | 노트북 | 결과와 다음 단계 전달 |
|---|---|---|
| 1 | `00_setup.ipynb` | `%store`에 `role`, `bucket` 저장 |
| 2 | `01_data_and_synthetic.ipynb` | `data/train.jsonl`과 held-out 데이터 생성 |
| 3 | `02_train_sft_sagemaker.ipynb` | `%store`에 코스별 `model_data` 저장 |
| 3-a | `02a_train_grpo_sagemaker.ipynb` | 추출과 분류 코스에서 `model_data`를 GRPO 결과로 갱신 |
| 3-b | `02b_local_serve.ipynb` | 선택적으로 로컬 vLLM 응답 확인 |
| 4 | `03_deploy_endpoint.ipynb` | `%store`에 코스별 `endpoint_name` 저장 |
| 5 | `04_evaluate.ipynb` | held-out 데이터의 평가 지표 출력 |
| 6 | `05_agentic_strands.ipynb` | 선택적으로 로컬 에이전트 실행 |
| 7 | `06_agentcore_deploy.ipynb` | 선택적으로 AgentCore Runtime 배포 |
| 8 | `99_cleanup.ipynb` | Endpoint, EndpointConfig와 Model 삭제 |

학습과 평가만 수행하려면 `00`, `01`, `02`, `03`, `04`, `99` 순서로 실행하면 됩니다.

`train_path`는 각 코스의 `data/train.jsonl`을 사용합니다. `model_data`와 `endpoint_name`은 코스별 `%store` 키를 먼저 읽으므로 다른 코스의 값을 사용하지 않는지 확인하세요.

Training Job과 Endpoint 생성은 AWS에서 계속 진행됩니다. 커널을 종료하거나 연결이 끊겨도 리소스는 멈추지 않으므로 다시 제출하기 전에 AWS 상태를 확인하세요.

### 멀티모달 코스 (05) 파이프라인

멀티모달 코스는 합성 데이터, GRPO, 로컬 vLLM, 별도 평가와 에이전트 단계가 없습니다.

| 단계 | 노트북 |
|---|---|
| 1 | `00_setup.ipynb` |
| 2 | `01_data_explore.ipynb` |
| 3 | `02_train_mm_sagemaker.ipynb` |
| 4 | `03_deploy_mm_endpoint.ipynb` |
| 5 | `99_cleanup.ipynb` |

## 완료와 정리

- 학습 Job이 `Completed`이고 모델 아티팩트가 생성되었습니다.
- Endpoint가 `InService`이고 테스트 요청에 정상 응답합니다.
- 텍스트 코스는 held-out 평가 결과를 확인했습니다.
- 실습이 끝나면 `99_cleanup.ipynb`를 실행했습니다.
- AgentCore Runtime을 배포했다면 `bash agentcore/cleanup_agent.sh --aws`로 정리했습니다.

Real-time Endpoint는 삭제할 때까지 인스턴스 비용이 발생합니다. 여러 번 배포했거나 리전을 변경했다면 AWS 콘솔에서 남은 Endpoint도 확인하세요.
