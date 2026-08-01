# 01 · SageMaker 기초 — Training Job과 Endpoint

!!! info "읽는 사람과 범위"
    Python은 쓰고 Jupyter도 써 봤지만 **SageMaker는 처음**인 ML 엔지니어를 위한 문서입니다. AWS 인프라 지식은 필요 없습니다.
    선행 조건: 없습니다. [시작하기](getting_started.md)로 설치를 마쳤다면 [실행 런북](RUN_E2E.md)으로 넘어가기 전에 이 문서를 읽으시면 노트북이 무엇을 하고 있는지 보입니다.
    다루는 것: **Training Job**과 **Endpoint** 두 가지 개념, 컨테이너 경로 계약, 시간 제한, 그리고 HyperPod / EC2 / on-prem과의 차이.
    다루지 않는 것: SageMaker의 모든 기능(Studio·Pipelines·Feature Store·Clarify 등)은 의도적으로 생략합니다. 추론 옵션 상세는 [SageMaker 추론](04_sagemaker_inference.md), 학습 상세는 [파인튜닝](03_finetuning.md)이 다룹니다.

!!! warning "빠르게 바뀌는 값"
    인스턴스 요금·리전 가용성·서비스 한도·GA 상태·Serverless의 GPU 지원 여부는 분기마다 바뀝니다.
    이 문서는 **개념과 계약**(경로·수명·과금 모델)에 집중하고 구체적인 수치는 최소한만 언급합니다. 수치를 인용할 때는 전부 **실행 직전 재확인** 대상이며, 확인처는 각 주장 옆에 붙은 공식 문서 링크입니다.

---

## TL;DR

**SageMaker AI에서 기억할 개념은 사실상 두 개입니다. Training Job은 "돌리고 나면 사라지는 계산"이고, Endpoint는 "지울 때까지 켜져 있는 서버"입니다. 이 둘의 수명과 과금 방식이 정반대라는 점만 잡으면 이 킷의 노트북 전체가 이해됩니다.**

정리하면 다음과 같습니다.

1. **Training Job은 컨테이너 이미지 + 내 스크립트 + S3 데이터 위치를 넘기면 SageMaker가 머신을 띄우고 스크립트를 돌린 뒤 머신을 파괴하는 잡입니다.** 과금은 잡 단위이며 끝나면 자동으로 멈춥니다 — [Training Job](#training-job--잡이-끝나면-사라지는-계산).
2. **입력과 출력은 컨테이너 안의 정해진 경로로 주고받습니다.** 입력은 `SM_CHANNEL_TRAIN`(`/opt/ml/input/data/train`), 출력은 `SM_MODEL_DIR`(`/opt/ml/model`)이고, 후자에 남은 것만 `model.tar.gz`가 되어 S3로 올라갑니다 — [경로 계약](#경로-계약--컨테이너-안의-정해진-경로).
3. **`MaxRuntimeInSeconds`는 학습 코드 시간이 아니라 인스턴스에서 잡이 시작된 뒤의 전 구간을 덮습니다**(이미지 pull → 데이터 복사 → 학습 → 저장 → 업로드). 이 킷은 이 한도 때문에 **학습이 100% 끝난 잡을 잃은 적이 있습니다** — [MaxRuntimeInSeconds가 덮는 시간 창](#maxruntimeinseconds가-덮는-시간-창).
4. **Endpoint는 상시 HTTP 서버입니다. 호출이 0건이어도 삭제 전까지 시간당 과금됩니다.** 초심자에게 가장 비싼 오해가 바로 이 지점입니다 — [Endpoint](#endpoint--삭제할-때까지-켜져-있는-서버).
5. **SageMaker AI / HyperPod / EC2 / on-prem은 "무엇을 내가 소유하는가"로 갈립니다.** 잡 단위로 빌릴지, 클러스터를 유지할지, 인프라까지 직접 만들지의 선택입니다 — [SageMaker vs HyperPod vs EC2 vs on-prem](#sagemaker-vs-hyperpod-vs-ec2-vs-on-prem).

---

## 기존 Pain Point

SageMaker를 처음 여는 분들이 실제로 막히는 지점은 다음과 같습니다.

- "노트북에서 `trainer.train()`을 눌렀는데 **내 코드가 어디서 도는지 모르겠습니다.**" — 로컬 커널이 아니라 SageMaker가 별도로 띄운 컨테이너에서 돕니다. 그래서 로컬 파일 경로가 통하지 않습니다.
- "학습 결과가 **어디로 갔는지** 모르겠습니다." — `/opt/ml/model`에 쓴 것만 S3로 올라갑니다. 다른 곳에 저장하면 잡이 끝날 때 인스턴스와 함께 사라집니다.
- "잡 상태가 `Failed`도 아니고 **`Stopped`인데 에러 로그가 없습니다.**" — 시간 제한(`MaxRuntimeExceeded`)에 걸린 것이며, 학습이 성공했어도 산출물이 불완전할 수 있습니다.
- "**테스트만 했는데 다음 날 청구서가 왔습니다.**" — Training Job과 Endpoint의 수명을 같다고 생각한 결과입니다. 전자는 자동으로 멈추지만 후자는 멈추지 않습니다.
- "**HyperPod가 더 좋은 거 아닌가요?**" — 더 좋은 것이 아니라 다른 층입니다. 잡 하나를 돌리려고 클러스터를 만들면 운영 부담만 늘어납니다.
- "회사에 GPU 서버가 있는데 **굳이 클라우드를 쓸 이유**가 있나요?" — 사용률과 데이터 소재 요건에 따라 실제로 없을 수도 있습니다. 이 문서는 그 판단 기준을 제시합니다.

이 문서는 위 여섯 가지를 개념 층위에서 해소합니다.

---

## Training Job — 잡이 끝나면 사라지는 계산

!!! abstract "쉽게 말하면"
    Training Job은 **렌터카**에 가깝습니다. 차를 사는 것이 아니라, 목적지(스크립트)와 짐(데이터)을 주면
    SageMaker가 차를 빌려 오고, 운행이 끝나면 트렁크에 실린 것만 창고(S3)에 옮긴 뒤 차를 반납합니다.
    차는 사라지므로 **트렁크에 넣지 않은 것은 전부 버려집니다.** 요금은 운행한 시간만큼입니다.

AWS는 이 방식을 [관리형 학습(how it works)](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)으로 문서화합니다. 내가 SageMaker에 넘기는 것은 세 가지입니다.

| 넘기는 것 | 무엇 | 이 킷에서는 |
|---|---|---|
| **컨테이너 이미지** | 학습 환경(파이썬·CUDA·프레임워크)이 들어 있는 Docker 이미지 | AWS가 게시한 PyTorch **DLC**(Deep Learning Containers — AWS가 미리 빌드해 ECR에 올려 둔 학습/추론용 컨테이너 이미지, `.env`의 `DLC_IMAGE_URI`) |
| **내 코드** | 진입 스크립트와 `requirements.txt`가 든 디렉터리 | `SourceCode(source_dir='scripts', entry_script='train.py')` |
| **데이터 위치** | S3 URI. **채널**(= 컨테이너 안에서 데이터가 마운트될 이름) 이름과 함께 지정 | `InputData(channel_name='train', data_source=train_s3)` |

그러면 SageMaker가 순서대로 다음을 합니다.

```
1. 인스턴스 프로비저닝        (상태: Pending — GPU 용량을 기다리는 구간)
2. 이미지 pull                (상태: Starting — StatusMessage에만 "Downloading the training image"로 드러남)
3. S3 데이터를 컨테이너로 복사  (상태: Downloading) → /opt/ml/input/data/train
4. entry_script 실행          (상태: Training — 여기서부터 CloudWatch 로그가 생김)
5. /opt/ml/model 을 tar.gz로 압축해 S3 업로드   (상태: Uploading) ← 이것이 model_data
6. 인스턴스 종료 + 과금 중지    (상태: Completed / Failed / Stopped)
```

[`DescribeTrainingJob`의 secondary status 정의](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html)를 보면 `Downloading`은 이미지가 아니라 **데이터**를 받는 구간입니다(`File` 입력 모드에서 S3 → ML 스토리지 볼륨). 이미지 다운로드는 별도 상태가 아니라 `StatusMessage` 문구로만 보입니다. 그리고 5번 `Uploading`이 이 킷이 실제로 다친 구간입니다 — 뒤의 [시간 제한](#maxruntimeinseconds가-덮는-시간-창)을 보세요.

여기서 초심자가 놓치기 쉬운 두 가지가 있습니다.

- **내 코드는 로컬 커널이 아니라 격리된 컨테이너에서 돕니다.** 그래서 노트북 옆의 파일을 열 수 없고, 데이터를 미리 S3에 올려야 합니다(이 킷의 `02_train_sft_sagemaker`가 `upload_if_changed()`로 처리합니다). 같은 이유로 `train.py`는 `common/`을 import하지 않는 **self-contained** 파일이어야 합니다 — SageMaker가 컨테이너에 올리는 것은 `source_dir` 하나뿐입니다([train.py 상세](03_finetuning.md#trainpy--로컬-dry-run과-sagemaker-학습-잡)).
- **과금 대상 시간은 학습 시간보다 넓습니다.** AWS 문서 기준으로 **데이터 다운로드 시간**과 **모델 아티팩트 압축·업로드 시간**도 billable time에 포함됩니다. 즉 "학습 5분"이 "요금 5분"은 아닙니다.

### 경로 계약 — 컨테이너 안의 정해진 경로

**이 킷의 모든 노트북이 이 계약 위에 서 있습니다.** SageMaker는 컨테이너 안의 정해진 경로를 통해서만 데이터를 주고받고, 각 경로에 대응하는 환경변수를 심어 줍니다. 경로별 역할은 [학습 스토리지 경로 매핑](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html) 문서에 정의돼 있습니다.

| 컨테이너 경로 | [환경변수](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html) | 용도 | 잡이 끝날 때 |
|---|---|---|---|
| `/opt/ml/input/data/<채널명>` | `SM_CHANNEL_<채널명>` | 입력 데이터(S3에서 복사됨) | — (읽기용) |
| `/opt/ml/input/data/code` | `SM_SOURCE_DIR` (+ `SM_ENTRY_SCRIPT`) | 업로드된 내 코드 | — (읽기용) |
| `/opt/ml/model` | `SM_MODEL_DIR` | **최종 모델 아티팩트** | ✅ `tar.gz`로 압축 후 S3 업로드 |
| `/opt/ml/output/data` | `SM_OUTPUT_DATA_DIR` | loss·중간 산출물 등 부가 출력 | ✅ 별도 `tar.gz`로 업로드 |
| [`/opt/ml/output/failure`](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html) | `SM_OUTPUT_FAILURE` | 실패 이유를 적는 파일 | 앞부분이 잡의 `FailureReason`이 됨 |
| [`/opt/ml/checkpoints`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) | (없음) | 체크포인트 | `CheckpointConfig(S3Uri=...)`를 **지정한 경우에만** 학습 중 S3와 동기화(압축 안 함) |
| `/tmp` | (없음) | 임시 작업 공간 | ❌ 업로드되지 않음 — 인스턴스와 함께 삭제 |

두 가지를 짚어 둡니다. 첫째, `SM_OUTPUT_DIR`은 `/opt/ml/output/data`가 아니라 **그 부모인 `/opt/ml/output`**입니다(그 아래에 `data`와 `failure`가 있습니다). `os.environ['SM_OUTPUT_DIR']`에 파일을 쓰면 `output.tar.gz`에 들어가지 않으니, 부가 출력은 `SM_OUTPUT_DATA_DIR`에 쓰세요. 둘째, 코드가 내려오는 경로는 SDK 세대에 따라 다릅니다 — 이 킷이 쓰는 [`ModelTrainer`](https://sagemaker.readthedocs.io/en/stable/)(SageMaker Python SDK v3)는 `source_dir`을 `code`라는 **입력 채널로 올려** `/opt/ml/input/data/code`에 마운트하고 거기서 `cd` 후 실행합니다. 그래서 `train.py` 안의 상대 경로는 그 디렉터리를 기준으로 풀립니다. 레거시 Estimator·추론 컨테이너에서 보이는 `/opt/ml/code` + `SAGEMAKER_SUBMIT_DIRECTORY`는 다른 계약입니다.

채널 이름은 내가 정합니다. `channel_name='train'`으로 주면 컨테이너에서 `SM_CHANNEL_TRAIN=/opt/ml/input/data/train`이 됩니다. 이 킷의 `train.py`가 하는 일도 정확히 그것입니다.

```python
# tracks/01_extraction_to_json/scripts/train.py — 입력
ch = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
for fn in sorted(os.listdir(ch)):          # 채널 디렉터리에서 첫 .jsonl
    if fn.endswith(".jsonl"):
        return os.path.join(ch, fn)

# 같은 파일 — 출력. 로컬 dry-run이면 ./out, SageMaker면 /opt/ml/model
p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./out"))
```

환경변수에 **기본값을 함께 준 것**이 요령입니다. 덕분에 같은 파일이 로컬 GPU에서도 그대로 돌아갑니다(`--train_file`을 직접 주면 채널을 보지 않습니다).

!!! danger "여기에 저장하지 않으면 결과가 사라집니다"
    `/opt/ml/model`에 없는 것은 **잡이 끝날 때 인스턴스와 함께 삭제됩니다.** `/tmp`나 현재 디렉터리에 저장한 모델은 회수할 방법이 없습니다.
    반대로 `/opt/ml/model`에 **필요 없는 것을 두면 손해**입니다. 이 디렉터리 전체가 압축되므로 중간 체크포인트가 쌓이면 업로드 시간이 늘고, 그 시간은 요금과 [시간 제한](#maxruntimeinseconds가-덮는-시간-창)을 함께 잡아먹습니다. 이 킷이 `SFTConfig(save_total_limit=1)`을 쓰는 이유입니다(실측: 체크포인트 3개 = 0.7GB, 전부 서빙에 불필요).

이 경로는 학습에서 끝나지 않고 **서빙까지 이어집니다.** `/opt/ml/model`은 배포 시 추론 컨테이너가 모델을 읽는 경로이기도 합니다. 그래서 이 킷의 `train.py`는 머지된 모델을 하위 폴더가 아니라 **아티팩트 루트**에 저장합니다 — 루트에 완전한 HF 모델(`config.json` + 가중치)이 없으면 vLLM이 엔진을 감지하지 못합니다([merge_adapter 상세](03_finetuning.md#merge_adapter--서빙-단순화)).

### MaxRuntimeInSeconds가 덮는 시간 창

`StoppingCondition(max_runtime_in_seconds=...)`은 **폭주 방지 상한**입니다. [StoppingCondition API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에 따르면 잡이 이 시간을 넘기면 SageMaker가 `SIGTERM`을 보내고 **120초를 준 뒤** 잡을 종료합니다.

중요한 것은 **이 시계가 학습 코드만 재지 않는다**는 점입니다. 시계는 잡이 **인스턴스에서 시작된 순간**(`TrainingStartTime`)부터 돌고, 그 앞의 용량 대기(`Pending`)는 이 한도가 아니라 별도 파라미터인 `MaxPendingTimeInSeconds`가 덮습니다.

```
   Pending          |<-------------- MaxRuntimeInSeconds -------------->|
 (용량 대기 —        이미지 pull → 데이터 복사 → 학습 루프 → 머지/저장 → 아티팩트 업로드
  MaxPendingTime)                                              ↑
                                     여기서 잘리면 "학습은 성공했는데 배포 불가"
```

이 킷은 실제로 이 함정에 빠졌습니다. `stopping_condition`을 생략하면 SDK가 1시간을 자동으로 넣는데(SDK 3.16.0 실측), 189/189 step을 전부 마친 잡이 **LoRA 머지 도중** 강제 종료되어 아티팩트에 어댑터와 체크포인트만 남았습니다. 상태는 `Failed`가 아니라 `Stopped`이고 `FailureReason`은 비어 있어서, CloudWatch 로그만 보면 정상 종료처럼 보입니다. 실측 타임라인과 대응은 [MaxRuntimeExceeded 함정](03_finetuning.md#maxruntimeexceeded--학습-뒤-머지에서-잘리는-함정)에 정리돼 있습니다.

!!! tip "한도는 넉넉히, 규모는 작게"
    한도를 크게 잡아도 **추가 요금이 없습니다.** 잡이 정상 종료되면 그 시점에 과금이 멈추기 때문입니다. 비용을 줄이고 싶다면 한도가 아니라 **데이터 건수와 epoch**를 줄이세요.
    이 킷의 노트북은 `MAX_RUNTIME_HOURS = 4`를 명시하고, 제출 전에 예상 시간을 계산해 한도를 넘으면 `assert`로 막습니다.

??? tip "함께 알아 두면 좋은 잡 레벨 옵션 (참고용)"
    이 킷이 기본으로 쓰지는 않지만, Training Job에는 문서화된 비용·복원력 옵션이 있습니다. 값과 동작은 **실행 전 재확인**하세요.

    - **[`MaxPendingTimeInSeconds`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)** — 용량을 기다리는 `Pending` 상태의 상한입니다. `MaxRuntimeInSeconds`와 **별개 파라미터**이며, API 유효 범위의 최솟값이 7,200초입니다. GPU 용량 대기로 잡이 무한정 걸려 있는 것을 막습니다.
    - **[Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)** — `EnableManagedSpotTraining=True` + `MaxWaitTimeInSeconds`(≥ `MaxRuntimeInSeconds`)로 켭니다. 절감률은 AWS 문서끼리도 수치가 갈리므로(같은 시점에 DG는 "최대 90%", `CreateTrainingJob` API는 "최대 80%") 절대값으로 약속하지 말고 **잡이 끝난 뒤 실측**하세요 — `(1 - BillableTimeInSeconds / TrainingTimeInSeconds) * 100`이 그 잡의 실제 절감률입니다. Spot은 중단될 수 있으므로 체크포인트를 남기는 것이 권장 구성인데, `/opt/ml/checkpoints`에 쓰는 것만으로는 부족하고 **`CheckpointConfig(S3Uri=...)`를 함께 지정**해야 S3로 동기화됩니다(지정하지 않으면 그냥 로컬 디스크라 인스턴스와 함께 사라집니다). 참고로 Spot과 warm pool은 함께 쓸 수 없습니다.
    - **[Warm pool](https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html)** — `keep_alive_period_in_seconds`(잡당 최대 3,600초)를 켜면 잡이 끝난 뒤에도 인스턴스가 살아 있어 다음 잡의 프로비저닝을 건너뜁니다. **살아 있는 동안 계속 과금되는 리소스**이므로, "Training Job은 끝나면 과금이 멈춘다"는 원칙의 유일한 예외입니다.
    - **[`RetryStrategy`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_RetryStrategy.html)** — `InternalServerError`로 실패한 잡을 자동 재시도합니다(`MaximumRetryAttempts` 1~30). 단 `RetryStrategy`를 쓰면 `MaxRuntimeInSeconds`는 **개별 시도가 아니라 전체 시도 합계**에 적용됩니다.
    - **[`InfraCheckConfig`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_InfraCheckConfig.html)** — `EnableInfraCheck=True`로 켜면 학습 시작 전에 SageMaker가 인스턴스 하드웨어와 클러스터 네트워크 연결을 점검합니다(SDK v3에서는 `ModelTrainer.with_infra_check_config()`). 검증 깊이는 공개되지 않아 HyperPod의 deep health check만큼 상세하다고 보기는 어렵습니다.
    - **`MaxRuntimeInSeconds` 자체의 범위** — API 기본값은 1일, 최대 28일입니다(메트릭 발행·아티팩트 업로드까지 포함한 총 실행 시간 상한은 30일). 1시간은 API가 아니라 **SDK 쪽 기본값**입니다.

---

## Endpoint — 삭제할 때까지 켜져 있는 서버

!!! abstract "쉽게 말하면"
    Endpoint는 렌터카가 아니라 **월세 가게**입니다. SageMaker가 GPU 인스턴스를 띄우고 그 위에 서빙 컨테이너를 올려
    HTTP를 받는 상태로 **계속 유지**합니다. 손님이 하루에 한 명도 안 와도 월세는 나갑니다.
    이 킷의 구성(`ModelBuilder` + 평범한 production variant)에서 문을 닫는 방법은 하나뿐입니다 — **삭제**.

배포는 세 리소스로 이루어집니다. `Model`(어떤 가중치를 어떤 컨테이너로), `EndpointConfig`(어떤 인스턴스로 몇 대), `Endpoint`(실제로 떠 있는 서버). 이 3층 구조와 `invoke_endpoint` 호출 스키마는 [Endpoint 3층 구조와 호출](04_sagemaker_inference.md#endpoint-3층-구조와-호출)에서 자세히 다루므로 여기서는 반복하지 않습니다.

### 컨테이너 계약 — 모델 아티팩트와 ping health check

Training Job에 경로 계약이 있듯이 Endpoint에도 계약이 있고, 학습 쪽 계약과 **한 지점에서 맞물립니다.**

- **모델 아티팩트는 `/opt/ml/model`에 풀립니다.** [추론 컨테이너 규약 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)대로 SageMaker가 S3의 `model.tar.gz`를 내려받아 컨테이너 시작 **전에** 이 경로로 압축을 풉니다. 컨테이너는 이 디렉터리에 **읽기 전용**으로 접근합니다. 서빙 엔진에게는 "그 경로를 모델로 로드하라"고 알려 주면 됩니다(이 킷은 `SM_VLLM_MODEL=/opt/ml/model`).
- **컨테이너는 8080 포트에서 `/invocations`와 `/ping`을 받아야 합니다.** `/invocations`가 추론, `/ping`이 health check입니다. AWS DLC(vLLM(기본)·SGLang·DJL LMI)는 이 계약을 이미 구현하고 있으므로 직접 만들 일은 거의 없습니다. 셋 중 무엇을 쓸지는 `.env`의 `SERVING_ENGINE`으로 고르며, 비교는 [서빙 컨테이너](05_serving_containers.md)가 다룹니다.
- **컨테이너 시작 후 일정 시간 안에 `/ping`이 200을 돌려주지 못하면 배포가 실패합니다.** 기본값은 8분이지만 고정된 상한이 아니라 **[`ProductionVariant.ContainerStartupHealthCheckTimeoutInSeconds`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html)(60~3,600초)로 올릴 수 있습니다** — vLLM 엔진 초기화 + 가중치 로드는 8분을 넘기기 쉬우므로 AWS도 LMI 계열에서는 올리라고 권장합니다(모델이 크면 `ModelDataDownloadTimeoutInSeconds`도 같은 범위에서 함께 올립니다). 개별 `/ping` 요청 자체의 타임아웃은 2초입니다. 이것이 배포 실패 메시지가 종종 `did not pass the ping health check`로만 보이는 이유입니다 — 실제 원인(예: CUDA OOM)은 **CloudWatch endpoint 로그에만** 남습니다([24GB GPU CUDA OOM](04_sagemaker_inference.md#24gb-gpu-cuda-oom--max_num_seqs-기본값)).

학습이 만든 `/opt/ml/model`이 서빙의 `/opt/ml/model`로 그대로 이어진다는 것이 핵심입니다. **두 단계를 잇는 유일한 매개는 S3의 `model_data` 하나**입니다.

### Training Job vs Endpoint — 수명과 과금의 대조

| 축 | [Training Job](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html) | [Endpoint (Real-time)](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) |
|---|---|---|
| 성격 | 일회성 배치 계산 | 상시 HTTP 서버 |
| 수명 | 스크립트가 끝나면 **자동 종료** | 내가 `delete_endpoint`를 부를 때까지 유지 |
| 과금 | 잡 실행 시간만(데이터 복사·업로드 포함) | 인스턴스 시간당, **호출 0건이어도 계속** |
| 자동 정지 | ✅ 있음(단 warm pool을 켜면 그 시간만큼 더 과금) | ❌ 기본 구성으로는 없음 |
| 결과물 | S3의 `model.tar.gz` | HTTP 응답 |
| 실패가 드러나는 곳 | 잡 상태 + CloudWatch 학습 로그 | endpoint 상태 `Failed` + CloudWatch endpoint 로그 |
| 이 킷의 노트북 | `02_train_sft_sagemaker` | `03_deploy_endpoint` |
| 잊었을 때의 손해 | 거의 없음(이미 종료됨) | **계속 청구됨** |

!!! danger "초심자에게 가장 비싼 오해 — “학습이 끝났으니 다 끝난 것”"
    **Training Job은 스스로 멈추지만 Endpoint는 스스로 멈추지 않습니다.** 이 한 줄의 차이가 실제 청구서를 만듭니다.
    Real-time endpoint는 요청이 0건이어도 GPU 인스턴스 시간당 요금이 계속 부과되며, 노트북 커널을 닫거나 랩톱을 꺼도 **AWS 쪽 서버는 계속 떠 있습니다.**
    실습이 끝나면 반드시 해당 트랙의 **`99_cleanup.ipynb`를 실행**하세요. 삭제 순서·잔여 리소스 훑는 방법·다른 리전 확인까지 실행 절차는 [비용과 cleanup](04_sagemaker_inference.md#비용과-cleanup)에 정리돼 있습니다.

!!! tip "0 인스턴스까지 내리는 길은 따로 있습니다"
    "삭제만이 정지"는 이 킷의 구성에서 맞는 이야기입니다. 다만 endpoint를 **inference component** 기반으로 구성하고 `ManagedInstanceScaling.MinInstanceCount = 0`으로 두면 auto scaling이 인스턴스를 0까지 줄일 수 있습니다([0 인스턴스까지 스케일 인](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html)). 이때 다시 늘리려면 `NoCapacityInvocationFailures` CloudWatch 알람에 연결된 step scaling 정책이 필요하고, 0에서 올라오는 몇 분 동안의 호출은 에러가 됩니다. 이 킷의 `ModelBuilder` 배포는 inference component를 쓰지 않으므로 해당되지 않습니다.

---

## 추론 4옵션은 어디에 있나

지금까지 본 Endpoint는 정확히는 **Real-time endpoint**이고, SageMaker 추론에는 이 밖에 **Serverless · Asynchronous · Batch Transform**이 더 있습니다. AWS는 넷을 [모델 배포 옵션 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)에서 함께 문서화합니다. 넷의 비교표와 선택 기준, 그리고 이 킷이 Real-time을 고른 이유(Serverless에는 GPU가 없습니다)는 앵커 문서인 [왜 Real-time인가 — 추론 4옵션 비교](04_sagemaker_inference.md#왜-real-time인가--추론-4옵션-비교)에 있습니다.

!!! warning "Serverless의 GPU 지원 여부는 재확인 대상"
    "Serverless에는 GPU가 없다"는 것은 현시점의 정책성 항목이라 언젠가 바뀔 수 있습니다. 설계를 확정하기 전에 [Serverless Inference 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)에서 **실행 전 재확인**하세요. 같은 이유로 payload 크기·health check 시간·pending 최솟값 같은 서비스 한도도 원문 값이 기준입니다.

---

## SageMaker vs HyperPod vs EC2 vs on-prem

!!! abstract "쉽게 말하면"
    같은 GPU 학습을 하더라도 **내가 어디까지 소유하는가**가 다릅니다.
    **SageMaker AI**는 잡 단위로 계산을 빌립니다(클러스터가 없습니다).
    **HyperPod**는 오래 사는 클러스터를 빌리고 스케줄러는 내가 씁니다.
    **EC2 자체 구성**은 인스턴스를 빌려 클러스터를 내가 조립합니다.
    **on-prem**은 하드웨어까지 내가 삽니다.
    오른쪽으로 갈수록 제어권이 커지고, 왼쪽으로 갈수록 관리해야 할 것이 줄어듭니다.

### 운영 관점 대조표

| 축 | SageMaker AI (Training Job / Endpoint) | [SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html) (Slurm 또는 EKS) | EC2 자체 구성 (DLAMI · [ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html)) | on-prem |
|---|---|---|---|---|
| **컨트롤 플레인 소유** | AWS. 클러스터라는 개념 자체가 노출되지 않음 | AWS가 클러스터를 프로비저닝·복구. **Slurm이면 컨트롤러 노드가 클러스터 안**, **EKS면 EKS 컨트롤 플레인 1:1 연결**(HyperPod은 워커 노드) | 전부 내 것(head 노드·스케줄러 설정·AMI) | 전부 내 것(+ 전원·냉각·네트워크) |
| **오래 사는 클러스터** | ❌ 없음(잡마다 생성·파괴) | ✅ 있음(persistent) | ✅ 있음 | ✅ 있음 |
| **노드가 죽으면** | 해당 잡이 실패. [`RetryStrategy`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_RetryStrategy.html)로 재시도(`InternalServerError` 대상) | health monitoring agent + basic/deep health check가 감지 → **자동 재부팅·교체**(recovery `Automatic`이 기본) + **학습 잡 auto-resume**([Slurm](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency.html)은 `srun --auto-resume=1`, [EKS](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-resiliency.html)는 job auto-resume) | ParallelCluster의 [`clustermgtd`](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-v3-cluster-health-metrics.html)도 **불건전 노드를 감지해 교체합니다**. 다만 감지 깊이·재시도 정책은 내가 설계·튜닝 | 사람이 대응. 예비 부품 재고가 곧 SLA |
| **하드웨어 검증 깊이** | [`InfraCheckConfig(EnableInfraCheck=True)`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_InfraCheckConfig.html)로 인스턴스 하드웨어 + 클러스터 네트워크 점검 가능(깊이는 비공개) | **[deep health check](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-deep-health-checks.html)** — `stress-ng`, DCGM 레벨 4 진단, EFA loopback, 다중 노드 NCCL `all_reduce`. 실패 노드는 격리·교체. 단 `OnStartDeepHealthChecks` 옵트인 | EC2 status check 수준이 기본. DCGM·NCCL 검증은 내가 붙임 | 직접 구축 |
| **셋업 노력** | 가장 낮음. SDK 몇 줄 | 중간. VPC·lifecycle script·(EKS면) 클러스터 구성 | 높음. AMI·드라이버·스케줄러·공유 파일시스템 | 가장 높음. 조달 리드타임 포함 |
| **비용 모델** | **잡 실행 시간**(학습) / **인스턴스 시간**(endpoint, 삭제 전까지) | 클러스터가 떠 있는 동안 인스턴스 시간 | 인스턴스 시간(+ EBS/FSx 등) | CapEx + 전력·상면·인건비 |
| **유휴 비용** | 학습은 거의 없음(잡이 끝나면 0). endpoint는 있음 | 있음. 클러스터를 유지하는 시간이 곧 요금 | 있음(중지 시 컴퓨트는 멈추지만 스토리지는 남음) | 있음(자산은 쉬어도 감가) |
| **스케줄러 제어** | 제한적. [AWS Batch service job](https://docs.aws.amazon.com/batch/latest/userguide/service-jobs.html)으로 **큐·우선순위·fair-share는 가능**(SDK v3 `TrainingQueue`). **선점·gang scheduling은 ❌** | ✅ 강함. Slurm 파티션/우선순위 또는 Kubernetes 스케줄링 | ✅ 강함(전부 내 설정) | ✅ 강함 |
| **컨테이너 자유도** | 있음(BYOC 가능) 단 잡/endpoint 계약(경로·`/ping`)을 지켜야 함 | 있음(Slurm은 pyxis/enroot·conda, EKS는 파드) | 완전 자유 | 완전 자유 |
| **[배포 가드레일](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)(blue/green·canary·production variant)** | ✅ SageMaker AI Inference endpoint의 기능(real-time·async 대상, serverless 제외) | ❌ **SageMaker AI endpoint의 기능이며 HyperPod의 기능이 아닙니다**(EKS면 Kubernetes rolling update로 대체) | 내가 구현(ALB·Ingress 등) | 내가 구현 |
| **필요한 팀 스킬** | Python + SDK | Slurm **또는** Kubernetes/EKS + 리눅스 운영 | 리눅스·네트워킹·스케줄러·스토리지 전반 | 위 전부 + 데이터센터 운영 |
| **언제 이기나** | 잡이 간헐적이고 인프라 팀이 없을 때 | 수십~수천 GPU를 **오래** 돌리고 스케줄러 제어와 장애 복원력이 모두 필요할 때 | 특수 커널·드라이버·토폴로지 요구가 있거나 이미 EC2 표준이 있을 때 | 사용률이 지속적으로 높고 데이터를 밖으로 낼 수 없을 때 |

!!! warning "HyperPod의 리전 가용성과 기능 범위는 재확인 대상"
    표의 HyperPod 열은 개념 대조용입니다. **리전 가용성과 지원 인스턴스 타입, 그리고 어떤 기능이 Slurm/EKS 중 어느 쪽에 있는지는 계속 바뀝니다** — 클러스터를 실제로 설계하기 전에 [HyperPod 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)에서 현재 범위를 확인하세요.

### 티어를 헷갈리게 만드는 오개념

앞의 표에서 특히 자주 틀리는 지점을 따로 정리합니다.

??? question "오개념 — “HyperPod의 차별점은 노드 자동 교체다”"
    **그것만으로는 차별점이 아닙니다.** AWS ParallelCluster도 `clustermgtd`로 불건전 노드를 감지해 교체합니다(CloudWatch 대시보드에 `Unhealthy Instance Errors` 지표가 있고, static 노드에는 `node_replacement_timeout`이 있습니다).
    HyperPod의 실제 차별점은 세 가지입니다.

    1. **노드 감지·복구 컨트롤 플레인을 AWS가 소유**합니다 — cluster agent, Health Monitoring Agent, Node Recovery System. ParallelCluster에서 같은 역할을 하는 `clustermgtd`는 내 head 노드에서 도는 것이 차이입니다. 다만 **Slurm 컨트롤 플레인 자체(`slurmctld`·`slurmdbd`)는 내 인스턴스 그룹**([`SlurmConfig.NodeType: Controller`](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ref.html))이고 deep health check 대상도 아니며, 컨트롤러 HA는 내가 설계하는 아키텍처입니다.
    2. **deep health check의 깊이** — `stress-ng`·DCGM 레벨 4·EFA loopback·다중 노드 NCCL `all_reduce`까지 돌려 통과하지 못한 노드는 워크로드를 받기 전에 격리·교체합니다. 단 기본 동작이 아니라 인스턴스 그룹별 `OnStartDeepHealthChecks` **옵트인**이고 노드당 두 시간 가까이 걸려 그 동안 노드가 막히므로, AWS는 소규모 클러스터에서는 끄기를 권장합니다.
    3. **중단된 학습 잡의 [auto-resume](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html)** — 노드를 교체한 뒤 잡을 재개합니다. 정확히는 HyperPod가 `srun` 스텝을 다시 띄워 주고 **체크포인트를 읽어 오는 것은 여전히 학습 스크립트의 몫**입니다. GPU 클러스터에서 흔한 GRES가 노드에 붙어 있으면 체크포인트가 아니라 **처음부터 다시 시작**하는 예외도 문서화돼 있습니다.

    즉 "노드를 살리는 것"이 아니라 **"진행 중인 학습을 살리는 것"**이 초점입니다.

??? question "오개념 — “HyperPod로 올리면 blue/green이나 canary 배포도 되겠지”"
    **가드레일은 아닙니다.** HyperPod에도 [추론 플랫폼](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment.html)이 있어 프로덕션 트래픽을 받을 수 있습니다. 하지만 blue/green·canary·rolling 배포 가드레일과 production variant A/B는 **SageMaker AI Inference endpoint**의 기능입니다 — `CreateEndpoint`/`UpdateEndpoint`의 `EndpointConfig` 교체를 기반으로, CloudWatch 알람으로 baking 기간을 감시하며 자동 롤백까지 하는 메커니즘입니다. 적용 대상은 real-time과 asynchronous이고 serverless는 제외되며, 제외 목록에 있는 기능(inference component 등)을 쓰는 endpoint에서는 가드레일 자체를 쓸 수 없습니다.
    HyperPod EKS에서 같은 목적을 달성하려면 Kubernetes rolling update로 직접 구성해야 합니다. 기능을 찾을 때는 **"어느 서비스의 기능인가"를 먼저 확인**하세요 — 티어 간 기능 귀속 오류가 아키텍처 결정을 가장 크게 망칩니다.

??? question "오개념 — “HyperPod는 하나다”"
    **오케스트레이터가 두 종류입니다.** **Slurm** 방식은 클러스터 안에 컨트롤러·로그인·워커 노드를 두고 `sbatch`/`srun`으로 제출하며, auto-resume은 `srun --auto-resume=1` 플래그로 켭니다.
    **EKS** 방식은 EKS 컨트롤 플레인과 HyperPod 클러스터(워커 노드)를 1:1로 연결하고, 워크로드는 컨테이너/파드로 제출합니다.
    둘은 제출 방식·팀 스킬셋·관측 스택이 모두 다릅니다. 한쪽 문서를 읽고 다른 쪽 동작을 가정하지 마세요.

??? question "오개념 — “DLC는 SageMaker 전용이다”"
    **아닙니다.** DLC(Deep Learning Containers)는 **워크로드 컨테이너 이미지**라서 EC2·ECS·EKS(HyperPod-EKS 포함) 어디서나 실행됩니다.
    비교 대상으로 자주 등장하는 **DLAMI는 노드(호스트) 머신 이미지**이며 층이 다릅니다. "관리형이니까 DLC, 자체 구성이니까 DLAMI"라는 대응은 성립하지 않습니다.

---

## 언제 무엇을 쓰나

절대적인 정답은 없고, **조건에 따라** 갈립니다.

- **SageMaker AI Training Job / Endpoint를 고르세요, 만약** 파인튜닝이나 서빙이 간헐적이고, 전담 인프라 팀이 없고, "지금 잡 하나를 돌려 결과를 보는 것"이 목적인 경우. 이 킷의 모든 트랙이 이 경우에 해당합니다.
- **HyperPod를 고르세요, 만약** 다수의 GPU를 **오래** 점유하며 여러 팀이 큐를 공유해야 하고, 노드 장애로 며칠짜리 학습이 처음부터 다시 시작되는 것을 감당할 수 없는 경우. 이때 Slurm과 EKS 중에서는 팀이 이미 쓰는 스택(HPC 관행이면 Slurm, Kubernetes 표준이 있으면 EKS)을 따르는 편이 운영 비용이 낮습니다.
- **EC2 자체 구성을 고르세요, 만약** 커널·드라이버·토폴로지를 직접 통제해야 하거나(특수 빌드, 실험적 라이브러리), 이미 EC2 기반 표준·자동화 자산이 충분해 관리형 계층이 오히려 제약이 되는 경우.
- **on-prem을 고르세요, 만약** GPU 사용률이 지속적으로 높아 감가상각이 시간당 요금보다 유리하거나, 데이터를 물리적으로 외부에 낼 수 없는 규제·계약 요건이 있는 경우. 반대로 **on-prem이 지는 지점은 두 가지**입니다 — (1) **탄력성**: 이번 주에만 GPU 8장이 더 필요할 때 살 수 없습니다, (2) **차별화되지 않는 운영 작업**: 드라이버 업그레이드, 장애 부품 교체, 용량 계획에 들어가는 시간은 모델 품질에 아무것도 기여하지 않습니다.
- **혼합을 고르세요, 만약** 위 조건이 워크로드마다 다른 경우. 실제로 흔한 조합은 "정기 대규모 사전학습은 HyperPod 또는 on-prem, 간헐적 파인튜닝과 프로덕션 서빙은 SageMaker AI"입니다. HyperPod도 추론 플랫폼을 갖추고 있으니 "HyperPod는 학습만"이라고 못 박지는 마세요 — 서빙 쪽에서 갈리는 지점은 **관리형 배포 가드레일**(알람 기반 baking·자동 롤백)이고, 그것은 SageMaker AI endpoint의 기능입니다.

!!! tip "결정을 미루는 것도 선택입니다"
    처음부터 클러스터를 고민하지 마세요. **Training Job으로 시작해 잡이 실제로 얼마나 자주, 얼마나 길게 도는지 측정한 뒤** 클러스터로 옮기는 편이 거의 항상 저렴합니다.
    HyperPod가 해결하는 문제(장애로 며칠짜리 학습이 날아가는 것, 큐 경합)는 **그 규모에 도달한 뒤에** 생깁니다.

---

## 이 킷에서는

개념이 어느 노트북에 대응하는지 정리하면 다음과 같습니다(플래그십 트랙 `tracks/01_extraction_to_json/` 기준).

| 노트북 | 만드는 SageMaker 리소스 | 개념 |
|---|---|---|
| `00_setup.ipynb` | 없음(자격증명·role·버킷 확인) | 사전 준비. **role**은 SageMaker가 내 S3·ECR·CloudWatch에 접근할 때 빌려 쓰는 IAM 신분입니다 |
| `01_data_and_synthetic.ipynb` | 없음(로컬 `data/train.jsonl` 생성) | 학습 입력 준비 |
| `02_train_sft_sagemaker.ipynb` | ✅ **Training Job** (`ModelTrainer.train()`) | 데이터 S3 업로드 → 잡 제출 → `model_data` 확보 |
| `02a_train_grpo_sagemaker.ipynb` (선택) | ✅ **Training Job** (SFT→GRPO 정련) | 추출·분류 트랙만 제공 |
| `02b_local_serve.ipynb` (선택) | 없음(로컬 vLLM) | 배포 전 프리플라이트 |
| `03_deploy_endpoint.ipynb` | ✅ **Endpoint** (+ `EndpointConfig` + `Model`) | `model_data` → 상시 서버. **여기서 시간당 과금이 시작됩니다** |
| `04_evaluate.ipynb` | 없음(endpoint 호출) | held-out 평가 |
| `05_agentic_strands.ipynb` | 없음(endpoint + Bedrock 호출) | agentic loop |
| `06_agentcore_deploy.ipynb` | AgentCore Runtime(SageMaker와 별개 과금) | 프로덕션 배포 |
| `99_cleanup.ipynb` | **삭제** — Endpoint → EndpointConfig → Model | **과금 중지. 반드시 실행** |

- Training Job을 만드는 노트북은 `02`(및 선택 `02a`)이고, **끝나면 인스턴스가 자동 해제**되므로 별도 정리가 필요 없습니다. 남는 것은 S3 아티팩트와 CloudWatch 로그(용량당 과금)입니다.
- Endpoint를 만드는 노트북은 `03`이며, **삭제하는 노트북은 `99`뿐입니다.** 삭제 순서가 정해져 있고(Endpoint → EndpointConfig → Model), `ModelBuilder`가 `model-42c30d1e` 같은 임의 이름을 만들기 때문에 `endpoint_name`만으로 지우면 Model이 조용히 남습니다(실측).
- 멀티모달 트랙(`tracks/05_multimodal_extraction/`)은 노트북 세트가 짧습니다 — Training Job은 `02_train_mm_sagemaker.ipynb`, Endpoint는 `03_deploy_mm_endpoint.ipynb`, 정리는 동일하게 `99_cleanup.ipynb`입니다.
- 학습 스크립트는 `tracks/*/scripts/train.py` 하나로, 로컬 `--dry_run`과 SageMaker Training Job에서 **같은 파일**이 돕니다. 먼저 로컬에서 파이프라인을 검증하고 클라우드로 제출하는 것이 이 킷의 규율입니다([시작하기](getting_started.md)의 방식 B).

다음 단계는 [실행 런북](RUN_E2E.md)입니다. 개념을 잡았다면 그 문서의 순서를 그대로 따라가면 됩니다.

---

## 킷 내 참조 파일

이 문서의 개념이 코드로 나타나는 곳입니다(플래그십 트랙 `tracks/01_extraction_to_json/` 기준).

경로 계약과 Training Job:

- `tracks/01_extraction_to_json/scripts/train.py` — `SM_CHANNEL_TRAIN`으로 입력 채널을 찾고 `SM_MODEL_DIR`을 출력 기본값으로 씀. `save_total_limit=1`로 아티팩트 크기를 억제
- `tracks/01_extraction_to_json/02_train_sft_sagemaker.ipynb` — `ModelTrainer` + `SourceCode`/`Compute`/`InputData`/`StoppingCondition` 조립, `MAX_RUNTIME_HOURS`로 시간 한도 명시

Endpoint의 생성과 삭제:

- `tracks/01_extraction_to_json/03_deploy_endpoint.ipynb` — `model_data`를 real-time endpoint로 배포하고 invoke 스모크 테스트
- `tracks/01_extraction_to_json/99_cleanup.ipynb` — Endpoint → EndpointConfig → Model 순서로 삭제해 과금을 멈춤

공용 헬퍼:

- `common/dlc.py` — 학습·서빙 DLC 이미지 URI 해석(`DLC_IMAGE_URI` 환경변수 오버라이드)
- `common/aws_utils.py` — endpoint 호출(`invoke_sagemaker_chat`), CloudWatch 링크(`cw_links`), 변경분만 올리는 S3 업로드(`upload_if_changed`)
