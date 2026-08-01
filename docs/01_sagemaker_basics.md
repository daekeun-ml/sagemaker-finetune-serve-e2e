# 01 · SageMaker 기초 — Training Job과 Endpoint

!!! info "Scope"
    Python은 쓰고 Jupyter도 써 봤지만 **SageMaker는 처음**인 ML 엔지니어를 위한 문서입니다. AWS 인프라 지식은 필요 없습니다.
    선행 조건: 없습니다. [시작하기](getting_started.md)로 설치를 마쳤다면 [실행 런북](RUN_E2E.md)으로 넘어가기 전에 이 문서를 읽으시면 노트북이 무엇을 하고 있는지 보입니다.
    다루는 것: **Training Job**과 **Endpoint** 두 가지 개념, 실행 role, 컨테이너 경로 계약, 시간 제한, 그리고 HyperPod / EC2 / on-prem과의 차이.
    다루지 않는 것: SageMaker의 모든 기능(Studio·Pipelines·Feature Store·Clarify 등)은 의도적으로 생략합니다. 추론 옵션 상세는 [SageMaker 추론](04_sagemaker_inference.md), 학습 상세는 [파인튜닝](03_finetuning.md)이 다룹니다.

이 문서는 두 개념만 확실히 잡는 것을 목표로 합니다 — **학습은 잡, 서빙은 endpoint**.

!!! warning "빠르게 바뀌는 값"
    인스턴스 요금·리전 가용성·서비스 한도·GA 상태·Serverless의 GPU 지원 여부는 분기마다 바뀝니다.
    이 문서는 **개념과 계약**(경로·수명·과금 모델)에 집중하고 구체적인 수치는 최소한만 언급합니다. 수치를 인용할 때는 전부 **실행 직전 재확인** 대상이며, 확인처는 각 주장 옆에 붙은 공식 문서 링크입니다.

---

## TL;DR

**SageMaker AI에서 기억할 개념은 사실상 두 개입니다. Training Job은 "돌리고 나면 사라지는 계산"이고, Endpoint는 "지울 때까지 켜져 있는 서버"입니다. 이 둘의 수명과 과금 방식이 정반대라는 점만 잡으면 이 킷의 노트북 전체가 이해됩니다.**

정리하면 다음과 같습니다.

1. **Training Job은 컨테이너 이미지 + 내 스크립트 + S3 데이터 위치 + 실행 role을 넘기면 SageMaker가 머신을 띄우고 스크립트를 돌린 뒤 머신을 파괴하는 잡입니다.** 과금은 잡 단위이며 끝나면 자동으로 멈춥니다 — [Training Job](#training-job--잡이-끝나면-사라지는-계산). 넷 중 role만 값을 채우는 것으로 끝나지 않습니다(잡이 그 role을 assume해 S3·ECR에 접근하므로) — [실행 role이 매개하는 것](#실행-role이-매개하는-것--s3와-ecr).
2. **입력과 출력은 컨테이너 안의 정해진 경로로 주고받습니다.** 입력은 `SM_CHANNEL_TRAIN`(`/opt/ml/input/data/train`), 출력은 `SM_MODEL_DIR`(`/opt/ml/model`)이고, 후자에 남은 것만 `model.tar.gz`가 되어 S3로 올라갑니다 — [경로 계약](#경로-계약--컨테이너-안의-정해진-경로).
3. **`MaxRuntimeInSeconds`는 학습 코드 시간이 아니라 잡이 도는 전 구간을 덮습니다**(데이터 복사 → 이미지 pull → 학습 → 저장). 용량 대기(`Pending`)는 별도 파라미터가 덮습니다. 이 킷은 이 한도 때문에 **학습이 100% 끝난 잡을 잃은 적이 있습니다** — [MaxRuntimeInSeconds가 덮는 시간 창](#maxruntimeinseconds가-덮는-시간-창).
4. **Endpoint는 상시 HTTP 서버입니다. 호출이 0건이어도 삭제 전까지 시간당 과금됩니다.** 초심자에게 가장 비싼 오해가 바로 이 지점입니다 — [Endpoint](#endpoint--삭제할-때까지-켜져-있는-서버).
5. **SageMaker AI / HyperPod / EC2 / on-prem은 "무엇을 내가 소유하는가"로 갈립니다.** 잡 단위로 빌릴지, 클러스터를 유지할지, 인프라까지 직접 만들지의 선택입니다 — [SageMaker vs HyperPod vs EC2 vs on-prem](#sagemaker-vs-hyperpod-vs-ec2-vs-on-prem).
6. **티어 비교는 시간당 단가가 아니라 세 칸(인프라 + 운영 + 규정 준수)으로 해야 합니다.** 관리형이 인프라 단가에서 지는 것은 맞고, 뒤의 두 칸을 세지 않은 비교라는 것도 맞습니다 — [인프라 비용은 TCO의 한 칸일 뿐입니다](00_overview.md#인프라-비용은-tco의-한-칸일-뿐입니다).

---

## 기존 Pain Point

SageMaker를 처음 여는 분들이 실제로 막히는 지점은 다음과 같습니다.

- "노트북에서 `trainer.train()`을 눌렀는데 **내 코드가 어디서 도는지 모르겠습니다.**" — 로컬 커널이 아니라 SageMaker가 별도로 띄운 컨테이너에서 돕니다. 그래서 로컬 파일 경로가 통하지 않습니다.
- "학습 결과가 **어디로 갔는지** 모르겠습니다." — `/opt/ml/model`에 쓴 것만 S3로 올라갑니다. 다른 곳에 저장하면 잡이 끝날 때 인스턴스와 함께 사라집니다.
- "잡 상태가 `Failed`도 아니고 **`Stopped`인데 에러 로그가 없습니다.**" — 시간 제한(`MaxRuntimeExceeded`)에 걸린 것이며, 학습이 성공했어도 산출물이 불완전할 수 있습니다.
- "**테스트만 했는데 다음 날 청구서가 왔습니다.**" — Training Job과 Endpoint의 수명을 같다고 생각한 결과입니다. 전자는 자동으로 멈추지만 후자는 멈추지 않습니다.
- "**HyperPod가 더 좋은 거 아닌가요?**" — 더 좋은 것이 아니라 다른 층입니다. 잡 하나를 돌리려고 클러스터를 만들면 운영 부담만 늘어납니다.
- "**그냥 EC2에 vLLM 띄우는 게 더 싸고 간단한데요?**" — 시간당 단가만 보면 맞습니다. 다만 인프라 비용은 TCO의 한 칸일 뿐이고, 셋업 노력·필요한 팀 스킬·장애 대응이 함께 계산되어야 합니다 — [운영 관점 비교](#운영-관점-비교).
- "회사에 GPU 서버가 있는데 **굳이 클라우드를 쓸 이유**가 있나요?" — 사용률과 데이터 소재 요건에 따라 실제로 없을 수도 있습니다. 이 문서는 그 판단 기준을 제시합니다.

이 문서는 위 일곱 가지를 개념 층위에서 해소합니다.

---

## Training Job — 잡이 끝나면 사라지는 계산

!!! abstract "쉽게 말하면"
    Training Job은 **렌터카**에 가깝습니다. 차를 사는 것이 아니라, 목적지(스크립트)와 짐(데이터)을 주면
    SageMaker가 차를 빌려 오고, 운행이 끝나면 트렁크에 실린 것만 창고(S3)에 옮긴 뒤 차를 반납합니다.
    차는 사라지므로 **트렁크에 넣지 않은 것은 전부 버려집니다.** 요금은 운행한 시간만큼입니다.

한 장으로 줄이면 Training Job은 **화살표 세 칸**입니다 — S3에 있는 입력, 잡이 도는 동안만 존재하는 계산 클러스터, 그리고 다시 S3로 나가는 출력.

[![Training Job 해부도. 바깥에서 "Invoke API (예: CreateTrainingJob())" 화살표가 아래로 내려와 잡 하나를 나타내는 큰 박스로 들어가고, 박스 안에서는 왼쪽의 Job Inputs(예: S3 버킷 내 훈련 데이터)가 가운데의 Ephemeral Compute Cluster로, 거기서 다시 오른쪽의 Job Outputs(예: S3 버킷 내 모델 아티팩트)로 화살표를 따라 흐른다. 가운데 클러스터에는 "Job code and runtime as Docker container(s)"라는 설명이 붙어 있다](images/sm_job_anatomy.png)](images/sm_job_anatomy.png)

*가운데 칸의 이름이 그대로 결론입니다 — Ephemeral, 즉 잡과 함께 생겼다가 잡과 함께 사라지는 계산입니다.*

이 그림에서 정작 중요한 것은 **박스 바깥에서 들어오는 화살표**입니다. 가운데 클러스터는 내가 미리 만들어 두는 리소스가 아니라 `CreateTrainingJob` 호출이 만들어 내는 것이고, 그 호출이 만든 잡이 끝나면 함께 사라집니다. 이 킷에서 그 호출을 실제로 하는 코드는 `02_train_sft_sagemaker.ipynb`의 `trainer.train(...)` 한 줄입니다 — `ModelTrainer`(SageMaker Python SDK v3)는 이 API **위에 얹힌 래퍼**이고, 그래서 노트북 어디에도 "클러스터를 만드는 셀"이 없습니다(래퍼와 raw API의 계층 관계는 [JumpStart vs 자체 train.py](03_finetuning.md#jumpstart-vs-자체-trainpy)에 그림으로 있습니다). 클러스터의 규모도 잡을 정의할 때 함께 정해집니다(`Compute(instance_type=..., instance_count=1)` — 그림의 클러스터는 여러 노드를 담을 수 있지만 이 킷은 단일 인스턴스입니다).

양쪽 끝이 모두 **S3**라는 점이 임시 클러스터를 마음 놓고 쓸 수 있게 만듭니다. 입력은 S3에서 컨테이너로 복사되고(이 킷은 `train` 채널 하나), 출력은 컨테이너에서 다시 S3로 올라갑니다. 즉 잡보다 오래 사는 것은 그림 양 끝의 두 S3 위치뿐이고, 가운데 칸에만 남긴 것은 회수할 방법이 없습니다 — 아래의 [경로 계약](#경로-계약--컨테이너-안의-정해진-경로)이 "컨테이너의 어느 경로가 오른쪽 S3로 올라가는가"를 정하는 규칙입니다. 뒤에서 다룰 Endpoint와의 대조도 여기서 나옵니다. Endpoint에는 이 그림의 가운데 칸에 해당하는 것이 **삭제 전까지 상시로** 떠 있습니다.

AWS는 이 방식을 [관리형 학습(how it works)](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)으로 문서화합니다. 그림 왼쪽 칸으로 들어가는 준비물, 즉 내가 SageMaker에 넘기는 것은 네 가지입니다.

| 넘기는 것 | 무엇 | 이 킷에서는 |
|---|---|---|
| **컨테이너 이미지** | 학습 환경(파이썬·CUDA·프레임워크)이 들어 있는 Docker 이미지 | AWS가 게시한 PyTorch **DLC**(Deep Learning Containers — AWS가 미리 빌드해 ECR에 올려 둔 학습/추론용 컨테이너 이미지, `.env`의 `DLC_IMAGE_URI`) |
| **내 코드** | 진입 스크립트와 `requirements.txt`가 든 디렉터리 | `SourceCode(source_dir='scripts', entry_script='train.py')` |
| **데이터 위치** | S3 URI. **채널**(= 컨테이너 안에서 데이터가 마운트될 이름) 이름과 함께 지정 | `InputData(channel_name='train', data_source=train_s3)` |
| **실행 role** | 잡이 내 대신 S3·ECR·CloudWatch에 접근할 때 쓰는 IAM 신분 | `role=config.resolve_sagemaker_role(sess)`(`SAGEMAKER_ROLE_ARN`). 넘기는 것 중 유일하게 값을 채우는 것만으로 끝나지 않습니다 — 아래 [실행 role이 매개하는 것](#실행-role이-매개하는-것--s3와-ecr) |

그러면 SageMaker가 순서대로 다음을 합니다.

```
1. 인스턴스 프로비저닝        (상태: Pending — GPU 용량을 기다리는 구간)
2. 인스턴스 준비              (상태: Starting)
3. S3 데이터를 컨테이너로 복사  (상태: Downloading) → /opt/ml/input/data/train
4. 이미지 pull → entry_script 실행 (상태: Training — 여기서부터 CloudWatch 로그가 생김)
5. /opt/ml/model 을 tar.gz로 압축해 S3 업로드   (상태: Uploading) ← 이것이 model_data
6. 인스턴스 종료 + 과금 중지    (상태: Completed / Failed / Stopped)
```

[`DescribeTrainingJob`의 secondary status 정의](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html)를 보면 `Downloading`은 이미지가 아니라 **데이터**를 받는 구간입니다(`File` 입력 모드에서 S3 → ML 스토리지 볼륨). **이미지 다운로드에는 전용 상태가 없습니다** — 과거의 `DownloadingTrainingImage`는 [secondary status transition 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_SecondaryStatusTransition.html)에 "no longer supported"로 명시돼 있고, 지금은 `StatusMessage` 문구로만 드러납니다. 그리고 그 문서의 예시는 `"Downloading the training image"`를 **`SecondaryStatus = Training`**과 짝지어 보여 줍니다(`Starting` 구간의 예시 문구는 `"Starting the training job"`, `"Launching requested ML instances"`, `"Preparing the instances for training"`입니다). 위 순서도의 단계 구분은 이 예시를 따른 것이며, **AWS는 "status message는 바뀔 수 있으니 코드에 넣지 말라"고 명시**하므로 문구 자체로 분기하지 마세요. 5번 `Uploading`이 이 킷이 실제로 다친 구간입니다 — 뒤의 [시간 제한](#maxruntimeinseconds가-덮는-시간-창)을 보세요.

여기서 초심자가 놓치기 쉬운 두 가지가 있습니다.

- **내 코드는 로컬 커널이 아니라 격리된 컨테이너에서 돕니다.** 그래서 노트북 옆의 파일을 열 수 없고, 데이터를 미리 S3에 올려야 합니다(이 킷의 `02_train_sft_sagemaker`가 `upload_if_changed()`로 처리합니다). 같은 이유로 `train.py`는 `common/`을 import하지 않는 **self-contained** 파일이어야 합니다 — SageMaker가 컨테이너에 올리는 것은 `source_dir` 하나뿐입니다([train.py 상세](03_finetuning.md#trainpy--로컬-dry-run과-sagemaker-학습-잡)).
- **과금 대상 시간은 학습 시간보다 넓습니다.** AWS 문서 기준으로 **데이터 다운로드 시간**과 **모델 아티팩트 압축·업로드 시간**도 billable time에 포함됩니다. 즉 "학습 5분"이 "요금 5분"은 아닙니다.

### 실행 role이 매개하는 것 — S3와 ECR

위 표의 네 번째 항목만 성격이 다릅니다. `SAGEMAKER_ROLE_ARN`은 **내가 쓰는 자격증명이 아니라 SageMaker에게 빌려주는 신분**이고, 잡이 도는 동안 S3와 ECR에 실제로 손을 뻗는 주체는 내 IAM user가 아니라 그 role입니다.

[![SageMaker 학습 잡의 IAM role 매개 구조 다이어그램. 왼쪽 위에는 ModelTrainer 스니펫에서 role_arn='[Your SageMaker-compatible IAM role]' 한 인자가 강조되어 있고, 거기서 아래로 점선이 내려가 IAM role 박스와 그에 붙은 Permissions 목록(허용은 체크, 거부는 x 표시)으로 이어진다. 가운데 Amazon SageMaker AI 박스 안에는 Cluster → Instance 1 → Processing container가 중첩되어 있고, 클러스터에서 IAM role 박스로 "Assume role"이라고 적힌 점선 화살표가 나간다. 오른쪽에는 컨테이너가 권한을 통과해 접근하는 대상이 나열된다 — 체크된 s3:GetObject는 s3://bucket/path/to/training/data와 s3://bucket/path/to/test/data로, 체크된 s3:PutObject는 s3://bucket/path/to/model로 이어진다. 오른쪽 위에는 PyTorch Container Image를 담은 Amazon ECR 박스가 있다](images/sm_security.png)](images/sm_security.png)

*체크 표시가 붙은 두 줄이 요점입니다 — 컨테이너가 S3를 읽고 쓸 수 있는 이유는 내 자격증명이 아니라 role의 정책에 그 action이 허용되어 있기 때문입니다.*

그림에서 가장 중요한 화살표는 클러스터에서 IAM 박스로 거꾸로 올라가는 **Assume role**입니다. 잡을 만들 때 role ARN 하나를 넘기면, SageMaker가 세운 클러스터가 그 role을 **대신 맡아(assume)** 컨테이너에 임시 자격증명을 심습니다. 그래서 컨테이너 안의 `train.py`는 자격증명 코드를 한 줄도 갖지 않는데도 S3에서 데이터를 받고 아티팩트를 올릴 수 있습니다. 이 킷에서 그 한 인자는 `02_train_sft_sagemaker`의 `ModelTrainer(..., role=role, ...)`이고(SDK v3의 인자 이름은 그림의 `role_arn`이 아니라 `role`입니다), 값은 `config.resolve_sagemaker_role(sess)`가 해석합니다. 같은 role이 배포에서 한 번 더 쓰입니다 — endpoint도 아티팩트를 읽고 서빙 이미지를 pull할 때 이 role을 assume합니다(아래 [배포 3단계](#배포-3단계--무엇을-어떤-순서로-넘기는가)의 IAM 역할 칸).

**이것이 권한 부족이 제출 시점이 아니라 잡 중간에 드러나는 이유입니다.** `CreateTrainingJob`은 role ARN의 형식과 내 `iam:PassRole`만 확인하고 잡을 받아들입니다 — 그림 오른쪽의 `s3:GetObject`·`s3:PutObject`·ECR pull은 클러스터가 뜬 **뒤에야** 시도되기 때문입니다. 그래서 권한이 모자라도 잡 접수는 정상으로 보이다가, 용량 대기(`Pending`)와 인스턴스 프로비저닝 요금을 다 치른 뒤 `Downloading`/`Training` 구간에서 `Failed`로 떨어집니다. 원인을 볼 수 있는 곳은 노트북 출력이 아니라 CloudWatch 로그입니다(런북의 [자주 막히는 곳](RUN_E2E.md#e2e-흐름에서-자주-막히는-곳) "학습 잡이 시작 직후 실패" 행이 이 증상입니다).

그림의 Permissions 목록에 x 표시가 함께 있는 것도 그대로 읽어야 합니다 — role이 **있다**는 것과 role에 **필요한 권한이 붙어 있다**는 것은 다릅니다. `resolve_sagemaker_role()`은 env → `get_execution_role()` → IAM 자동 탐지 → (opt-in) `AmazonSageMaker-DefaultRole` 생성 순으로 role을 찾는데, 3단계의 자동 탐지가 보는 것은 **`sagemaker.amazonaws.com`을 신뢰하는지**(즉 assume이 가능한지)뿐입니다. S3·ECR 권한이 실제로 붙어 있는지는 확인하지 않으므로, 자동으로 잡힌 role이라도 첫 완주 전에 정책을 한 번 열어 보세요. 반대로 4단계는 `AmazonSageMakerFullAccess`를 붙인 role을 만들기 때문에 기본이 opt-in(`SAGEMAKER_CREATE_DEFAULT_ROLE=1`)입니다 — 편의와 최소 권한이 정확히 반대 방향인 지점입니다.

그림의 일반형과 이 킷의 값이 갈리는 곳은 두 군데입니다. 그림은 채널이 둘(training/test)이지만 이 킷은 `train` 채널 하나만 쓰므로 `s3:GetObject`가 필요한 prefix도 하나이고, `s3:PutObject` 대상인 `s3://bucket/path/to/model`은 이 킷에서 `S3_BUCKET`(비우면 `sess.default_bucket()`) 아래의 SDK 기본 출력 경로입니다. 가운데 박스의 라벨이 `Processing container`인 것도 그림이 잡 종류를 가리지 않는 일반형이기 때문이며, 학습 잡에서 그 자리에 뜨는 것은 training container입니다.

### 경로 계약 — 컨테이너 안의 정해진 경로

**이 킷의 모든 노트북이 이 계약 위에 서 있습니다.** SageMaker는 컨테이너 안의 정해진 경로를 통해서만 데이터를 주고받고, 각 경로에 대응하는 환경변수를 심어 줍니다. 경로별 역할은 [학습 스토리지 경로 매핑](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html) 문서에 정의돼 있습니다.

| 컨테이너 경로 | 환경변수 | 용도 | 잡이 끝날 때 |
|---|---|---|---|
| `/opt/ml/input/data/<채널명>` | `SM_CHANNEL_<채널명>` | 입력 데이터(S3에서 복사됨) | — (읽기용) |
| `/opt/ml/input/data/code` | `SM_SOURCE_DIR` (+ `SM_ENTRY_SCRIPT`) † | 업로드된 내 코드 | — (읽기용) |
| `/opt/ml/model` | `SM_MODEL_DIR` | **최종 모델 아티팩트** | ✅ 업로드 — `tar.gz`로 압축 후 S3로 |
| `/opt/ml/output/data` | `SM_OUTPUT_DATA_DIR` † | loss·중간 산출물 등 부가 출력 | ✅ 업로드 — 별도 `tar.gz`로 |
| [`/opt/ml/output/failure`](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html) | `SM_OUTPUT_FAILURE` † | 실패 이유를 적는 파일 | 앞부분이 잡의 `FailureReason`이 됨 |
| [`/opt/ml/checkpoints`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) | (없음) | 체크포인트 | `CheckpointConfig(S3Uri=...)`를 **지정한 경우에만** 학습 중 S3와 동기화(압축 안 함) |
| `/tmp` | (없음) | 임시 작업 공간 | ❌ 업로드되지 않음 — 인스턴스와 함께 삭제 |

† 표시한 네 변수는 **AWS DG의 [환경변수 요약 표](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html)에 없거나 다르게 적혀 있습니다.** 그 표는 `SM_OUTPUT_DATA_DIR`·`SM_OUTPUT_FAILURE`·`SM_SOURCE_DIR`·`SM_ENTRY_SCRIPT`를 아예 나열하지 않고 `/opt/ml/output/data`를 `SM_OUTPUT_DIR`로 매핑하는데, 실제로 컨테이너에 값을 심는 SDK 소스(`sagemaker/train/container_drivers/scripts/environment.py`, sagemaker-train 1.16.0)는 위 표대로 설정합니다. 위 네 행의 근거는 DG 표가 아니라 **이 SDK 소스**([aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk))이며, DG 표는 오래된/단순화된 버전으로 보입니다.

두 가지를 짚어 둡니다. 첫째, `SM_OUTPUT_DIR`은 `/opt/ml/output/data`가 아니라 **그 부모인 `/opt/ml/output`**입니다(그 아래에 `data`와 `failure`가 있습니다 — 위 SDK 소스 기준). `os.environ['SM_OUTPUT_DIR']`에 파일을 쓰면 `output.tar.gz`에 들어가지 않으니, 부가 출력은 `SM_OUTPUT_DATA_DIR`에 쓰세요. SDK 세대에 따라 값이 달라질 수 있으므로 잡 로그에서 실제 값을 한 번 확인하는 편이 안전합니다. 둘째, 코드가 내려오는 경로는 SDK 세대에 따라 다릅니다 — 이 킷이 쓰는 [`ModelTrainer`](https://sagemaker.readthedocs.io/en/stable/)(SageMaker Python SDK v3)는 `source_dir`을 `code`라는 **입력 채널로 올려** `/opt/ml/input/data/code`에 마운트하고 거기서 `cd` 후 실행합니다. 그래서 `train.py` 안의 상대 경로는 그 디렉터리를 기준으로 풀립니다. 레거시 Estimator·추론 컨테이너에서 보이는 `/opt/ml/code` + `SAGEMAKER_SUBMIT_DIRECTORY`는 다른 계약입니다.

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
    반대로 `/opt/ml/model`에 **필요 없는 것을 두면 손해**입니다. 이 디렉터리 전체가 압축되므로 중간 체크포인트가 쌓이면 업로드 시간이 늘고, 그 시간은 요금과 [시간 제한](#maxruntimeinseconds가-덮는-시간-창)을 함께 잡아먹습니다. 이 킷이 `SFTConfig(save_total_limit=1)`을 쓰는 이유입니다(실측 2026-07-31: 체크포인트 3개 = 0.7GB, 전부 서빙에 불필요).

이 경로는 학습에서 끝나지 않고 **서빙까지 이어집니다.** `/opt/ml/model`은 배포 시 추론 컨테이너가 모델을 읽는 경로이기도 합니다. 그래서 이 킷의 `train.py`는 머지된 모델을 하위 폴더가 아니라 **아티팩트 루트**에 저장합니다 — 루트에 완전한 HF 모델(`config.json` + 가중치)이 없으면 vLLM이 엔진을 감지하지 못합니다([merge_adapter 상세](03_finetuning.md#merge_adapter--서빙-단순화)).

### MaxRuntimeInSeconds가 덮는 시간 창

`StoppingCondition(max_runtime_in_seconds=...)`은 **폭주 방지 상한**입니다. [StoppingCondition API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에 따르면 잡이 이 시간을 넘기면 SageMaker가 `SIGTERM`을 보내고 **120초를 준 뒤** 잡을 종료합니다.

중요한 것은 **이 시계가 학습 코드만 재지 않는다**는 점입니다. 문서로 확인되는 사실은 세 가지입니다 — (1) `MaxRuntimeInSeconds`는 "잡이 중지되기까지 실행될 수 있는 최대 시간"이고, (2) 그 앞의 용량 대기(`Pending`)는 이 한도가 아니라 별도 파라미터인 `MaxPendingTimeInSeconds`가 덮으며, (3) [과금 시간](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html)에는 데이터 다운로드와 아티팩트 압축·업로드가 포함됩니다.

```
   Pending          |<-------------- MaxRuntimeInSeconds -------------->|
 (용량 대기 —        데이터 복사 → 이미지 pull → 학습 루프 → 머지/저장 → (아티팩트 업로드)
  MaxPendingTime)                                       ↑
                              여기서 잘리면 "학습은 성공했는데 배포 불가"
```

**아티팩트 업로드가 이 시간 창의 안인지 밖인지는 문서가 갈립니다.** 같은 API 문서가 "메트릭 발행과 **중지된 뒤의** 모델 아카이브·업로드까지 포함한 총 실행 시간 상한은 30일"이라고 적고 있어, 업로드를 `MaxRuntime` 컷오프 **밖**의 시간으로 기술합니다. AWS는 `TrainingStartTime`을 시계의 기점으로 명시하지도 않습니다. 다만 실무에 필요한 결론은 문서 해석과 무관합니다 — **학습 루프가 끝난 뒤의 후처리(머지·저장)는 확실히 이 창 안**이고, 이 킷은 바로 거기서 잘렸습니다(아래 실측). 한도는 학습 시간이 아니라 **후처리까지 포함해** 잡으세요.

이 킷은 실제로 이 함정에 빠졌습니다. `stopping_condition`을 생략하면 SDK가 1시간을 자동으로 넣는데(SDK 3.16.0 실측 2026-07-31), 189/189 step을 전부 마친 잡이 **LoRA 머지 도중** 강제 종료되어 아티팩트에 어댑터와 체크포인트만 남았습니다. 상태는 `Failed`가 아니라 `Stopped`이고 `FailureReason`은 비어 있어서, CloudWatch 로그만 보면 정상 종료처럼 보입니다. 실측 타임라인과 대응은 [MaxRuntimeExceeded 함정](03_finetuning.md#maxruntimeexceeded--학습-뒤-머지에서-잘리는-함정)에 정리돼 있습니다.

??? tip "한도는 넉넉히, 규모는 작게"
    한도를 크게 잡아도 **추가 요금이 없습니다.** 잡이 정상 종료되면 그 시점에 과금이 멈추기 때문입니다. 비용을 줄이고 싶다면 한도가 아니라 **데이터 건수와 epoch**를 줄이세요.
    이 킷의 노트북은 `MAX_RUNTIME_HOURS = 4`를 명시하고, 제출 전에 예상 시간을 계산해 한도를 넘으면 `assert`로 막습니다.

한도 외에도 잡 레벨에서 켤 수 있는 비용·복원력 옵션이 몇 가지 더 있습니다.

??? tip "함께 알아 두면 좋은 잡 레벨 옵션 (참고용)"
    이 킷이 기본으로 쓰지는 않지만, Training Job에는 문서화된 비용·복원력 옵션이 있습니다. 값과 동작은 **실행 전 재확인**하세요.

    - **[`MaxPendingTimeInSeconds`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)** — 용량을 기다리는 `Pending` 상태의 상한입니다. `MaxRuntimeInSeconds`와 **별개 파라미터**이며, API 유효 범위의 최솟값이 7,200초입니다. GPU 용량 대기로 잡이 무한정 걸려 있는 것을 막습니다.
    - **[Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)** — `EnableManagedSpotTraining=True` + `MaxWaitTimeInSeconds`(≥ `MaxRuntimeInSeconds`)로 켭니다. 절감률은 AWS 문서끼리도 수치가 갈리므로(2026-08 확인: DG는 "최대 90%", `CreateTrainingJob` API는 "최대 80%") 절대값으로 약속하지 말고 **잡이 끝난 뒤 실측**하세요 — `(1 - BillableTimeInSeconds / TrainingTimeInSeconds) * 100`이 그 잡의 실제 절감률입니다. Spot은 중단될 수 있으므로 체크포인트를 남기는 것이 권장 구성인데, `/opt/ml/checkpoints`에 쓰는 것만으로는 부족하고 **`CheckpointConfig(S3Uri=...)`를 함께 지정**해야 S3로 동기화됩니다(지정하지 않으면 그냥 로컬 디스크라 인스턴스와 함께 사라집니다). 참고로 Spot과 warm pool은 함께 쓸 수 없습니다.
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

### 배포 3단계 — 무엇을 어떤 순서로 넘기는가

리소스 3층은 "무엇이 만들어지는가"의 그림이고, 실제로 내가 하는 일은 **준비물을 세 단계에 걸쳐 넘기는 것**입니다. 1단계에서 넘길 것을 전부 확보하고, 2단계에서 그것을 어떤 인스턴스·어떤 서빙 스택으로 띄울지 정하고, 3단계부터 클라이언트가 HTTP로 호출합니다.

[![endpoint 배포 3단계 흐름도. 1단계 모델 준비 완료는 모델 아티팩트, 컨테이너 이미지(AWS에서 사전 구축한 컨테이너 또는 직접 준비), IAM 역할 세 가지를 담고 있고, 이것이 입력으로 2단계 엔드포인트 구성에 들어간다. 2단계는 1단계의 매개변수, 배포 모드, 컴퓨팅·GPU 선택, 모델 서빙 스택, 자동 확장 정책으로 이루어진다. 3단계 실시간 HTTP 요청에서는 클라이언트 애플리케이션이 SageMaker Endpoint로 추론 요청을 보내고 추론 결과를 돌려받는다](images/sm_endpoint_01.png)](images/sm_endpoint_01.png)

*1단계의 산출물이 그대로 2단계의 입력이 되고, 3단계에 도달하면 이 킷의 구성(`ModelBuilder` + 평범한 production variant)에서는 삭제 전까지 그 상태로 남습니다.*

각 칸이 이 킷에서 실제로 무엇인지 대응시키면 다음과 같습니다(`03_deploy_endpoint.ipynb` 기준).

| 단계 | 넘기는 것 | 이 킷에서는 |
|---|---|---|
| **1. 모델 준비 완료** | 모델 아티팩트 | Training Job이 S3에 올린 `model.tar.gz` = `model_data`. **학습과 서빙을 잇는 유일한 매개** |
| | 컨테이너 이미지 | AWS가 사전 구축한 서빙 DLC(`.env`의 `VLLM_IMAGE_URI` / `SGLANG_IMAGE_URI` / `LMI_IMAGE_URI`). 직접 준비(BYOC)도 되지만 아래 컨테이너 계약을 내가 구현해야 합니다 |
| | IAM 역할 | `role_arn`. endpoint가 **S3 아티팩트를 읽고 ECR 이미지를 pull하고 CloudWatch에 로그를 쓸 때** assume하는 신분이며, 학습 잡에 넘긴 것과 같은 role입니다([실행 role이 매개하는 것](#실행-role이-매개하는-것--s3와-ecr)) |
| **2. 엔드포인트 구성** | 1단계의 매개변수 | 위 셋이 그대로 `Model`이 됩니다 |
| | 배포 모드 | 추론 4옵션 중 **Real-time**(Serverless에 GPU가 없어서). SDK v3에는 이와 별개로 실행 위치를 고르는 `Mode` 축(`IN_PROCESS` · `LOCAL_CONTAINER` · `SAGEMAKER_ENDPOINT`)이 있습니다 |
| | 컴퓨팅·GPU 선택 | `.env`의 `INFER_INSTANCE_TYPE=ml.g6.2xlarge`(L4 24GB) × `initial_instance_count=1` |
| | 모델 서빙 스택 | `.env`의 `SERVING_ENGINE` — `vllm`(기본) · `sglang` · `lmi` |
| | 자동 확장 정책 | 걸지 않습니다(1대 고정). 시점이 그림과 다르니 아래 주의를 보세요 |
| **3. 실시간 HTTP 요청** | 클라이언트 ↔ Endpoint | `invoke_endpoint`로 요청·응답. 받는 쪽은 컨테이너의 8080 `/invocations`입니다 |

!!! warning "자동 확장 정책은 endpoint 구성에 함께 들어가지 않습니다"
    위 그림은 자동 확장 정책을 2단계(엔드포인트 구성) 안에 그렸지만, 평범한 production variant에서는 **순서가 다릅니다.** AWS는 auto scaling을 쓰려면 "**이미 endpoint를 만들어 두었어야 한다**"고 명시하고([auto scaling 사전 조건](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-prerequisites.html)), 정책은 `EndpointConfig`의 필드가 아니라 **Application Auto Scaling API**(scalable target 등록 → 정책 정의 → 적용)로 endpoint가 뜬 **뒤에** 붙입니다. `EndpointConfig` 안에 스케일링 범위가 들어가는 길은 하나 있습니다 — production variant의 [`ManagedInstanceScaling`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html)(`MinInstanceCount`/`MaxInstanceCount`)입니다. 다만 그 **최솟값을 0으로** 내리려면 inference component 기반 구성이 필요하고, 그 이야기는 이 문서 뒤쪽의 「0 인스턴스까지 내리는 길은 따로 있습니다」가 다룹니다. 그림의 그 칸은 "구성 시점에 함께 넣는 값"이 아니라 "결국 챙겨야 하는 것"의 목록으로 읽으세요.

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

"삭제가 유일한 정지 수단인가"에는 예외가 하나 있습니다.

??? tip "0 인스턴스까지 내리는 길은 따로 있습니다"
    "삭제만이 정지"는 이 킷의 구성에서 맞는 이야기입니다. 다만 endpoint를 **inference component** 기반으로 구성하고 `ManagedInstanceScaling.MinInstanceCount = 0`으로 두면 auto scaling이 인스턴스를 0까지 줄일 수 있습니다([0 인스턴스까지 스케일 인](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html)). 이때 다시 늘리려면 `NoCapacityInvocationFailures` CloudWatch 알람에 연결된 step scaling 정책이 필요하고, 0에서 올라오는 몇 분 동안의 호출은 에러가 됩니다. 이 킷의 `ModelBuilder` 배포는 inference component를 쓰지 않으므로 해당되지 않습니다.

---

## 추론 4옵션은 어디에 있나

지금까지 본 Endpoint는 정확히는 **Real-time endpoint**이고, SageMaker 추론에는 이 밖에 **Serverless · Asynchronous · Batch Transform**이 더 있습니다. [모델 배포 옵션 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)의 "endpoint에 모델 배포" 목록에는 **endpoint 3종(Real-time · Serverless · Asynchronous)만** 있고, **Batch Transform은 별도 문서**로 다뤄집니다 — endpoint를 만들지 않는 옵션이라 계열이 다릅니다. 넷의 비교표와 선택 기준, 그리고 이 킷이 Real-time을 고른 이유(Serverless에는 GPU가 없습니다)는 앵커 문서인 [왜 Real-time인가 — 추론 4옵션 비교](04_sagemaker_inference.md#왜-real-time인가--추론-4옵션-비교)에 있습니다.

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

### 운영 관점 비교

| 축 | SageMaker AI (Training Job / Endpoint) | [SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html) (Slurm 또는 EKS) | EC2 자체 구성 (DLAMI · [ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html)) | on-prem |
|---|---|---|---|---|
| **컨트롤 플레인 소유** | AWS. 클러스터라는 개념 자체가 노출되지 않음 | AWS가 클러스터를 프로비저닝·복구. **Slurm이면 컨트롤러 노드가 클러스터 안**, **EKS면 EKS 컨트롤 플레인 1:1 연결**(HyperPod은 워커 노드) | 전부 내 것(head 노드·스케줄러 설정·AMI) | 전부 내 것(+ 전원·냉각·네트워크) |
| **오래 사는 클러스터** | ❌ 없음(잡마다 생성·파괴) | ✅ 있음(persistent) | ✅ 있음 | ✅ 있음 |
| **노드가 죽으면** | 실행 중이면 해당 잡이 실패 → [`RetryStrategy`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_RetryStrategy.html)로 재시도(`InternalServerError` 대상). 단 **기동 시점의 불건전 노드는 SageMaker가 조용히 교체**합니다([`Starting` status message](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_SecondaryStatusTransition.html)에 `"Launched instance was unhealthy, replacing it!"`, `"Insufficient capacity error from EC2 while launching instances, retrying!"`) | health monitoring agent + basic/deep health check가 감지 → **자동 재부팅·교체**(recovery `Automatic`이 기본) + **학습 잡 auto-resume**([Slurm](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency.html)은 `srun --auto-resume=1`, [EKS](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-resiliency.html)는 job auto-resume) | ParallelCluster의 [`clustermgtd`](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-v3-cluster-health-metrics.html)도 **불건전 노드를 감지해 교체합니다**. 다만 감지 깊이·재시도 정책은 내가 설계·튜닝 | 사람이 대응. 예비 부품 재고가 곧 SLA |
| **하드웨어 검증 깊이** | [`InfraCheckConfig(EnableInfraCheck=True)`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_InfraCheckConfig.html)로 인스턴스 하드웨어 + 클러스터 네트워크 점검 가능(깊이는 비공개) | **[deep health check](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-deep-health-checks.html)** — `stress-ng`, DCGM 레벨 4 진단, EFA loopback, 다중 노드 NCCL `all_reduce`. 실패 노드는 격리·교체. 단 `OnStartDeepHealthChecks` 옵트인 | EC2 status check 수준이 기본. DCGM·NCCL 검증은 내가 붙임 | 직접 구축 |
| **셋업 노력** | 가장 낮음. SDK 몇 줄 | 중간. VPC·lifecycle script·(EKS면) 클러스터 구성 | 높음. AMI·드라이버·스케줄러·공유 파일시스템 | 가장 높음. 조달 리드타임 포함 |
| **비용 모델** | **잡 실행 시간**(학습) / **인스턴스 시간**(endpoint, 삭제 전까지) | 클러스터가 떠 있는 동안 인스턴스 시간 | 인스턴스 시간(+ EBS/FSx 등) | CapEx + 전력·상면·인건비 |
| **유휴 비용** | 학습은 거의 없음(잡이 끝나면 0). endpoint는 있음 | 있음. 클러스터를 유지하는 시간이 곧 요금 | 있음(중지 시 컴퓨트는 멈추지만 스토리지는 남음) | 있음(자산은 쉬어도 감가) |
| **스케줄러 제어** | 제한적. [AWS Batch service job](https://docs.aws.amazon.com/batch/latest/userguide/service-jobs.html)으로 **큐·우선순위·fair-share는 가능**(SDK v3 `TrainingQueue`의 `share_identifier`, 잡 단위 [`schedulingPriority`는 0~9999](https://docs.aws.amazon.com/batch/latest/APIReference/API_SubmitServiceJob.html)). 선점·gang scheduling은 문서화된 기능이 **없습니다**(부재를 명시한 문서도 없어 추정) | ✅ 강함. Slurm 파티션/우선순위 또는 Kubernetes 스케줄링 | ✅ 강함(전부 내 설정) | ✅ 강함 |
| **컨테이너 자유도** | 있음(BYOC 가능) 단 잡/endpoint 계약(경로·`/ping`)을 지켜야 함 | 있음(Slurm은 pyxis/enroot·conda, EKS는 파드) | 완전 자유 | 완전 자유 |
| **보안 패치·규정 준수** | 호스트와 관리형 런타임은 AWS. 내 몫은 **컨테이너 이미지 태그를 최신으로 올리는 것**뿐(이 킷은 `.env`에 고정) | 호스트 OS·드라이버는 내 것. 플랫폼 소프트웨어 패치는 [`UpdateClusterSoftware`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_UpdateClusterSoftware.html)로 **내가 호출**하고, 그 사이 클러스터 가동률이 영향을 받습니다 | guest OS·드라이버·서빙 프로세스 패치가 전부 내 몫([공동 책임 모델](https://aws.amazon.com/compliance/shared-responsibility-model/)) | 위 전부 + 물리 보안·감사 대응 |
| **[배포 가드레일](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)(blue/green·canary·production variant)** | ✅ 지원 — SageMaker AI Inference endpoint의 기능(real-time·async 대상, serverless 제외) | ❌ 미지원 — **SageMaker AI endpoint의 기능이며 HyperPod의 기능이 아닙니다**(EKS면 Kubernetes rolling update로 대체) | 내가 구현(ALB·Ingress 등) | 내가 구현 |
| **필요한 팀 스킬** | Python + SDK | Slurm **또는** Kubernetes/EKS + 리눅스 운영 | 리눅스·네트워킹·스케줄러·스토리지 전반 | 위 전부 + 데이터센터 운영 |
| **언제 이기나** | 잡이 간헐적이고 인프라 팀이 없을 때 | 수십~수천 GPU를 **오래** 돌리고 스케줄러 제어와 장애 복원력이 모두 필요할 때 | 특수 커널·드라이버·토폴로지 요구가 있거나 이미 EC2 표준이 있을 때 | 사용률이 지속적으로 높고 데이터를 밖으로 낼 수 없을 때 |

표의 HyperPod 열은 개념 대조용입니다. **리전 가용성과 지원 인스턴스 타입, 그리고 어떤 기능이 Slurm/EKS 중 어느 쪽에 있는지는 계속 바뀌므로 실행 전 재확인 대상입니다** — 클러스터를 실제로 설계하기 전에 [HyperPod 개요](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)에서 현재 범위를 확인하세요.

이 표는 십여 개 축을 나란히 놓았지만, 실제로 티어를 고를 때 사람들이 보는 축은 **시간당 단가** 하나뿐입니다. 그 비교에서 관리형이 지는 것은 사실이고, 동시에 그것이 총 소유비용의 한 칸일 뿐이라는 것도 사실입니다 — 나머지 두 칸(운영·규정 준수)을 세는 방법은 [인프라 비용은 TCO의 한 칸일 뿐입니다](00_overview.md#인프라-비용은-tco의-한-칸일-뿐입니다)에 그림과 표로 있습니다.

### 티어를 헷갈리게 만드는 오개념

앞의 표에서 특히 자주 틀리는 지점을 따로 정리합니다.

??? question "오개념 — “HyperPod의 차별점은 노드 자동 교체다”"
    **그것만으로는 차별점이 아닙니다.** AWS ParallelCluster도 `clustermgtd`로 불건전 노드를 감지해 교체합니다(CloudWatch 대시보드에 `Unhealthy Instance Errors` 지표가 있고, static 노드에는 `node_replacement_timeout`이 있습니다).
    HyperPod의 실제 차별점은 세 가지입니다.

    1. **노드 감지·복구 컨트롤 플레인을 AWS가 소유**합니다 — cluster agent, Health Monitoring Agent, Node Recovery System. ParallelCluster에서 같은 역할을 하는 `clustermgtd`는 내 head 노드에서 도는 것이 차이입니다. 다만 **Slurm 컨트롤 플레인 자체(`slurmctld`·`slurmdbd`)는 내 인스턴스 그룹**([`SlurmConfig.NodeType: Controller`](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ref.html))이고 deep health check 대상도 아니며, 컨트롤러 HA는 내가 설계하는 아키텍처입니다.
    2. **deep health check의 깊이** — `stress-ng`·DCGM 레벨 4·EFA loopback·다중 노드 NCCL `all_reduce`까지 돌려 통과하지 못한 노드는 워크로드를 받기 전에 격리·교체합니다. 단 기본 동작이 아니라 인스턴스 그룹별 `OnStartDeepHealthChecks` **옵트인**이고 노드당 두 시간 가까이 걸려 그 동안 노드가 막히므로, AWS는 소규모 클러스터에서는 끄기를 권장합니다.
    3. **중단된 학습 잡의 [auto-resume](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html)** — 노드를 교체한 뒤 잡을 재개합니다. 정확히는 HyperPod가 `srun` 스텝을 다시 띄워 주고 **체크포인트를 읽어 오는 것은 여전히 학습 스크립트의 몫**입니다. GPU 클러스터에서 흔한 GRES가 노드에 붙어 있으면 체크포인트가 아니라 **처음부터 다시 시작**하는 예외도 문서화돼 있습니다.

    즉 "노드를 살리는 것"이 아니라 **"진행 중인 학습을 살리는 것"**이 초점입니다.

복원력이 정리되면 다음 혼동은 배포 기능의 귀속입니다.

??? question "오개념 — “HyperPod로 올리면 blue/green이나 canary 배포도 되겠지”"
    **가드레일은 아닙니다.** HyperPod에도 [추론 플랫폼](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment.html)이 있어 프로덕션 트래픽을 받을 수 있습니다. 하지만 blue/green·canary·rolling 배포 가드레일과 production variant A/B는 **SageMaker AI Inference endpoint**의 기능입니다 — `CreateEndpoint`/`UpdateEndpoint`의 `EndpointConfig` 교체를 기반으로, CloudWatch 알람으로 baking 기간을 감시하며 자동 롤백까지 하는 메커니즘입니다. 적용 대상은 real-time과 asynchronous이고 serverless는 제외됩니다. [제외 목록](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-exclusions.html)은 **Marketplace 컨테이너와 Inf1(Inferentia) 인스턴스 endpoint**이며(rolling 배포는 추가로 serverless·multi-variant endpoint 제외), inference component는 제외 대상이 **아닙니다.** 그리고 제외 기능을 쓰는 endpoint라고 해서 가드레일을 아예 못 쓰는 것이 아니라, **"all at once 트래픽 전환 + baking 기간 없음"의 blue/green으로 폴백**됩니다.
    HyperPod EKS에서 같은 목적을 달성하려면 Kubernetes rolling update로 직접 구성해야 합니다. 기능을 찾을 때는 **"어느 서비스의 기능인가"를 먼저 확인**하세요 — 티어 간 기능 귀속 오류가 아키텍처 결정을 가장 크게 망칩니다.

HyperPod 자체를 단일 제품으로 보는 것도 자주 나오는 착각입니다.

??? question "오개념 — “HyperPod는 하나다”"
    **오케스트레이터가 두 종류입니다.** **Slurm** 방식은 클러스터 안에 컨트롤러·로그인·워커 노드를 두고 `sbatch`/`srun`으로 제출하며, auto-resume은 `srun --auto-resume=1` 플래그로 켭니다.
    **EKS** 방식은 EKS 컨트롤 플레인과 HyperPod 클러스터(워커 노드)를 1:1로 연결하고, 워크로드는 컨테이너/파드로 제출합니다.
    둘은 제출 방식·팀 스킬셋·관측 스택이 모두 다릅니다. 한쪽 문서를 읽고 다른 쪽 동작을 가정하지 마세요.

마지막은 컨테이너 이미지와 호스트 이미지를 같은 층으로 보는 착각입니다.

??? question "오개념 — “DLC는 SageMaker 전용이다”"
    **아닙니다.** DLC(Deep Learning Containers)는 **워크로드 컨테이너 이미지**라서 EC2·ECS·EKS(HyperPod-EKS 포함) 어디서나 실행됩니다.
    비교 대상으로 자주 등장하는 **DLAMI는 노드(호스트) 머신 이미지**이며 층이 다릅니다. "관리형이니까 DLC, 자체 구성이니까 DLAMI"라는 대응은 성립하지 않습니다.

---

## 언제 무엇을 쓰나

절대적인 정답은 없고, **조건에 따라** 갈립니다. 조건을 하나만 남긴다면 [TCO 세 칸](00_overview.md#인프라-비용은-tco의-한-칸일-뿐입니다) 중 운영·규정 준수 칸을 **내가 이미 내고 있는지**입니다.

- **SageMaker AI Training Job / Endpoint를 고르세요, 만약** 파인튜닝이나 서빙이 간헐적이고, 전담 인프라 팀이 없고, "지금 잡 하나를 돌려 결과를 보는 것"이 목적인 경우. 이 킷의 모든 트랙이 이 경우에 해당합니다.
- **HyperPod를 고르세요, 만약** 다수의 GPU를 **오래** 점유하며 여러 팀이 큐를 공유해야 하고, 노드 장애로 며칠짜리 학습이 처음부터 다시 시작되는 것을 감당할 수 없는 경우. 이때 Slurm과 EKS 중에서는 팀이 이미 쓰는 스택(HPC 관행이면 Slurm, Kubernetes 표준이 있으면 EKS)을 따르는 편이 운영 비용이 낮습니다.
- **EC2 자체 구성을 고르세요, 만약** 커널·드라이버·토폴로지를 직접 통제해야 하거나(특수 빌드, 실험적 라이브러리), 이미 EC2 기반 표준·자동화 자산이 충분해 관리형 계층이 오히려 제약이 되는 경우. 후자가 곧 "운영·규정 준수 칸을 이미 지불했다"는 상태이고, 그때는 시간당 단가 비교가 실제로 유효합니다.
- **on-prem을 고르세요, 만약** GPU 사용률이 지속적으로 높아 감가상각이 시간당 요금보다 유리하거나, 데이터를 물리적으로 외부에 낼 수 없는 규제·계약 요건이 있는 경우. 반대로 **on-prem이 지는 지점은 두 가지**입니다 — (1) **탄력성**: 이번 주에만 GPU 8장이 더 필요할 때 살 수 없습니다, (2) **차별화되지 않는 운영 작업**: 드라이버 업그레이드, 장애 부품 교체, 용량 계획에 들어가는 시간은 모델 품질에 아무것도 기여하지 않습니다.
- **혼합을 고르세요, 만약** 위 조건이 워크로드마다 다른 경우. 실제로 흔한 조합은 "정기 대규모 사전학습은 HyperPod 또는 on-prem, 간헐적 파인튜닝과 프로덕션 서빙은 SageMaker AI"입니다. HyperPod도 추론 플랫폼을 갖추고 있으니 "HyperPod는 학습만"이라고 못 박지는 마세요 — 서빙 쪽에서 갈리는 지점은 **관리형 배포 가드레일**(알람 기반 baking·자동 롤백)이고, 그것은 SageMaker AI endpoint의 기능입니다.

??? tip "결정을 미루는 것도 선택입니다"
    처음부터 클러스터를 고민하지 마세요. **Training Job으로 시작해 잡이 실제로 얼마나 자주, 얼마나 길게 도는지 측정한 뒤** 클러스터로 옮기는 편이 거의 항상 저렴합니다.
    HyperPod가 해결하는 문제(장애로 며칠짜리 학습이 날아가는 것, 큐 경합)는 **그 규모에 도달한 뒤에** 생깁니다.

---

## 이 킷에서는

개념이 어느 노트북에 대응하는지 정리하면 다음과 같습니다(플래그십 트랙 `tracks/01_extraction_to_json/` 기준).

| 노트북 | 만드는 SageMaker 리소스 | 개념 |
|---|---|---|
| `00_setup.ipynb` | 없음(자격증명·role·버킷 확인) | 사전 준비. **role**은 SageMaker가 내 S3·ECR·CloudWatch에 접근할 때 assume하는 IAM 신분입니다([실행 role이 매개하는 것](#실행-role이-매개하는-것--s3와-ecr)) |
| `01_data_and_synthetic.ipynb` | 없음(로컬 `data/train.jsonl` 생성) | 학습 입력 준비 |
| `02_train_sft_sagemaker.ipynb` | ✅ 생성 — **Training Job** (`ModelTrainer.train()`) | 데이터 S3 업로드 → 잡 제출 → `model_data` 확보 |
| `02a_train_grpo_sagemaker.ipynb` (선택) | ✅ 생성 — **Training Job** (SFT→GRPO 정련) | 추출·분류 트랙만 제공 |
| `02b_local_serve.ipynb` (선택) | 없음(로컬 vLLM) | 배포 전 프리플라이트 |
| `03_deploy_endpoint.ipynb` | ✅ 생성 — **Endpoint** (+ `EndpointConfig` + `Model`) | `model_data` → 상시 서버. **여기서 시간당 과금이 시작됩니다** |
| `04_evaluate.ipynb` | 없음(endpoint 호출) | held-out 평가 |
| `05_agentic_strands.ipynb` | 없음(endpoint + Bedrock 호출) | agentic loop |
| `06_agentcore_deploy.ipynb` | AgentCore Runtime(SageMaker와 별개 과금) | 프로덕션 배포 |
| `99_cleanup.ipynb` | **삭제** — Endpoint → EndpointConfig → Model | **과금 중지. 반드시 실행** |

- Training Job을 만드는 노트북은 `02`(및 선택 `02a`)이고, **끝나면 인스턴스가 자동 해제**되므로 별도 정리가 필요 없습니다. 남는 것은 S3 아티팩트와 CloudWatch 로그(용량당 과금)입니다.
- Endpoint를 만드는 노트북은 `03`이며, **삭제하는 노트북은 `99`뿐입니다.** 삭제 순서가 정해져 있고(Endpoint → EndpointConfig → Model), `ModelBuilder`가 `model-42c30d1e` 같은 임의 이름을 만들기 때문에 `endpoint_name`만으로 지우면 Model이 조용히 남습니다(실측 2026-07-31).
- 멀티모달 트랙(`tracks/05_multimodal_extraction/`)은 노트북 세트가 짧습니다 — Training Job은 `02_train_mm_sagemaker.ipynb`, Endpoint는 `03_deploy_mm_endpoint.ipynb`, 정리는 동일하게 `99_cleanup.ipynb`입니다.
- 학습 스크립트는 `tracks/*/scripts/train.py` 하나로, 로컬 `--dry_run`과 SageMaker Training Job에서 **같은 파일**이 돕니다. 먼저 로컬에서 파이프라인을 검증하고 클라우드로 제출하는 것이 이 킷의 규율입니다([시작하기](getting_started.md)의 방식 B).

다음 단계는 [실행 런북](RUN_E2E.md)입니다. 개념을 잡았다면 그 문서의 순서를 그대로 따라가면 됩니다.

---

## 관련 리포지토리 파일

이 문서의 개념이 코드로 나타나는 곳입니다(플래그십 트랙 `tracks/01_extraction_to_json/` 기준).

경로 계약과 Training Job:

- `tracks/01_extraction_to_json/scripts/train.py` — `SM_CHANNEL_TRAIN`으로 입력 채널을 찾고 `SM_MODEL_DIR`을 출력 기본값으로 씀. `save_total_limit=1`로 아티팩트 크기를 억제
- `tracks/01_extraction_to_json/02_train_sft_sagemaker.ipynb` — `ModelTrainer` + `SourceCode`/`Compute`/`InputData`/`StoppingCondition` 조립, `MAX_RUNTIME_HOURS`로 시간 한도 명시

Endpoint의 생성과 삭제:

- `tracks/01_extraction_to_json/03_deploy_endpoint.ipynb` — `model_data`를 real-time endpoint로 배포하고 invoke 스모크 테스트
- `tracks/01_extraction_to_json/99_cleanup.ipynb` — Endpoint → EndpointConfig → Model 순서로 삭제해 과금을 멈춤

공용 헬퍼:

- `common/config.py` — `resolve_sagemaker_role()`이 env → `get_execution_role()` → IAM 자동 탐지 → (opt-in) DefaultRole 생성 순으로 실행 role을 해석
- `common/dlc.py` — 학습·서빙 DLC 이미지 URI 해석(`DLC_IMAGE_URI` 환경변수 오버라이드)
- `common/aws_utils.py` — endpoint 호출(`invoke_sagemaker_chat`), CloudWatch 링크(`cw_links`), 변경분만 올리는 S3 업로드(`upload_if_changed`)
