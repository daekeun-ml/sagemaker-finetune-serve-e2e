# 시작하기

이 프로젝트를 처음 접한다면 이 문서의 순서대로 진행하세요.
"노트북은 어디에 있나?", "dry-run은 어떻게 하나?", "Amazon SageMaker AI에서는 어떻게 실행하나?"에 순서대로 답합니다.

!!! info "이 문서의 범위"
    설치부터 첫 학습까지를 다룹니다. 프로젝트 전체 구조는 [전체 지도](00_overview.md),
    전체 실행 절차는 [E2E 실행 가이드](RUN_E2E.md)를 보세요.

---

## 실행 방식 3가지

| 방식 | 무엇 | GPU/AWS 필요? | 언제 |
|---|---|---|---|
| **A. smoke test** | 순수 로직 검증(데이터 어댑터, 포맷터, 메트릭) | 불필요 | 기본 로직이 정상인지 빠르게 확인 |
| **B. 로컬 GPU dry-run** | `train.py`를 내 GPU에서 소량, 짧게 실제 학습 | ✅ 필요: GPU만 (AWS 불필요) | 학습 파이프라인이 도는지 확인 |
| **C. SageMaker AI E2E** | 태스크별 실습 코스로 SageMaker AI 학습, 배포, agentic 단계 실행 | ✅ 필요: AWS 계정, 사용량에 따른 비용 | SageMaker AI에서 전체 과정 실행 |

**처음 실행할 때 권장하는 순서: A → B → C**

!!! tip "전체 과정을 실행한다면"
    단계별 핸드오프, 비용, 체크리스트, 문제 해결은 [E2E 실행 가이드](RUN_E2E.md)에 정리돼 있습니다.

---

## 설치

```bash
# uv 미설치 시 (이미 설치돼 있으면 건너뛰기)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/sagemaker-finetune-serve-e2e
uv venv --python 3.12            # .venv 생성
source .venv/bin/activate        # 활성화
uv pip install -r pyproject.toml # 코어 설치 (sagemaker/boto3/transformers/trl/peft/strands...)
```
> pip만 쓰려면: `pip install -r requirements.txt`
> 최신으로 올리려면: `uv lock --upgrade` (또는 `uv lock --upgrade-package transformers`)

!!! warning "LiteLLM은 코어 의존성이 아닙니다"
    litellm이 요구하는 `importlib-metadata>=8`이 sagemaker(`<7`)와 충돌합니다.
    `common/llm_gateway.py`(LiteLLM 경유 호출)가 필요하면 **별도 환경**에 설치하세요:
    `pip install 'litellm>=1.93.0'`. 코어 파이프라인(학습, 배포, agentic)은 boto3를 직접
    호출하므로 litellm 없이 완결됩니다.

---

## 방식 A: smoke test

먼저 코드의 순수 로직을 확인합니다. **모델 다운로드도, AWS도 필요 없습니다.**

```bash
cd ~/sagemaker-finetune-serve-e2e
python tests/test_smoke.py
```
기대 출력: 마지막 줄이 `9/9 passed`(테스트를 추가하면 그만큼 늘어납니다). 실패 없이 끝나는지만 보면 됩니다. 데이터 어댑터와 Gemma 포맷터와 합성 파서와 메트릭 로직과 멀티모달 코스 등록을 검증합니다.

---

## 방식 B: 로컬 GPU dry-run

`scripts/train.py`는 로컬과 SageMaker AI에서 같은 파일을 사용합니다.
`--dry_run`이면 최대 32행, 1 epoch, 짧은 시퀀스로 학습 흐름만 검증합니다.

### HF 캐시 위치 지정 (선택)
모델 가중치(~24GB)를 기본 캐시가 아닌 별도 폴더에 저장하려면:
```bash
export HF_HOME=~/hf-cache        # EBS 루트 여유 공간
```

### dry-run용 소량 데이터 만들기
```bash
python tests/prepare_dryrun_data.py --track extraction --n 24 --out /tmp/dryrun_train.jsonl
```
> `--track` 은 `extraction | classification | summarization | domain_qa` 중 하나.

### dry-run 실행
```bash
python tracks/01_extraction_to_json/scripts/train.py \
    --dry_run \
    --model_id google/gemma-4-12B-it \
    --train_file /tmp/dryrun_train.jsonl \
    --use_qlora True \
    --merge_adapter False \
    --max_seq_length 512 \
    --output_dir /tmp/dryrun_out
```
기대: 가중치 다운로드 → QLoRA 학습 1 epoch → `어댑터 저장` → `DRY-RUN 완료` 로그.
(L40S 48GB에서 실제 검증됨. 첫 실행은 가중치 다운로드로 수 분 소요, 이후 캐시 재사용.)

!!! tip "더 작고 빠르게"
    `--model_id google/gemma-3-1b-it`로 줄일 수 있습니다. 단 gemma-3은 gated라
    `HF_TOKEN`이 필요합니다. gemma-4 계열은 apache-2.0/ungated라 토큰이 필요 없습니다.

---

## 방식 C: SageMaker AI E2E

프로젝트에는 태스크 하나의 데이터 준비, 학습, 배포, 평가, 정리를 다루는
**Jupyter 실습 코스가 5개** 있습니다. 각 텍스트 코스 폴더에는
`00`~`06`,`99` 노트북이 (+ 선택 `02a`/`02b`), 멀티모달 코스에는 별도의 짧은 세트가 있습니다. 번호 순서대로 실행하면 됩니다.
폴더 이름과 코드 식별자는 초기 이름인 `track`을 그대로 씁니다(`tracks/`, `track_data.py`). 본문의 "코스"와 같은 대상을 가리킵니다.

### 어느 노트북부터
```
tracks/01_extraction_to_json/     ← 플래그십 (여기부터 시작 추천)
├── 00_setup.ipynb                ① 환경, 자격증명, 설치 확인
├── 01_data_and_synthetic.ipynb   ② 데이터 준비 + grounded 합성
├── 02_train_sft_sagemaker.ipynb      ③ SageMaker AI 학습 Job (+ 로컬 dry-run 셀 포함)
├── 02a_train_grpo_sagemaker.ipynb    (선택) SFT→GRPO 정련 (RLHF): 추출, 분류 코스만
├── 02b_local_serve.ipynb             (선택) 배포 전 로컬 vLLM 검증
├── 03_deploy_endpoint.ipynb      ④ real-time endpoint 배포 (vLLM 기본)
├── 04_evaluate.ipynb             ⑤ held-out 평가 (성공기준 수치화)
├── 05_agentic_strands.ipynb      ⑥ SLM + Bedrock Claude agentic 루프
├── 06_agentcore_deploy.ipynb     ⑦ AgentCore production 배포
└── 99_cleanup.ipynb              리소스 삭제 (과금 중단: 반드시 실행)
```
다른 텍스트 코스는 `02_classification/`, `03_summarization/`, `04_domain_qa/`이고 **구조와 순서가 같습니다**.
- `02a_train_grpo_sagemaker`(SFT→GRPO 정련)는 reward를 코드로 계산할 수 있는 **추출과 분류 코스에만** 있습니다. 요약과 domain_qa에는 없습니다.
- `02b_local_serve`(배포 전 로컬 vLLM preflight)는 모든 텍스트 코스에서 선택적으로 제공됩니다.

**멀티모달 코스는 구조가 다릅니다.** `tracks/05_multimodal_extraction/`(이미지 → 구조화 JSON 추출, 영수증, gemma-4 vision)은
합성 데이터 단계가 없고 이미지 입력이라 노트북 세트가 짧습니다:
```
tracks/05_multimodal_extraction/  ← 이미지 입력 (텍스트 코스와 별개 구조)
├── 00_setup.ipynb                ① 환경, 자격증명, 설치 확인
├── 01_data_explore.ipynb         ② cord-v2 영수증 이미지 + 구조화 JSON 탐색 (합성 단계 없음)
├── 02_train_mm_sagemaker.ipynb   ③ SageMaker AI 학습 (vision tower 동결 + language LoRA)
├── 03_deploy_mm_endpoint.ipynb   ④ 멀티모달 endpoint 배포 (이미지 입력 허용, 텍스트 전용 re-export 아님)
└── 99_cleanup.ipynb              리소스 삭제 (과금 중단: 반드시 실행)
```
멀티모달 코스가 쓰는 시드와 스크립트:

- `naver-clova-ix/cord-v2`: 시드 데이터셋(cc-by-4.0, ungated). 영수증 이미지 + 구조화 JSON 라벨
- `tracks/05_multimodal_extraction/scripts/train_mm.py`: 이미지→JSON 멀티모달 SFT. `AutoModelForImageTextToText` + `AutoProcessor`로 이미지를 처리하고, vision tower를 유지해 텍스트 re-export를 하지 않습니다

### Jupyter 실행
```bash
cd ~/sagemaker-finetune-serve-e2e
source .venv/bin/activate
jupyter lab           # 브라우저에서 tracks/01_.../00_setup.ipynb 부터 순서대로
```

### 노트북 실행 전 필요한 것
```bash
export AWS_REGION=us-west-2         # config 기본값. .env의 DLC 이미지 URI 리전과 일치해야 합니다
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5   # 정확한 ID는 콘솔에서 확인
# export HF_TOKEN=hf_...          # gemma-3 등 gated 모델 쓸 때만
export DRY_RUN=1                  # 데이터 준비와 평가 규모를 줄여 먼저 검증
```
`DRY_RUN=1`은 데이터 준비, 평가, 로컬 dry-run 규모를 줄입니다. SageMaker AI Training Job의 `MAX_TRAIN_SAMPLES`와 `EPOCHS`, endpoint 비용은 자동으로 줄이지 않습니다. 자세한 범위는 [DRY_RUN으로 먼저 검증하는 이유](RUN_E2E.md#dry_run으로-먼저-검증하는-이유)를 확인하세요.

??? tip "HF 토큰을 한 번만 저장하기 (gated 모델을 쓸 때)"
    `config.get_hf_token()`은 **env(`HF_TOKEN`) → `hf auth login` 저장 토큰** 순으로 조회하므로
    아래 둘 중 무엇이든 자동 인식됩니다.

    - **`hf auth login`** (권장): 한 번 실행하면 토큰이 파일로 저장되고 config가 읽습니다.
    - **셸 프로파일**: `echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc` 후 새 터미널.

    gemma-4 계열(ungated)만 쓸 거라면 토큰 자체가 필요 없습니다.

    **커스텀 캐시 경로를 쓴다면**, `HF_HOME=~/hf-cache`로 `hf auth login`을 했다면 토큰이
    `~/hf-cache/token`에 저장됩니다. 이때 **노트북 프로세스에도 같은 `HF_HOME`이 있어야**
    `huggingface_hub`이 그 토큰을 찾습니다.

    ``` bash
    echo 'export HF_HOME=/home/ubuntu/hf-cache' >> ~/.bashrc
    ```

    `.bashrc`는 비대화형 셸에서 조기 return될 수 있으니 `source` 대신 **새 터미널을 열어**
    반영하세요. 이미 실행 중인 커널은 재시작해야 합니다.

### 비용 주의
- real-time endpoint는 **삭제 전까지 시간당 과금**됩니다 → 실습 후 반드시 **`99_cleanup.ipynb`** 실행.
- Bedrock 호출과 AgentCore도 과금 → 자세한 내용은 각 노트북 상단 경고와 [`docs/04_sagemaker_inference.md`](04_sagemaker_inference.md) 참고.

---

## 더 알아보기

문서는 파일명 번호가 곧 읽는 순서입니다. 처음이라면 00부터 차례로, 특정 단계만 필요하면 해당 항목으로 가세요.

- [전체 지도](00_overview.md): 전체 지도. 노트북과 문서 매핑
- [SageMaker AI 기초](01_sagemaker_basics.md): 개념. Training Job vs Endpoint, 경로 규약, 수명과 과금 (방식 C 전에 읽기 권장)
- [합성 데이터](02_synthetic_data.md): 데이터 준비. grounded 합성과 critique/refine
- [파인튜닝](03_finetuning.md): 학습. PyTorch DLC + TRL LoRA/QLoRA
- [SageMaker AI 추론](04_sagemaker_inference.md): 배포. 추론 4옵션과 endpoint 선택 기준
- [서빙 컨테이너](05_serving_containers.md): 배포. vLLM vs SGLang vs DJL LMI 엔진 선택
- [Agentic loop](06_agentic.md): 활용. Strands + Bedrock Claude, AgentCore 배포
- [E2E 실행 가이드](RUN_E2E.md): 단계별 핸드오프, 비용, 완료 조건

---

## 자주 묻는 항목
- **"어느 노트북부터?"** → `tracks/01_extraction_to_json/00_setup.ipynb`.
- **"로컬 GPU에서 먼저 실행하고 싶다"** → 위 [방식 B](#방식-b-로컬-gpu-dry-run), AWS 불필요.
- **"AWS 없이 코드만 확인하고 싶다"** → 위 [방식 A](#방식-a-smoke-test).
- **"gated 모델 접근 오류"** → gemma-3은 HF 약관 수락 + `HF_TOKEN` 필요. 또는 ungated `gemma-4-12B-it` 사용.
- **"비용을 줄이고 싶다"** → `DRY_RUN=1`로 데이터 규모를 줄이고 Training Job 파라미터도 직접 낮추세요. endpoint 확인 후 `99_cleanup.ipynb`를 실행합니다.
