# 시작하기

이 kit이 처음이라면 **이 문서 하나만 위에서 아래로** 따라 하면 됩니다.
"노트북은 어디?", "dry-run 어떻게?", "SageMaker는 어떻게 돌려?"에 순서대로 답합니다.

!!! info "이 문서의 범위"
    설치부터 첫 학습까지를 다룹니다. kit 전체 구조는 [전체 지도](00_overview.md),
    파이프라인 완주 runbook은 [실행 runbook](RUN_E2E.md)을 보세요.

---

## 큰 그림 — 3가지 실행 방식

| 방식 | 무엇 | GPU/AWS 필요? | 언제 |
|---|---|---|---|
| **A. 스모크 테스트** | 순수 로직 검증(데이터 어댑터·포맷터·메트릭) | ❌ 아무것도 불필요 | 코드가 멀쩡한지 5초 확인 |
| **B. 로컬 GPU dry-run** | `train.py`를 내 GPU에서 소량·짧게 실제 학습 | ✅ 필요 — GPU만 (AWS 불필요) | 학습 파이프라인이 도는지 확인 |
| **C. SageMaker E2E** | 태스크별 실습 코스(노트북 한 세트)로 클라우드 학습→배포→agentic 완주 | ✅ 필요 — AWS 계정 + 과금 | 진짜 파인튜닝·서빙 |

**초심자 추천 순서: A → B → C** (아래에서 그대로 따라 하면 됩니다.)

!!! tip "전체를 완주할 거라면"
    단계별 핸드오프·비용·체크리스트·문제해결은 [실행 runbook](RUN_E2E.md)에 정리돼 있습니다.

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
    `pip install 'litellm>=1.93.0'`. 코어 파이프라인(학습·배포·agentic)은 boto3를 직접
    호출하므로 litellm 없이 완결됩니다.

---

## 방식 A — 스모크 테스트

가장 먼저 이걸로 코드가 멀쩡한지 확인하세요. **모델 다운로드도, AWS도 필요 없습니다.**

```bash
cd ~/sagemaker-finetune-serve-e2e
python tests/test_smoke.py
```
기대 출력: 마지막 줄이 `9/9 passed`(테스트를 추가하면 그만큼 늘어납니다). 실패 없이 끝나는지만 보면 됩니다 — 데이터 어댑터·Gemma 포맷터·합성 파서·메트릭 로직·멀티모달 코스 등록을 검증합니다.

---

## 방식 B — 로컬 GPU dry-run

**"파이썬 dry-run"이 바로 이겁니다.** `scripts/train.py`는 self-contained라 로컬과 SageMaker에서
같은 파일이 돕니다. `--dry_run`이면 소량(≤32행)·1 epoch·짧은 시퀀스로 파이프라인만 검증합니다.

### HF 캐시 위치 지정 (선택)
모델 가중치(~24GB)를 홈 기본 캐시 말고 별도 폴더에:
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

## 방식 C — SageMaker E2E

**"초심자를 위한 주피터 노트북"이 바로 이겁니다.** kit에는 태스크 하나를 데이터 준비부터 학습·배포·평가·정리까지
끝내는 **실습 코스가 5개** 있고, 각 텍스트 코스 폴더에
`00`~`06`,`99` 노트북이 (+ 선택 `02a`/`02b`), 멀티모달 코스에는 별도의 짧은 세트가 있습니다. 번호 순서대로 실행하면 됩니다.
폴더 이름과 코드 식별자는 초기 이름인 `track`을 그대로 씁니다(`tracks/`, `track_data.py`) — 아래에서 말하는 코스와 같은 것입니다.

### 어느 노트북부터
```
tracks/01_extraction_to_json/     ← 플래그십 (여기부터 시작 추천)
├── 00_setup.ipynb                ① 환경·자격증명·설치 확인
├── 01_data_and_synthetic.ipynb   ② 데이터 준비 + grounded 합성
├── 02_train_sft_sagemaker.ipynb      ③ SageMaker 학습 Job (+ 로컬 dry-run 셀 포함)
├── 02a_train_grpo_sagemaker.ipynb    (선택) SFT→GRPO 정련 (RLHF) — 추출·분류 코스만
├── 02b_local_serve.ipynb             (선택) 배포 전 로컬 vLLM 검증
├── 03_deploy_endpoint.ipynb      ④ real-time endpoint 배포 (vLLM 기본)
├── 04_evaluate.ipynb             ⑤ held-out 평가 (성공기준 수치화)
├── 05_agentic_strands.ipynb      ⑥ SLM + Bedrock Claude agentic 루프
├── 06_agentcore_deploy.ipynb     ⑦ AgentCore 프로덕션 배포
└── 99_cleanup.ipynb              리소스 삭제 (과금 중단 — 반드시 실행)
```
다른 텍스트 task는 `02_classification/`, `03_summarization/`, `04_domain_qa/` — **구조·순서 동일**.
- `02a_train_grpo_sagemaker`(SFT→GRPO 정련)는 **추출·분류 코스에만** 있습니다(리워드가 프로그램적으로 계산됨). 요약·domain_qa에는 없습니다.
- `02b_local_serve`(배포 전 로컬 vLLM 프리플라이트)는 모든 텍스트 코스에서 선택적으로 제공됩니다.

**멀티모달 코스는 구조가 다릅니다** — `tracks/05_multimodal_extraction/` (이미지 → 구조화 JSON 추출, 영수증, gemma-4 vision)은
합성 데이터 단계가 없고 이미지 입력이라 노트북 세트가 짧습니다:
```
tracks/05_multimodal_extraction/  ← 이미지 입력 (텍스트 코스와 별개 구조)
├── 00_setup.ipynb                ① 환경·자격증명·설치 확인
├── 01_data_explore.ipynb         ② cord-v2 영수증 이미지 + 구조화 JSON 탐색 (합성 단계 없음)
├── 02_train_mm_sagemaker.ipynb   ③ SageMaker 학습 (vision tower 동결 + language LoRA)
├── 03_deploy_mm_endpoint.ipynb   ④ 멀티모달 endpoint 배포 (이미지 입력 허용, 텍스트 전용 re-export 아님)
└── 99_cleanup.ipynb              리소스 삭제 (과금 중단 — 반드시 실행)
```
멀티모달 코스가 쓰는 시드와 스크립트:

- `naver-clova-ix/cord-v2` — 시드 데이터셋(cc-by-4.0, ungated). 영수증 이미지 + 구조화 JSON 라벨
- `tracks/05_multimodal_extraction/scripts/train_mm.py` — 이미지→JSON 멀티모달 SFT. `AutoModelForImageTextToText` + `AutoProcessor`로 이미지를 처리하고, vision tower를 유지해 텍스트 re-export를 하지 않습니다

### 주피터 실행
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
export DRY_RUN=1                  # 먼저 파이프라인 검증, 실제 클라우드 학습 시 0
```
`DRY_RUN=1`이면 노트북이 소량·저비용으로 파이프라인만 확인합니다. 확인 후 `0`으로 바꿔 실제 실행하세요.

??? tip "HF 토큰을 한 번만 저장하기 (gated 모델을 쓸 때)"
    `config.get_hf_token()`은 **env(`HF_TOKEN`) → `hf auth login` 저장 토큰** 순으로 조회하므로
    아래 둘 중 무엇이든 자동 인식됩니다.

    - **`hf auth login`** (권장) — 한 번 실행하면 토큰이 파일로 저장되고 config가 읽습니다.
    - **셸 프로파일** — `echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc` 후 새 터미널.

    gemma-4 계열(ungated)만 쓸 거라면 토큰 자체가 필요 없습니다.

    **커스텀 캐시 경로를 쓴다면** — `HF_HOME=~/hf-cache`로 `hf auth login`을 했다면 토큰이
    `~/hf-cache/token`에 저장됩니다. 이때 **노트북 프로세스에도 같은 `HF_HOME`이 있어야**
    `huggingface_hub`이 그 토큰을 찾습니다.

    ``` bash
    echo 'export HF_HOME=/home/ubuntu/hf-cache' >> ~/.bashrc
    ```

    `.bashrc`는 비대화형 셸에서 조기 return될 수 있으니 `source` 대신 **새 터미널을 열어**
    반영하세요. 이미 실행 중인 커널은 재시작해야 합니다.

### 비용 주의
- real-time endpoint는 **삭제 전까지 시간당 과금**됩니다 → 실습 후 반드시 **`99_cleanup.ipynb`** 실행.
- Bedrock 호출·AgentCore도 과금 → 자세한 건 각 노트북 상단 경고 + [`docs/04_sagemaker_inference.md`](04_sagemaker_inference.md).

---

## 더 알아보기

문서는 파일명 번호가 곧 읽는 순서입니다. 처음이라면 00부터 차례로, 특정 단계만 필요하면 해당 항목으로 가세요.

- [전체 지도](00_overview.md) — 전체 지도. 노트북과 문서 매핑
- [SageMaker 기초](01_sagemaker_basics.md) — 개념. Training Job vs Endpoint, 경로 규약, 수명과 과금 (방식 C 전에 읽기 권장)
- [합성 데이터](02_synthetic_data.md) — 데이터 준비. grounded 합성과 critique/refine
- [파인튜닝](03_finetuning.md) — 학습. PyTorch DLC + TRL LoRA/QLoRA
- [SageMaker 추론](04_sagemaker_inference.md) — 배포. 추론 4옵션과 endpoint 선택 기준
- [서빙 컨테이너](05_serving_containers.md) — 배포. vLLM vs SGLang vs DJL LMI 엔진 선택
- [Agentic loop](06_agentic.md) — 활용. Strands + Bedrock Claude, AgentCore 배포
- [실행 runbook](RUN_E2E.md) — E2E 완주 runbook. 단계별 핸드오프와 비용 가드

---

## 자주 막히는 곳
- **"어느 노트북부터?"** → `tracks/01_extraction_to_json/00_setup.ipynb`.
- **"GPU에서 그냥 돌려보고 싶다"** → 위 [방식 B](#방식-b--로컬-gpu-dry-run)(dry-run), AWS 불필요.
- **"AWS 없이 코드만 확인"** → 위 [방식 A](#방식-a--스모크-테스트)(스모크 테스트).
- **"gated 모델 접근 오류"** → gemma-3은 HF 약관 수락 + `HF_TOKEN` 필요. 또는 ungated `gemma-4-12B-it` 사용.
- **"과금이 무섭다"** → `DRY_RUN=1`로 시작, 끝나면 `99_cleanup.ipynb` 필수.
