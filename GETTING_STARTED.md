# 시작하기 (Getting Started) — 초심자용

이 킷이 처음이라면 **이 문서 하나만 위에서 아래로** 따라 하면 됩니다.
"노트북은 어디?", "dry-run 어떻게?", "SageMaker는 어떻게 돌려?" 에 순서대로 답합니다.

> 📍 **지금 위치**: `~/gemma-e2e-toolkit` (전체 지도는 [`docs/00_overview.md`](docs/00_overview.md))

---

## 0. 큰 그림 — 3가지 실행 방식이 있다

| 방식 | 무엇 | GPU/AWS 필요? | 언제 |
|---|---|---|---|
| **A. 스모크 테스트** | 순수 로직 검증(데이터 어댑터·포맷터·메트릭) | ❌ 아무것도 불필요 | 코드가 멀쩡한지 5초 확인 |
| **B. 로컬 GPU dry-run** | `train.py`를 내 GPU에서 소량·짧게 실제 학습 | ✅ GPU만 (AWS 불필요) | 학습 파이프라인이 도는지 확인 |
| **C. SageMaker E2E** | 노트북으로 클라우드 학습→배포→agentic | ✅ AWS 계정 + 과금 | 진짜 파인튜닝·서빙 |

**초심자 추천 순서: A → B → C** (아래에서 그대로 따라 하면 됩니다.)

> 🔁 전체 파이프라인을 **처음부터 끝까지 실제로 완주**할 거라면 → [`docs/RUN_E2E.md`](docs/RUN_E2E.md) (단계별 핸드오프·비용·체크리스트·문제해결 런북).

---

## 1. 설치 (uv 권장, 1회)

```bash
# uv 미설치 시 (이미 설치돼 있으면 건너뛰기)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/gemma-e2e-toolkit
uv venv --python 3.12            # .venv 생성
source .venv/bin/activate        # 활성화
uv pip install -r pyproject.toml # 코어 설치 (sagemaker/boto3/transformers/trl/peft/strands...)
```
> pip만 쓰려면: `pip install -r requirements.txt`
> 최신으로 올리려면: `uv lock --upgrade` (또는 `uv lock --upgrade-package transformers`)
>
> ⚠️ **LiteLLM 게이트웨이는 코어에 없음** — litellm(현행)이 요구하는 `importlib-metadata>=8`이
> sagemaker(`<7`)와 충돌하기 때문. `common/llm_gateway.py`(LiteLLM 경유 Bedrock/endpoint)가 필요하면
> **별도 환경**에 설치하세요: `pip install 'litellm>=1.93.0'`. 코어 파이프라인(학습·배포·agentic)은
> boto3 직접 호출로 동작하므로 litellm 없이도 완결됩니다.

---

## 2. 방식 A — 스모크 테스트 (5초, GPU/AWS 불필요)

가장 먼저 이걸로 코드가 멀쩡한지 확인하세요. **모델 다운로드도, AWS도 필요 없습니다.**

```bash
cd ~/gemma-e2e-toolkit
python tests/test_smoke.py
```
기대 출력: `7/7 passed`. (데이터 어댑터·Gemma 포맷터·합성 파서·메트릭 로직 검증)

---

## 3. 방식 B — 로컬 GPU dry-run (내 GPU에서 실제 소량 학습)

**"파이썬 dry-run"이 바로 이겁니다.** `scripts/train.py`는 self-contained라 로컬과 SageMaker에서
같은 파일이 돕니다. `--dry_run`이면 소량(≤32행)·1 epoch·짧은 시퀀스로 파이프라인만 검증합니다.

### 3-1. HF 캐시 위치 지정 (선택, 권장)
모델 가중치(~24GB)를 홈 기본 캐시 말고 별도 폴더에:
```bash
export HF_HOME=~/hf-cache        # EBS 루트 여유 공간
```

### 3-2. dry-run용 소량 학습 데이터 만들기 (Bedrock 합성 없이, 시드만)
```bash
python tests/prepare_dryrun_data.py --track extraction --n 24 --out /tmp/dryrun_train.jsonl
```
> `--track` 은 `extraction | classification | summarization | domain_qa` 중 하나.

### 3-3. dry-run 실행 (ungated 모델이라 HF 토큰 불필요)
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
기대: 가중치 다운로드 → QLoRA 학습 1 epoch → `✅ 어댑터 저장` → `🧪 DRY-RUN 완료`.
(L40S 46GB에서 실제 검증됨. 첫 실행은 가중치 다운로드로 수 분 소요, 이후 캐시 재사용.)

> 💡 **더 작고 빠르게**: `--model_id google/gemma-3-1b-it` (단, gemma-3은 gated → `export HF_TOKEN=hf_...` 필요).
> gemma-4 계열은 apache-2.0/ungated라 토큰이 필요 없습니다.

---

## 4. 방식 C — SageMaker E2E (노트북, 클라우드)

**"초심자를 위한 주피터 노트북"이 바로 이겁니다.** 킷에는 **5개 트랙**이 있고, 각 텍스트 트랙 폴더에
`00`~`06`,`99` 노트북이 (+ 선택 `02a`/`02b`), 멀티모달 트랙에는 별도의 짧은 세트가 있습니다. 번호 순서대로 실행하면 됩니다.

### 4-1. 어느 노트북부터?
```
tracks/01_extraction_to_json/     ← 플래그십 (여기부터 시작 추천)
├── 00_setup.ipynb                ① 환경·자격증명·설치 확인
├── 01_data_and_synthetic.ipynb   ② 데이터 준비 + grounded 합성
├── 02_train_sft_sagemaker.ipynb      ③ SageMaker 학습 잡 (+ 로컬 dry-run 셀 포함)
├── 02a_train_grpo_sagemaker.ipynb    (선택) SFT→GRPO 정련 (RLHF) — 추출·분류 트랙만
├── 02b_local_serve.ipynb             (선택) 배포 전 로컬 vLLM 검증
├── 03_deploy_endpoint.ipynb      ④ real-time endpoint 배포 (DJL LMI 기본)
├── 04_evaluate.ipynb             ⑤ held-out 평가 (성공기준 수치화)
├── 04b_sagemaker_eval.ipynb          (선택) ⑤-b SageMaker managed 평가 잡 (🔴 별도 컴퓨트·비용)
├── 05_agentic_strands.ipynb      ⑥ SLM + Bedrock Claude agentic 루프
├── 06_agentcore_deploy.ipynb     ⑦ AgentCore 프로덕션 배포
└── 99_cleanup.ipynb              🔴 리소스 삭제 (과금 중단 — 꼭 실행!)
```
다른 텍스트 task는 `02_classification/`, `03_summarization/`, `04_domain_qa/` — **구조·순서 동일**.
- `02a_train_grpo_sagemaker`(SFT→GRPO 정련)는 **추출·분류 트랙에만** 있습니다(리워드가 프로그램적으로 계산됨). 요약·domain_qa에는 없습니다.
- `02b_local_serve`(배포 전 로컬 vLLM 프리플라이트)는 모든 텍스트 트랙에서 선택적으로 제공됩니다.
- `04b_sagemaker_eval`(SageMaker managed 평가 잡)은 `04_evaluate`(로컬·빠름·저렴)의 **선택 대안**으로, 모든 텍스트 트랙(01~04)에 함께 제공됩니다. 🔴 별도 컴퓨트·비용이 발생합니다. 멀티모달 트랙(05)에는 없습니다.

**멀티모달 트랙은 구조가 다릅니다** — `tracks/05_multimodal_extraction/` (이미지 → 구조화 JSON 추출, 영수증, gemma-4 vision)은
합성 데이터 단계가 없고 이미지 입력이라 노트북 세트가 짧습니다:
```
tracks/05_multimodal_extraction/  ← 이미지 입력 (텍스트 트랙과 별개 구조)
├── 00_setup.ipynb                ① 환경·자격증명·설치 확인
├── 01_data_explore.ipynb         ② cord-v2 영수증 이미지 + 구조화 JSON 탐색 (합성 단계 없음)
├── 02_train_mm_sagemaker.ipynb   ③ SageMaker 학습 (vision tower 동결 + language LoRA)
├── 03_deploy_mm_endpoint.ipynb   ④ 멀티모달 endpoint 배포 (이미지 입력 허용, 텍스트 전용 재-export 아님)
└── 99_cleanup.ipynb              🔴 리소스 삭제 (과금 중단 — 꼭 실행!)
```
> 시드: `naver-clova-ix/cord-v2` (cc-by-4.0, ungated) · 학습 스크립트: `scripts/train_mm.py` (AutoModelForImageTextToText + processor).

### 4-2. 주피터 실행
```bash
cd ~/gemma-e2e-toolkit
source .venv/bin/activate
jupyter lab           # 브라우저에서 tracks/01_.../00_setup.ipynb 부터 순서대로
```

### 4-3. 노트북 실행 전 필요한 것 (00_setup에서 안내)
```bash
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=us.anthropic.claude-...   # 🔴 모델 상세페이지에서 정확 ID 확인
# export HF_TOKEN=hf_...          # gemma-3 등 gated 모델 쓸 때만
export DRY_RUN=1                  # 먼저 파이프라인 검증, 실제 클라우드 학습 시 0
```
> `DRY_RUN=1`이면 노트북이 소량·저비용으로 파이프라인만 확인합니다. 확인 후 `0`으로 바꿔 실제 실행.

> 💡 **HF 토큰을 매번 붙여넣기 싫다면 (한 번만 저장)** — gated 모델(gemma-3 등)을 쓸 때 유용합니다.
> 이 킷의 `config.get_hf_token()`은 **env(`HF_TOKEN`) → `hf auth login` 저장 토큰** 순으로 조회하므로, 아래 둘 중 아무거나면 자동 인식됩니다.
>
> - **방법 A — `hf auth login`** (권장): `hf auth login` 한 번이면 토큰이 파일로 영구 저장되고 config가 읽습니다.
> - **방법 B — 셸 프로파일에 export**: `echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc` → 새 터미널마다 자동 적용.
>
> 🔴 **커스텀 캐시 경로(`HF_HOME`)를 쓴다면 주의** — 예를 들어 모델 캐시를 `~/hf-cache`에 두려고 `HF_HOME=~/hf-cache`로
> `hf auth login`을 했다면, 토큰이 `~/hf-cache/token`에 저장됩니다. 이 경우 **노트북 프로세스에도 같은 `HF_HOME`이 있어야**
> `huggingface_hub`이 그 토큰을 찾습니다. `.bashrc`에 함께 넣어 두면 새 터미널·커널에서 자동 적용됩니다:
> ```bash
> echo 'export HF_HOME=/home/ubuntu/hf-cache' >> ~/.bashrc   # 캐시+토큰 경로 고정
> # 그 뒤 새 터미널에서 jupyter lab 을 실행하면 커널이 HF_HOME을 상속합니다
> ```
> (⚠️ `.bashrc`는 비대화형 셸에선 조기 return 될 수 있으니, `source` 대신 **새 터미널을 열어** 반영하세요. 이미 떠 있는 커널은 재시작해야 합니다.)
>
> - ungated(gemma-4 계열)만 쓸 거라면 토큰 자체가 필요 없습니다.

### 4-4. 🔴 비용 주의
- real-time endpoint는 **삭제 전까지 시간당 과금**됩니다 → 실습 후 반드시 **`99_cleanup.ipynb`** 실행.
- Bedrock 호출·AgentCore도 과금 → 자세한 건 각 노트북 상단 경고 + [`docs/01_sagemaker_inference.md`](docs/01_sagemaker_inference.md).

---

## 5. 더 알아보기 (docs/)
- [`00_overview.md`](docs/00_overview.md) — 전체 파이프라인 지도
- [`01_sagemaker_inference.md`](docs/01_sagemaker_inference.md) — ⭐ **SageMaker 추론 핵심 가이드** (4옵션·endpoint·서빙 컨테이너)
- [`02_finetuning.md`](docs/02_finetuning.md) — HF DLC + TRL LoRA/QLoRA
- [`03_synthetic_data.md`](docs/03_synthetic_data.md) — grounded 합성 데이터
- [`04_agentic.md`](docs/04_agentic.md) — Strands + AgentCore
- [`05_serving_containers.md`](docs/05_serving_containers.md) — DJL LMI vs vLLM vs TGI

---

## 자주 막히는 곳
- **"어느 노트북부터?"** → `tracks/01_extraction_to_json/00_setup.ipynb`.
- **"GPU에서 그냥 돌려보고 싶다"** → 위 **3번**(dry-run), AWS 불필요.
- **"AWS 없이 코드만 확인"** → 위 **2번**(스모크 테스트).
- **"gated 모델 접근 오류"** → gemma-3은 HF 약관 수락 + `HF_TOKEN` 필요. 또는 ungated `gemma-4-12B-it` 사용.
- **"과금이 무섭다"** → `DRY_RUN=1`로 시작, 끝나면 `99_cleanup.ipynb` 필수.
