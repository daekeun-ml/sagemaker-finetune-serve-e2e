# Gemma E2E Fine-tuning Toolkit (SageMaker)

문제 정의 → 오픈 데이터/모델 → (grounded) 합성 데이터 → **SageMaker 학습** → **real-time endpoint** →
**agentic loop (SLM + Bedrock Claude, Strands → AgentCore)** 까지, 초심자도 step-by-step으로
따라할 수 있는 **production 지향 실용 에셋 킷**.

> 이 킷은 인터뷰(문제·모델·데이터·학습경로·배포·agentic)로 스펙을 확정한 뒤 생성됐습니다.

> 🚀 **처음이라면 → [`GETTING_STARTED.md`](GETTING_STARTED.md)** (설치 → 스모크 → 로컬 dry-run → 노트북).
> 🔁 **전체를 처음부터 끝까지 돌릴 거라면 → [`docs/RUN_E2E.md`](docs/RUN_E2E.md)** (E2E 실행 런북).

---

## TL;DR
- **모델**: `google/gemma-3-4b-it` 기본 (텍스트 SLM 스윗스팟, 최다 문서화). `MODEL_ID` 만 바꾸면
  `google/gemma-4-12B-it`(apache-2.0·**ungated**)로 승급. 🔴 `gemma-4-31B`는 (요청대로) 제외.
- **학습 경로**: HuggingFace DLC + TRL `SFTTrainer` + PEFT LoRA/QLoRA (`scripts/train.py`는
  로컬 GPU `--dry_run` 과 SageMaker `.fit()` 겸용, **self-contained**).
- **배포**: SageMaker real-time endpoint (serverless는 GPU가 없어 SLM 부적합).
- **Agentic**: Strands Agent (reasoning=Bedrock Claude, tool=`call_slm`→endpoint) → AgentCore Runtime.
- **데이터**: 스모크 시드(공개 permissive 데이터셋) + **grounded 합성 500건/트랙** (Bedrock Converse + critique/refine).
- **개발환경 GPU**: 모든 학습/합성에 `DRY_RUN=1` 로 빠른 파이프라인 검증 후 실제 실행.

## 5개 트랙 (독립 E2E)
| 트랙 | task | 시드 데이터셋 (라이선스) |
|---|---|---|
| `01_extraction_to_json` | 텍스트 → 구조화 JSON 추출 (**플래그십**) | `glaiveai/glaive-function-calling-v2` (apache-2.0) |
| `02_classification` | 텍스트 분류 (intent) | `mteb/banking77` (mit) |
| `03_summarization` | 문서 요약 | `FiscalNote/billsum` (cc0-1.0) |
| `04_domain_qa` | 도메인 QA / instruction | `databricks/databricks-dolly-15k` (cc-by-sa-3.0) |
| `05_multimodal_extraction` | **이미지 → 구조화 JSON** (영수증, gemma-4 vision) | `naver-clova-ix/cord-v2` (cc-by-4.0) |

텍스트 트랙(01~04) 노트북 순서: `00_setup` → `01_data_and_synthetic` → `02_train_sft_sagemaker` →
`03_deploy_endpoint` → `04_evaluate` → `05_agentic_strands` → `06_agentcore_deploy` → `99_cleanup`.
- 추출·분류 트랙은 **(선택) `02a_train_grpo_sagemaker`** (SFT→GRPO 정련)와 **(선택) `02b_local_serve`**(배포 전 로컬 vLLM 검증)를 추가로 포함합니다.
- 모든 텍스트 트랙에 **(선택) `04b_sagemaker_eval`**(SageMaker managed 평가) 노트북이 함께 생성됩니다.

멀티모달 트랙(05)은 이미지 입력이라 구조가 다릅니다: `00_setup` → `01_data_explore` → `02_train_mm_sagemaker` →
`03_deploy_mm_endpoint` → `99_cleanup` (합성 데이터 단계 없음, 서빙은 이미지 입력을 받는 멀티모달 endpoint).

**평가 — 두 가지 방식**
- **`04_evaluate`** (기본, 로컬·빠름·저렴): held-out 세트로 endpoint를 직접 호출해 성공기준 수치화 (🔴 합성/학습셋 아님). 트랙별 메트릭:
  추출=arg_f1+valid_json_rate / 분류=macro-F1+accuracy / 요약=ROUGE-L+LLM-judge / QA=LLM-judge+ROUGE-L.
- **`04b_sagemaker_eval`** (선택, SageMaker managed 평가 잡): SDK v3 `sagemaker.train.evaluate`의 evaluator 3종을 트랙 성격에 맞게 조합. 🔴 별도 컴퓨트·비용 발생.
  | evaluator | 용도 | 적용 트랙 |
  |---|---|---|
  | `BenchMarkEvaluator` | 표준 벤치(ifeval/mmlu) 일반능력 + baseline 비교 | 전 트랙 공통 |
  | `CustomScorerEvaluator` | 프로그램적 채점 (arg_f1·라벨) | 추출·분류 |
  | `LLMAsJudgeEvaluator` | judge LLM 주관 평가 (Correctness/Helpfulness/Faithfulness) | 요약·QA |

## 디렉토리
```
sagemaker-finetune-serve-e2e/
├── common/                       # 얇은 공통 레이어 (중복 제거, 노트북이 import)
│   ├── config.py                 # MODEL_ID/region/role/bucket 플레이스홀더+env, DRY_RUN
│   ├── gemma_format.py           # 표준 messages 어댑터 (apply_chat_template에 위임)
│   ├── aws_utils.py              # invoke_endpoint · bedrock converse · CloudWatch 링크 · 비용가드
│   ├── llm_gateway.py            # (LiteLLM) Bedrock+SageMaker 단일 인터페이스   [litellm-recon 검증 후]
│   └── synth/
│       ├── bedrock_synth.py      # grounded 합성 기본 경로 (boto3 Converse + critique/refine, 무의존성)
│       └── README.md             # 오픈 라이브러리 대안 (활발히 유지보수되는 툴킷)  [synth-toolkit-recon 검증 후]
├── tracks/01_extraction_to_json/          # (텍스트 트랙 대표, 02~04도 동일 구조)
│   ├── track_data.py             # 시드 로드 + messages 어댑터 (트랙별)
│   ├── *.ipynb                   # step-by-step 튜토리얼 노트북
│   └── scripts/
│       ├── train.py              # SFT (self-contained, 로컬 dry-run + SageMaker 겸용)
│       ├── train_grpo.py         # (추출·분류) SFT→GRPO 정련
│       └── serve_local_vllm.sh   # 로컬 vLLM 프리플라이트
├── tracks/05_multimodal_extraction/       # 🖼️ 이미지→JSON (gemma-4 vision)
│   ├── track_data.py             # cord-v2 로더 (images+messages 어댑터)
│   ├── samples/                  # 배포 검증용 영수증 2장 + 정답 (held-out, 즉시 로드)
│   └── scripts/train_mm.py       # 멀티모달 SFT (processor, vision freeze + language LoRA)
├── tools/
│   └── clear_outputs.py          # 노트북 셀 출력 정리 (커밋 전 / --check 로 검사만)
└── agentcore/                    # ARM64 컨테이너 스캐폴드 (Strands → AgentCore Runtime)
```

🔴 **노트북은 빌더가 생성합니다** — `tracks/*.ipynb`를 직접 고치지 말고 `tracks/_shared_build.py`(공통 셀)와
`tracks/build_all_tracks.py`(트랙별 spec)를 고친 뒤 재생성하세요. 직접 고치면 다음 빌드에서 덮어써집니다.

```bash
python tracks/build_all_tracks.py                        # 02·03·04 트랙
python tracks/01_extraction_to_json/_build_notebooks.py  # 01 (spec + 06 전용 절)
python tracks/05_multimodal_extraction/_build_notebooks.py
python tools/clear_outputs.py                            # 테스트로 남은 셀 출력 정리
```

## 사전 준비

### 1) 의존성 설치 — uv 권장
```bash
# uv 미설치 시
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/sagemaker-finetune-serve-e2e
uv venv && source .venv/bin/activate     # (선택) 전용 가상환경
uv pip install -r pyproject.toml         # 의존성 설치
# 또는 잠금 기반 재현 설치:  uv sync

# 최신 버전으로 업그레이드 (transformers 등)
uv lock --upgrade                        # 전체 최신
uv lock --upgrade-package transformers   # 특정 패키지만
```
> pip만 쓰려면: `pip install -r requirements.txt` (동일 floor 핀).
> 버전은 실측 최신을 `>=` floor로 고정(2026-07: transformers 5.14.1 / trl 1.8.0 / peft 0.19.1 / datasets 5.0.0).
> ⚠️ **로컬 transformers ≠ SageMaker DLC transformers**: DLC 이미지 태그는 `common/config.py`의 `HF_*_VERSION`(AWS 게시 조합), 컨테이너 내 업그레이드는 `tracks/*/scripts/requirements.txt`가 담당.

### 2) 환경변수 (플레이스홀더 — 시크릿 하드코딩 금지)
```bash
# gated 모델(gemma-3/2/3n) 사용 시: HF에서 모델 약관 수락 후 토큰 발급
export HF_TOKEN=hf_xxx            # gemma-4 계열이면 불필요
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<SageMakerRole>
export BEDROCK_CLAUDE_MODEL_ID=us.anthropic.claude-...   # 🔴 모델 상세 페이지에서 정확 ID 확인
export DRY_RUN=1                  # 먼저 파이프라인 검증, 실제 실행 시 0
```

## 🔴 비용 & cleanup (필수)
- real-time endpoint 는 삭제 전까지 시간당 과금 → 실습 후 **반드시** `99_cleanup.ipynb` 실행.
- Bedrock Converse 는 토큰 과금 (합성 대량 생성 시 비용). AgentCore 배포 시 Runtime 과금.
- 각 노트북은 학습/배포 직후 CloudWatch 다이렉트 링크를 출력.

## ⚠️ 라이선스
- Gemma 3/2/3n = 커스텀 **Gemma Terms** + gated (HF 토큰·약관 수락, 서빙 시 use-restriction 전파 의무).
- Gemma 4 = **apache-2.0 + ungated** (마찰 최소).
- 시드 데이터셋은 전부 permissive만 선별 (NC/ND/미선언 데이터셋은 배제). share-alike(dolly/squad) 파생물 주의.

## 근거 (라이브 검증 2026-07)
Gemma 로스터·JumpStart·HF DLC·Bedrock Claude·Strands/AgentCore·데이터셋 라이선스는
플러그인의 `verified-facts-2026-07.md` + 정찰 워크플로우로 실측·적대적 검증됨.
빠르게 바뀌는 항목(모델 ID·SDK 버전·리전)은 실행 직전 재확인 (`# TODO verify` 주석).
