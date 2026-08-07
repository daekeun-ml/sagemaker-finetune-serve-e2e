# 분류 코스: 은행 고객 문의를 intent 라벨 하나로

!!! info "Scope"
    고객 문의, 티켓, 로그처럼 **입력 하나에 라벨 하나**를 붙이는 일을 SLM 파인튜닝으로
    해결하려는 분을 위한 코스입니다(`tracks/02_classification`).

    - **산출물**: `mteb/banking77`의 77개 intent를 텍스트로 생성하는 Gemma 4 LoRA 모델,
      그것을 서빙하는 real-time endpoint, held-out macro-F1
    - **선행 조건**: AWS 자격증명과 Amazon SageMaker AI 실행 role (`00_setup`이 확인).
      SageMaker AI가 처음이면 [SageMaker AI 기초](../01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: task 정의, 시드 데이터셋, 성공 기준, 노트북 구성, 코스별 설정값
    - **여기서 다루지 않는 것**: 학습 방식은 [파인튜닝](../03_finetuning.md),
      배포와 서빙은 [SageMaker AI 추론](../04_sagemaker_inference.md), 완주 절차는
      [실행 runbook](../RUN_E2E.md)
    - **다른 코스**: 스키마가 있는 JSON은 [추출](extraction.md), 자유서술은 [도메인 QA](domain_qa.md)

이 코스와 관련된 리포지토리 파일입니다(디렉터리 이름 `tracks/`와 `TRACKS`, `track_data.py` 같은 식별자는 역사적 이유로 그대로 둡니다):

- `tracks/02_classification/track_data.py`: 시드 로드와 셔플, `{input, output}` 어댑터, `SYSTEM_PROMPT`, 라벨 목록 조회
- `tracks/02_classification/scripts/train.py`, `train_grpo.py`: SFT / GRPO 학습(로컬 dry-run ↔ SageMaker AI 겸용)
- `tracks/02_classification/*.ipynb`: 이 코스의 노트북 10개
- `common/config.py`: `TRACKS['classification']` 레지스트리(시드 데이터셋, `max_seq_length=512`)
- `common/eval_utils.py`: `normalize_label()` + `eval_classification()`(accuracy, macro-F1, weighted-F1)
- `tracks/build_all_tracks.py`: 이 코스의 `TrackSpec`(엔드포인트 prefix, 서빙과 생성 길이, GRPO reward 종류)

---

## 이 코스가 푸는 문제

!!! abstract "쉽게 말하면"
    "카드가 아직 안 왔어요" 같은 은행 고객 메시지를 읽고, **미리 정해진 77개 intent 중 정확히 하나**를 snake_case 문자열로 출력하는 모델을 만듭니다.
    분류 헤드(softmax 레이어)를 붙이는 게 아니라, **라벨 이름을 텍스트로 생성**하게 학습합니다.

`track_data.py`의 어댑터가 원본 row를 학습 형태로 바꾸는 과정입니다. 아래 값은 `mteb/banking77` train 스플릿의 첫 행을 실제로 읽은 것입니다.

```text
# 원본 row (mteb/banking77)
text:       I am still waiting on my card?
label:      11
label_text: card_arrival
```

`load_seed_examples()`가 `text` → `input`, `label_text` → `output`으로 옮기고, `to_messages()`가 `SYSTEM_PROMPT`를 첫 user 턴 앞에 붙여 학습 JSONL 한 줄을 만듭니다.

```json
{"messages": [
  {"role": "user", "content": "You are an intent classifier for banking customer messages. Output ONLY the single most appropriate intent label (snake_case), nothing else.\n\nI am still waiting on my card?"},
  {"role": "assistant", "content": "card_arrival"}
]}
```

정답은 라벨 문자열 한 줄뿐입니다. `data/train.jsonl` 500건을 Gemma 4 E4B 토크나이저로 재면 **정답이 median 5토큰 / max 16토큰**입니다. 이 코스의 모든 길이 설정값이 다른 코스보다 작은 이유가 여기 있습니다.

??? question "오해: “분류인데 왜 생성 모델을 쓰나요?”"
    이 kit은 JumpStart의 분류 전용 헤드를 쓰지 않고 DLC + 커스텀 `train.py`(TRL `SFTTrainer`)로 학습합니다([왜 커스텀 train.py 경로인가](../03_finetuning.md#왜-커스텀-trainpy-경로인가)). 그래서 라벨을 **텍스트로 생성**시키고, 평가 시점에 `common/eval_utils.normalize_label`이 자유 텍스트 출력을 닫힌 라벨셋에 다시 매핑합니다(정확 일치 → substring → rapidfuzz 유사도 순).
    이 방식의 이점은 같은 파이프라인과 같은 서빙 컨테이너를 다른 네 코스와 그대로 공유한다는 점입니다. 대신 모델이 라벨셋 밖의 문자열을 낼 수 있으므로 정규화 단계가 필수입니다.

---

## 시드 데이터셋

`mteb/banking77`(mit, ungated, parquet)을 씁니다. BANKING77의 parquet 미러로, train 9,993건 / test 3,076건이며 test 스플릿에 77개 라벨이 모두 등장합니다. 컬럼은 `text`(고객 메시지) + `label`(정수) + `label_text`(intent 이름)입니다.

`track_data._CANDIDATES`는 `mteb/banking77` → `gtfintechlab/banking77`(cc-by-4.0) 순으로 시도하므로, 첫 미러가 사라져도 다음으로 넘어갑니다. 두 스키마가 달라(`label_text` vs `ClassLabel`) `_label_str()`이 양쪽을 모두 처리합니다.

!!! warning "원본 PolyAI/banking77은 로드되지 않습니다"
    원본 리포는 **스크립트 기반**(`banking77.py`)이라 이 kit이 핀한 `datasets>=5.0.0`에서 `RuntimeError: Dataset scripts are no longer supported`로 로드 자체가 실패합니다. parquet 자동변환본(`refs/convert/parquet`)도 없어 되살릴 방법이 없습니다.
    `legacy-datasets/banking77`은 카드에 "deprecated and will be deleted"가 명시돼 있어 쓰지 않고, `gtfintechlab/banking77`은 동작하지만 다운로드 수가 적어 폴백으로만 둡니다.
    그래서 `04_evaluate`도 `load_dataset('PolyAI/banking77').features['label'].names`를 직접 부르지 않고 `track_data.load_label_names()`를 씁니다. 미러가 또 바뀌어도 고칠 곳이 한 군데입니다.

!!! danger "셔플 없이 앞에서부터 뽑으면 평가가 무너집니다"
    banking77의 train 스플릿은 **라벨 정렬 순서**입니다. 셔플을 빼고 앞에서부터 300건을 그냥 뽑으면 학습 구간에 클래스가 3개뿐이고(실측: `card_arrival` 153 / `card_linking` 139 / `exchange_rate` 8), `04_evaluate`가 쓰는 held-out 구간인 행 300~349는 **50건 전부 `exchange_rate` 단일 라벨**이 됩니다. 77클래스 macro-F1이 아무 의미도 갖지 못하는 상태입니다.
    그래서 `load_seed_examples()`는 `ds.shuffle(seed=42)`를 먼저 적용합니다. 셔플 후 실측: 앞 300건에 **73개 클래스**, held-out으로 쓰는 300~349번 50건에 **35개 클래스**가 들어옵니다. 시드를 42로 고정하므로 학습과 평가가 같은 인덱스를 부르면 항상 같은 결과가 나옵니다(재현성).

합성 데이터 단계에서는 `seed_texts_for_synth()`가 시드를 `MESSAGE: ...\nINTENT: ...` 형태로 직렬화해 Bedrock에 grounding으로 넘깁니다. 평가용 held-out에는 합성이 한 건도 섞이지 않습니다([held-out 규율](../02_synthetic_data.md#held-out-규율-합성으로-평가-금지)).

---

## 성공 기준

주 지표는 **macro-F1**이고 accuracy와 weighted-F1을 함께 출력합니다(`eval_utils.eval_classification`).

| 지표 | 계산 | 왜 보는가 |
|---|---|---|
| `macro_f1` (primary) | `f1_score(..., labels=label_set, average='macro')` | 77클래스가 균등하지 않으므로, 소수 클래스를 통째로 놓쳐도 점수가 떨어지게 합니다 |
| `accuracy` | 정답 라벨과 정확 일치한 비율 | 사람이 바로 이해하는 기준선. 다수 클래스에 쏠리면 과대평가됩니다 |
| `weighted_f1` | 클래스 빈도로 가중한 F1 | 실제 트래픽 분포에 가까운 체감 성능 |

accuracy만 보면 안 되는 이유가 이 코스에서 특히 분명합니다. held-out 50건에 35개 클래스가 들어오므로 대부분의 클래스는 1~2건뿐이고, 그 클래스를 전부 틀려도 accuracy는 조금만 떨어집니다. macro-F1은 클래스별 F1을 단순 평균하므로 같은 실수를 크게 반영합니다.

!!! warning "기본 설정에서 macro_f1의 상한은 1.0이 아니라 약 0.45입니다"
    `eval_classification()`은 `labels=label_set`, 즉 `td.load_label_names()`가 돌려주는 **77개 라벨 전체**를 평균 대상으로 넘깁니다. 그런데 held-out 50건에는 35개 클래스만 등장하므로, 나머지 42개 라벨은 정답도 예측도 없어 F1이 무조건 0으로 계산되고 그대로 평균에 들어갑니다.
    예측을 정답과 **완전히 일치**시켜도 `macro_f1 = 0.4545`(= 35/77)이고, 같은 조건의 `weighted_f1`은 `1.0`입니다.

    그래서 `accuracy`, `weighted_f1`의 숫자는 문자 그대로 읽어도 되지만, `macro_f1`은 **같은 held-out에서의 다른 실행값과 비교하는 상대 지표**로만 쓰세요. 절대값 0.3을 보고 "형편없다"고 판단하면 오진입니다. 0.45가 그 판의 만점입니다. 절대값을 그대로 읽고 싶다면 `labels`를 held-out에 등장한 클래스로 좁히면 되고(같은 완전 일치 조건에서 `1.0`), 그때는 실행 간 라벨 집합이 달라져 비교 가능성이 떨어진다는 대가를 치릅니다. `N_EVAL`을 키워 등장 클래스 수를 늘리는 것도 같은 방향의 완화책입니다.

모델은 자유 텍스트로 답하므로 채점 전에 `normalize_label()`이 예측을 닫힌 라벨셋에 매핑합니다: 소문자와 공백→`_` 정규화 후 **정확 일치 → substring 포함 → rapidfuzz 유사도** 순이고, 전부 실패하면 `label_set[0]`으로 폴백합니다(즉 오답으로 계산됩니다). `04_evaluate`는 `temperature=0.0`으로 호출해 재현성을 확보하고, held-out은 `NUM_SEED_SAMPLES`(300)건을 **명시적으로 건너뛴 뒤** `N_EVAL`건(기본 50, `DRY_RUN`이면 20)을 씁니다.

---

## 노트북 순서

이 코스의 노트북은 **10개**입니다. `02a`(GRPO)와 `02b`(로컬 서빙)를 모두 갖는 두 코스 중 하나입니다(다른 하나는 [추출 코스](extraction.md), 요약과 도메인 QA는 `02a`가 없어 9개입니다).

| 노트북 | 산출물 |
|---|---|
| `00_setup` | 리전, role, bucket 확인 후 `%store` 저장, 의존성 설치 |
| `01_data_and_synthetic` | 셔플된 시드 300건 + Bedrock grounded 합성을 합친 `data/train.jsonl` |
| `02_train_sft_sagemaker` | SFT LoRA 학습 Job → 머지된 모델 아티팩트(S3), `%store md_classification` |
| `02a_train_grpo_sagemaker` | **(선택)** SFT 산출물을 base로 GRPO 정련 → 새 아티팩트 |
| `02b_local_serve` | **(선택)** 로컬 GPU `vllm serve`로 배포 전 검증 + `vllm bench` 측정값 |
| `03_deploy_endpoint` | `gemma-classification-vllm-<timestamp>` real-time endpoint + invoke 스모크 |
| `04_evaluate` | held-out accuracy, macro-F1, weighted-F1 |
| `05_agentic_strands` | `classify_intent` tool을 가진 Strands 에이전트(reasoning은 Bedrock Claude) |
| `06_agentcore_deploy` | AgentCore Runtime 배포 절차 |
| `99_cleanup` | endpoint → endpoint-config → model 삭제, 로컬 모델 정리 |

`02a`가 있는 이유는 **reward를 프로그램으로 채점할 수 있기 때문**입니다. `scripts/train_grpo.py`의 `reward_classification`은 예측 라벨이 정답과 정확 일치하면 `1.0`, 라벨이 텍스트 안에 포함되기만 하면 `0.3`(형식 어긋남), 그 외는 `0.0`을 줍니다. 요약과 도메인 QA에는 이런 채점 함수가 없어 GRPO 노트북이 아예 생성되지 않습니다. 판단 근거는 [왜 추출과 분류 코스에만 GRPO가 있나](../03_finetuning.md#왜-추출과-분류-코스에만-grpo가-있나).

GRPO의 prompt 소스로 `failures`(=`04_evaluate`에서 틀린 건)를 고르면 `common/grpo_data.py`가 `pred.lower() != gold.lower()`로 실패를 판정해 그 prompt만 모읍니다. 기본값 `synth`를 쓸 때는 이 코스 전용 난이도 제약이 생성 프롬프트에만 붙습니다("두 유사 intent의 경계에 놓인 메시지, 간접과 감정적 표현, 희소 intent 위주"). 배경은 [SFT에서 GRPO로](../03_finetuning.md#sft에서-grpo로-데이터를-갈아야-하는-이유)에 있습니다.

---

## 코스별 설정값

다른 코스와 다른 값만 모았습니다. 값의 출처는 `common/config.py`의 `TRACKS`와 `tracks/build_all_tracks.py`의 `TrackSpec`입니다.

| 설정 | 이 코스 | 요약 코스(비교) | 근거 |
|---|---|---|---|
| `max_seq_length` | **512** | 2048 | 학습 전체(입력+정답)가 실측 median 58 / p90 69 / max 110 토큰: 500건 중 512를 넘는 건이 0건입니다 |
| `serve_max_model_len` | **미지정 → 1024** | 4096 | spec에 값이 없어 `max_seq_length × 2`가 됩니다([학습 길이와 서빙 길이](../00_overview.md#학습-길이와-서빙-길이는-다른-값입니다)). 프롬프트 max 103토큰 + 생성 256토큰이므로 여유가 큽니다 |
| `gen_max_tokens` | **256**(기본값 그대로) | 512 | 정답 라벨이 max 16토큰이라 절단 위험이 없습니다. 요약 코스는 256이면 held-out 40%가 잘려 512로 올렸습니다 |
| `grpo_reward_kind` | **`classification`** | (없음) | 라벨 정확 일치로 채점 가능 → `02a` 노트북이 생성됩니다 |
| `eval_kind` | `classification` | `summarization` | `04_evaluate`가 `eval_classification`(macro-F1)을 부르고, 실시간 추론 셀의 **스트리밍이 기본 off**가 됩니다 |
| `endpoint_prefix` | `gemma-classification` | `gemma-summarization` | 학습 Job, endpoint 이름과 `%store` 키(`ep_classification`, `md_classification`)의 접두어 |
| `multimodal` | `False` | `False` | 텍스트 전용이라는 표식입니다(레지스트리 기본값). 노트북 세트를 정하는 것은 이 값이 아니라 빌더이며, 이 코스는 `01_data_and_synthetic`을 씁니다 |

스트리밍이 기본 off인 이유는 이 코스의 응답이 **라벨 한 줄**이라 완성돼야 파싱과 라우팅에 쓸 수 있기 때문입니다. 스트리밍은 첫 토큰 체감만 줄이고 전체 생성 시간이나 처리량은 바꾸지 않습니다([스트리밍이 개선하지 않는 것](../05_serving_containers.md#스트리밍이-개선하지-않는-것)).

!!! tip "짧은 시퀀스가 이 코스를 가장 저렴하게 만듭니다"
    `02_train_sft_sagemaker`는 step 시간을 시퀀스 길이로 추정하는데, ml.g6.2xlarge 실측이 **seq 512에서 약 7초/step, seq 2048에서 약 17초/step**입니다. 핸즈온 기본값(`MAX_TRAIN_SAMPLES=200`, `EPOCHS=2` → 약 50 step)이면 학습이 10분 안쪽입니다.
    다섯 코스 중 하나만 완주해 볼 생각이라면 이 코스가 가장 빠르고, GRPO까지 곁들여 볼 수 있는 코스이기도 합니다.

!!! warning "서빙 파라미터는 GPU를 바꾸면 함께 조정하세요"
    `max_num_seqs=32` / `gpu_memory_utilization=0.90`은 24GB GPU에서 CUDA OOM을 피하려고 낮춰 둔 값입니다. 더 큰 GPU로 옮기면 함께 올리세요([24GB GPU CUDA OOM](../05_serving_containers.md#24gb-gpu-cuda-oom-max_num_seqs-기본값)).

---

## 이어서 볼 문서

- [00 전체 지도](../00_overview.md#5개-독립-코스와-공통-레이어): 5개 코스 비교와 공통 `common/` 레이어
- [02 합성 데이터](../02_synthetic_data.md): grounded 합성과 held-out 규율
- [03 파인튜닝](../03_finetuning.md#lora-vs-qlora와-인스턴스-사이징): LoRA/QLoRA, Gemma 관용구, 인스턴스 선택
- [04 SageMaker AI 추론](../04_sagemaker_inference.md#endpoint-3층-구조와-호출): endpoint 3층 구조와 호출 스키마
- [05 서빙 컨테이너](../05_serving_containers.md): 엔진 선택, OOM, 절단 실측 함정
- [실행 runbook](../RUN_E2E.md#단계별-실행과-데이터-핸드오프): 단계별 실행 순서, 비용 가드, 완료 기준

!!! danger "비용과 cleanup"
    학습 Job은 실행 시간만큼 과금되고 **endpoint는 삭제할 때까지 시간당 계속 과금**됩니다. 코스를 마쳤으면 `99_cleanup`을 반드시 실행해 endpoint, endpoint-config, model을 모두 지우세요.
    `02a`의 GRPO는 prompt당 rollout을 여러 개 생성하므로 SFT보다 오래 걸립니다(노트북 기본 `MAX_RUNTIME_HOURS=6`). 합성 데이터 생성과 `05_agentic_strands`는 Bedrock 호출로 별도 과금됩니다.
