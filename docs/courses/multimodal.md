# 멀티모달 추출 코스: 영수증 사진에서 필드를 JSON으로

!!! info "Scope"
    **이미지를 입력으로 받는** 모델을 만들려는 분을 위한
    코스입니다(`tracks/05_multimodal_extraction`).
    "영수증, 전표, 서식 사진을 주면 필드를 뽑아 JSON으로 돌려준다"가 목표라면 이 코스가 맞습니다.

    - **선행 조건**: AWS 자격증명과 Amazon SageMaker AI 실행 role (`00_setup`이 확인).
      SageMaker AI가 처음이면 [SageMaker AI 기초](../01_sagemaker_basics.md)부터
    - **여기서 다루는 것**: task 정의, 시드 데이터셋, 성공 기준, 노트북 구성, 코스별 설정값
    - **여기서 다루지 않는 것**: 학습 방식은 [파인튜닝](../03_finetuning.md),
      배포와 서빙은 [SageMaker AI 추론](../04_sagemaker_inference.md), 컨테이너 메모리 함정은
      [서빙 컨테이너](../05_serving_containers.md)
    - **이 코스에 없는 단계**: 합성 데이터와 agentic 단계. 그 두 단계는 텍스트 코스(01~04)에 있습니다
    - **다른 코스**: 텍스트에서 JSON을 뽑는 문제는 [추출](extraction.md)

이 코스와 관련된 repository 파일입니다(디렉터리 이름 `tracks/`와 `TRACKS`, `track_data.py` 같은 식별자는 역사적 이유로 그대로 둡니다):

- `tracks/05_multimodal_extraction/track_data.py`: cord-v2 로드, `{images, messages}` 어댑터, `INSTRUCTION`
- `tracks/05_multimodal_extraction/scripts/train_mm.py`: 멀티모달 SFT(`AutoModelForImageTextToText` + `AutoProcessor`)
- `tracks/05_multimodal_extraction/samples/`: 배포 검증용 영수증 2장 + 정답 JSON(`README.md`에 선정 근거)
- `tracks/05_multimodal_extraction/*.ipynb`: 이 코스의 노트북 5개
- `common/config.py`: `TRACKS['mm_extraction']` 레지스트리(시드 데이터셋, `max_seq_length=2048`, `num_train_epochs=2`, `multimodal=True`)
- `tracks/05_multimodal_extraction/_build_notebooks.py`: 이 코스 전용 `TrackSpec`과 노트북 빌더(공용 `_shared_build`에서 셀 helper만 재사용)

---

## 이 코스가 푸는 문제

!!! abstract "쉽게 말하면"
    영수증 **사진 한 장**을 넣으면 `{"menu": [{"name", "count", "price"}, ...]}` JSON을 반환하는 모델을 만듭니다. 텍스트 코스와 달리 입력이 문자열이 아니라 이미지입니다.

원본 row는 이미지와 문자열 JSON 두 개입니다(cord-v2 `train` 스플릿).

```text
image:        <영수증 이미지, 예 864x1296 PIL>
ground_truth: {"gt_parse": {"menu": [{"nm":"Nasi Campur Bali","cnt":"1 x","price":"75,000"}, ...],
                            "sub_total": {...}, "total": {...}}}
```

`track_data.to_example()`을 통과한 뒤의 학습 예시는 이렇습니다.

```text
images:   [<PIL.Image>]        # ← messages가 아니라 별도 컬럼
messages[0] role=user       : You are a receipt-parsing engine. Extract the receipt into strict JSON
                              with a 'menu' array of {name, count, price} items. Output ONLY valid
                              JSON, no prose.
messages[1] role=assistant  : {"menu": [{"name": "Nasi Campur Bali", "count": "1 x",
                                         "price": "75,000"}, ...]}      ← 문자열 하나(학습 타깃)
```

읽어 둘 만한 변환 규칙이 셋 있습니다.

- **`gt_parse.menu`만 남깁니다.** `_simplify_gt()`가 `nm`/`cnt`/`price` → `name`/`count`/`price`로 이름을 바꾸고 `sub_total`, `total`은 버립니다("핵심 필드만, 학습 안정").
- **값은 전부 문자열이고 빈 문자열이 흔합니다.** repository에 커밋된 정답을 보면 `{"name": "J.STB PROMO", "count": "", "price": "17500"}`처럼 `cnt`가 없는 항목이 그대로 `count: ""`가 됩니다. 가격도 `"17500"`과 `"13,000"`이 섞여 있어(쉼표 유무) 정규화하지 않습니다. 원본 표기를 그대로 재현하도록 학습합니다.
- **지시문은 system role이 아니라 첫 user 턴 텍스트입니다.** Gemma chat template이 system role을 거부하므로 `INSTRUCTION`을 user 텍스트에 접어 넣습니다([chat template과 system fold](../03_finetuning.md#chat-template과-system-fold)).

??? question "오해: “이미지는 messages content 안에 넣는 거 아닌가요?”"
    추론에서는 그렇습니다(`{"type":"image_url", ...}` + `{"type":"text", ...}`). 하지만 **학습에서는 아닙니다.** TRL의 VLM collator는 이미지를 별도 `images` 컬럼으로 받고 `messages`에는 텍스트만 두며, 이미지 자리표시자를 collator가 직접 주입합니다. `messages` content에 `{"type":"image"}`를 직접 넣으면 이미지 개수와 자리표시자 개수가 어긋나 에러가 납니다(실측 확인). `to_example()`이 이 규약을 지키는 형태를 만들어 줍니다.

---

## 시드 데이터셋

시드는 [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)(CORD: Consolidated Receipt Dataset, **cc-by-4.0**, ungated)입니다. 영수증 이미지와 사람이 만든 구조화 `ground_truth`가 짝지어 있고, 상점명과 주소 같은 개인정보는 **원본에서 이미 마스킹된 상태**입니다.

이미지 태스크에서 permissive license의 라벨 데이터는 드물어 이 코스의 seed 선택 폭은 넓지 않습니다. cc-by-4.0은 출처를 표기하면 재배포할 수 있어 아래 `samples/`를 repository에 포함했습니다.

!!! warning "이미지가 parquet에 내장돼 있어 첫 로드가 느립니다"
    이 코스에서 자주 발생하는 지연입니다. cache가 없으면 **1건을 꺼내는 데 약 40초**가 걸립니다.

    | 로드 방식 | 첫 회 | 재실행 |
    |---|---|---|
    | `load_sample_receipts()` (repository에 커밋된 2장) | **0.03초** | 0.03초 |
    | `load_seed_examples(1)`: split 슬라이스 | ~40초 (전량 준비) | **0.15초** (캐시 히트) |
    | `streaming=True` | 23초 | **24초**: 매번 다시 냅니다 |

    그래서 `load_seed_examples()`는 `streaming=True`를 쓰지 않고 `split="train[offset:offset+n]"` 슬라이스로 받습니다. 노트북은 같은 셀을 여러 번 돌리므로, 첫 회를 감수하고 이후 캐시 히트를 얻는 쪽이 낫다는 판단입니다.

배포 검증용으로는 데이터셋을 아예 건드리지 않습니다. `samples/`에 영수증 **2장**(`receipt_01.jpg` 768×1024, `receipt_02.jpg` 682×1024, 각 메뉴 3항목)과 `ground_truth.json`이 커밋돼 있고 `load_sample_receipts()`가 즉시 로드합니다. 선정 기준이 세 개 다 근거가 있습니다.

- **`test` 스플릿에서 골랐습니다** (`test[1]`, `test[6]`). `train_mm.py`는 `split="train"`만 쓰므로 이 두 장은 모델이 본 적 없습니다. 학습 이미지로 데모하면 정답이 그대로 나와 "잘 된다"고 착각하게 됩니다.
- **항목 수가 적은 것을 골랐습니다.** 생성 토큰 수가 곧 추론 시간입니다(L4 실측 약 40ms/토큰). `train` 첫 영수증은 메뉴 22개와 592토큰이라 추론에 ~24초가 걸립니다.
- **긴 변 1024로 축소하고 JPEG q88로 저장합니다.** payload 크기와 추론 시간은 무관하다는 실측을 바탕으로, 품질을 유지하면서 repository 용량만 줄였습니다.

---

## 성공 기준

이 코스의 목표 지표는 **valid JSON 비율 + 필드 정확도**입니다. 파싱 결과를 사람이 읽는 것이 아니라 다음 시스템이 먹기 때문에, "그럴듯한 문장"은 0점이고 `json.loads()`가 통과하는지가 먼저입니다. 그다음이 `name`/`count`/`price` 필드가 정답과 일치하는지입니다.

!!! warning "이 코스에는 `04_evaluate` 노트북이 없습니다: 검증은 눈으로 대조하는 단계까지입니다"
    `03_deploy_mm_endpoint`가 `samples/`의 영수증을 endpoint에 보내고, `show_image_inference()`로 이미지와 예측 JSON을 나란히 렌더한 뒤 `ground_truth`를 함께 출력해 **육안 대조**하도록 합니다. 정량 지표를 내는 자동 노트북은 이 코스에 포함돼 있지 않습니다.

    직접 붙일 때 알아 둘 것: `common/eval_utils.py`의 `eval_extraction()`은 gold를 `{"name", "arguments"}` 형태로 가정하고 `valid_json_rate`를 그 구조 유효성으로 계산합니다. 즉 함수호출 추출(01 코스) 전용이며 `{"menu": [...]}`에는 그대로 쓸 수 없습니다. held-out 이미지로 채점하려면 `menu` 항목 집합에 대한 F1을 별도로 짜야 합니다. `TrackSpec`의 `eval_kind="extraction"`이 이 코스 spec에도 들어 있지만, 이 값을 읽는 코드(`_c06` = `04_evaluate` 빌더)를 05가 호출하지 않으므로 실제로는 쓰이지 않습니다.

held-out 원칙은 텍스트 코스와 같습니다. 학습에 쓴 이미지로 채점하지 마세요([held-out 규율](../02_synthetic_data.md#held-out-규율-합성으로-평가-금지)). `samples/`가 `test` 스플릿에서 뽑힌 이유도 이것입니다.

---

## 노트북 순서

이 코스의 노트북은 **5개**입니다(텍스트 코스는 9~10개). 공용 빌더(`tracks/_shared_build.py`)에서 셀 helper만 빌려 쓰는 별도 파이프라인입니다.

| 노트북 | 결과 |
|---|---|
| `00_setup` | 자격증명, 리전, role 확인, 의존성 설치. 마지막에 `01_data_explore.ipynb`로 안내합니다 |
| `01_data_explore` | cord-v2 3건 로드 → 이미지 렌더 + 타깃 JSON 확인. **생성물 없음**(탐색 전용) |
| `02_train_mm_sagemaker` | `scripts/train_mm.py`를 SageMaker AI 학습 Job으로 실행 → S3에 **머지된 멀티모달** 모델 |
| `03_deploy_mm_endpoint` | 이미지 입력을 허용하는 real-time endpoint(`gemma-mm-extraction-*`) + `samples/` 영수증 추론 |
| `99_cleanup` | endpoint → endpoint-config → model 삭제 |

없는 노트북과 그 이유가 이 코스의 정체성입니다.

| 텍스트 코스 노트북 | 05에 있나 | 이유 |
|---|---|---|
| `01_data_and_synthetic` | ❌ (`01_data_explore`로 대체) | **이미지 합성은 별개 문제**입니다. Bedrock으로 텍스트를 늘리는 방식이 통하지 않아 시드 라벨을 직접 씁니다 |
| `02a_train_grpo_sagemaker` | ❌ | `grpo_reward_kind`가 비어 있습니다. `train_grpo.py`의 `--reward_kind`는 `extraction`/`classification`만 받고, 이 코스에는 `train_grpo.py` 자체가 없습니다 |
| `02b_local_serve` | ❌ | 코스 spec의 `has_local_serve=False`. 그래서 `99_cleanup`에 '로컬 모델 정리' 섹션도 넣지 않습니다(없는 스크립트를 안내하지 않기 위해) |
| `04_evaluate` | ❌ | [성공 기준](#성공-기준) 참고 |
| `05_agentic_strands`, `06_agentcore_deploy` | ❌ | agentic 단계는 텍스트 코스 전용입니다 |

학습 데이터를 S3에 올리는 채널도 없습니다. `train_mm.py`가 컨테이너 안에서 `load_dataset(seed_dataset, split="train")`으로 이미지를 직접 받으므로, `02` 노트북은 `input_data` 채널 없이 하이퍼파라미터만 넘깁니다.

!!! tip "먼저 DRY_RUN으로 한 바퀴 도세요"
    `train_mm.py --dry_run`은 앞 16건, 1 epoch로 파이프라인 형태만 빠르게 검증합니다. 멀티모달 학습은 텍스트보다 느리므로 `02` 노트북도 `MAX_TRAIN_SAMPLES=200`으로 시작하도록 되어 있습니다. 이 값을 `None`으로 두면 하이퍼파라미터가 전달되지 않고 스크립트 기본값 **500건**이 쓰입니다(무제한이 아닙니다. 더 쓰려면 숫자를 명시하세요). 단계별 핸드오프와 비용 가드는 [실행 runbook](../RUN_E2E.md#멀티모달-코스-05-파이프라인)에 정리돼 있습니다.

---

## 코스별 설정값

### 학습: vision tower 동결 + language LoRA

| 설정 | 이 코스의 값 | 근거 |
|---|---|---|
| `max_seq_length` | 2048 | 정답 JSON이 메뉴 22개에서 592토큰까지 갑니다(실측 100건). 이미지 토큰까지 함께 들어가므로 512, 1024로는 부족합니다 |
| `num_train_epochs` | **2** | 5개 코스 중 유일하게 3이 아닙니다(`TRACKS['mm_extraction']`). 이미지 forward가 비싸 1 epoch 비용이 텍스트 코스보다 큽니다 |
| `multimodal` | `True` | "이 코스는 이미지 입력"이라는 **선언용 메타데이터**입니다(분기하는 코드는 없습니다: 아래 참고) |
| LoRA target | `.*language_model\..*\.(q_proj\|k_proj\|v_proj\|o_proj\|gate_proj\|up_proj\|down_proj)$` | `all-linear`나 이름 리스트를 주면 vision/audio proj(`Gemma4ClippableLinear`)까지 매칭돼 `get_peft_model`이 크래시합니다: 상세는 [LoRA target](../03_finetuning.md#lora-target-멀티모달은-language_model만) |
| `freeze_vision` | `True` (기본) | vision/audio 파라미터를 `requires_grad=False`로 두고 language LoRA만 학습합니다 |
| `use_qlora`, `merge_adapter` | 둘 다 `True` | 4bit nf4로 학습하고, 머지는 base(bf16)를 CPU에 다시 올려 수행합니다. 멀티모달 full 모델은 vision+audio를 포함해 특히 커서 호스트 RAM 여유가 필요합니다 |

`scripts/requirements.txt`에 텍스트 코스에 없는 한 줄이 있습니다: `torchvision>=0.20.0`(Gemma4 image processor 의존). 이것이 빠지면 processor 로드부터 실패합니다.

!!! warning "`multimodal=True`를 읽고 분기하는 코드는 없습니다"
    이 필드는 레지스트리를 볼 때 코스 성격을 알려 주는 표식이고, 파이프라인 동작을 바꾸지는 않습니다. 실제 분기는 **코드가 놓인 위치**가 만듭니다. 합성 단계가 없는 것은 `_build_notebooks.py`의 빌더 목록에 `01_data_and_synthetic`이 아예 없기 때문이고(`build_00/01/02/03/99`), processor 경로는 `train_mm.py`에 `AutoProcessor.from_pretrained`가 그대로 적혀 있기 때문입니다.
    repository 전체에서 이 필드를 읽는 곳은 `02_train_mm_sagemaker`의 확인용 `print` 한 줄과 `tests/test_smoke.py`의 assert뿐입니다. 같은 이유로 `eval_kind="extraction"`도 이 코스에서는 쓰이지 않습니다([성공 기준](#성공-기준) 참고). 값을 바꿔도 노트북 세트는 그대로이므로, 새 코스를 만들 때 이 플래그만 켜고 builder를 그대로 두면 동작이 달라지지 않습니다.

### 서빙: 멀티모달 그대로

**이 코스는 텍스트 re-export를 하지 않습니다.** 텍스트 코스는 merge 후 `language_model` submodule만 `*ForCausalLM`으로 다시 저장하지만(생략하면 serving container가 image processor를 찾다가 종료됩니다: [텍스트 전용 re-export](../03_finetuning.md#텍스트-전용-re-export와-kv-shared-복원)), 여기서는 vision tower를 **유지한 채** 저장해 vLLM이 이미지 입력을 받도록 합니다. 저장 직전 KV-shared dead weight 복원은 텍스트 코스와 동일하며, key prefix만 `model.language_model.*`로 다릅니다.

배포 셀의 값과 근거입니다.

| 값 | 설정 | 근거 |
|---|---|---|
| `mm_limit={"image": 1}` | 이미지 입력 허용 | 텍스트 코스의 배포 셀은 이 인자를 주지 않습니다(re-export로 이미 텍스트 전용이라 불필요: re-export하지 않은 멀티모달 artifact를 텍스트로만 쓸 때 `{"image": 0, "audio": 0}`으로 막는 주석이 남아 있습니다). 여기서 `image=0`을 쓰면 이미지가 거부됩니다 |
| `max_model_len=2048` | 서빙 컨텍스트 | 이 코스는 학습 길이와 같은 값을 씁니다(공용 빌더의 `serve_max_model_len` 경로를 타지 않고 노트북이 직접 지정). 입력이 짧은 지시문 + 이미지라 요약, QA 코스처럼 프롬프트가 컨텍스트를 잡아먹지 않습니다([학습 길이와 서빙 길이](../00_overview.md#학습-길이와-서빙-길이는-다른-값입니다)) |
| `max_tokens=768` | 생성 상한 | 공용 기본값 256이 아닙니다. 정답 JSON 최대 **592토큰**(실측 100건)이라 512로는 잘립니다. L4에서 768 생성에 **21.3초** 실측: `/invocations` 60초 한도의 3분의 1입니다 |
| `max_num_seqs=32`, `gpu_memory_utilization=0.90` | 24GB GPU OOM 회피 | 아래 참고 |

!!! danger "기본값으로 배포하면 24GB GPU에서 endpoint가 Failed합니다"
    멀티모달 artifact는 vision tower를 포함해 가중치가 **15.18 GiB**입니다(텍스트 코스 14.23 GiB). `ml.g6.2xlarge`(L4 22.9GB) 예산 20.21 GiB에서 vLLM이 KV를 4.69 GiB로 과대 배정하면 여유가 0.34 GiB뿐이고, 실제로 더 필요한 양이 1.12 GiB라 **0.78 GiB 부족**으로 CUDA OOM이 납니다. 증상은 `did not pass the ping health check` 한 줄뿐이고 진짜 원인은 CloudWatch 로그 안에 있습니다.
    원인은 모델 크기가 아니라 `max_num_seqs`의 vLLM 기본값 256입니다. sampler logits buffer가 `256 × vocab 262,144 × 4B = 정확히 256 MiB`입니다. GPU를 바꿀 필요는 없습니다. 전체 예산 표와 L40S 재현 실측은 [메모리 예산](../05_serving_containers.md#메모리-예산-l4-229gb-실측)에 있습니다.

호출 스키마는 텍스트 코스와 같은 OpenAI 호환 chat이고, 이미지만 base64 data URL로 실어 보냅니다. `03` 노트북은 PNG 대신 **JPEG로 인코딩**합니다(payload가 1/8, 추론 시간은 동일). real-time endpoint의 요청 payload 한도가 6 MB라 이미지를 여러 장 묶으면 실제로 닿을 수 있는 벽입니다([SageMaker AI 추론](../04_sagemaker_inference.md)).

---

## 이어서 볼 문서

- [00 전체 지도](../00_overview.md#5개-독립-코스와-공통-레이어): 5개 코스 비교표와 이 코스의 위치(단계 도해는 [멀티모달 코스 05의 별도 파이프라인](../00_overview.md#멀티모달-코스-05의-별도-파이프라인))
- [03 파인튜닝](../03_finetuning.md): LoRA/QLoRA, Gemma 관용구, 머지와 KV-shared 복원
- [04 SageMaker AI 추론](../04_sagemaker_inference.md): endpoint 3층 구조, 호출 스키마, payload, timeout 한도
- [05 서빙 컨테이너](../05_serving_containers.md): 엔진 선택, OOM, 절단 실측 함정
- [실행 runbook](../RUN_E2E.md#멀티모달-코스-05-파이프라인): 단계별 실행 순서, 비용 가드, 완료 기준

!!! danger "비용과 cleanup"
    학습 Job은 실행 시간만큼 과금되고 **endpoint는 호출하지 않아도 삭제할 때까지 시간당 계속 과금**됩니다. 코스를 마쳤으면 `99_cleanup`을 반드시 실행해 endpoint, endpoint-config, model을 모두 지우세요.
    이 코스는 이미지 forward가 비싸 1 epoch 비용이 텍스트 코스보다 크므로 `MAX_TRAIN_SAMPLES`를 낮춰 시작하세요. 합성과 agentic 단계가 없어 Bedrock 과금은 발생하지 않습니다.
