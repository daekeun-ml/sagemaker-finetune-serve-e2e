# samples/: 배포 검증용 영수증 이미지 2장

`03_deploy_mm_endpoint.ipynb`가 endpoint 스모크 테스트에 쓰는 이미지입니다. `track_data.load_sample_receipts()`로 로드합니다.

## repository에 포함한 이유

시드 데이터셋(cord-v2)은 이미지가 parquet에 내장돼 있어 **캐시가 없으면 1건 로드에 약 40초**가 걸립니다(실측 2026-07-31). 배포 검증은 이미지 1~2장이면 충분하므로 미리 저장해 두고 즉시 씁니다.

| 방식 | 로드 시간 |
|---|---|
| `load_sample_receipts()` (이 폴더) | **0.03초** |
| `load_seed_examples(1)`, 캐시 없음 | ~40초 |
| `load_seed_examples(1)`, 캐시 있음 | ~1초 |

학습과 평가처럼 전량이 필요할 때는 `load_seed_examples()`를 쓰세요.

## 파일

| 파일 | 크기 | 메뉴 항목 | 원본 |
|---|---|---|---|
| `receipt_01.jpg` | 768×1024 | 3개 | **`test[1]`** |
| `receipt_02.jpg` | 682×1024 | 3개 | **`test[6]`** |

🔴 **`test` split에서 골랐으며 학습에 쓰이지 않은 이미지입니다.** `train_mm.py`는 `split="train"`만 사용하므로(`train_mm.py:188`) 이 두 장은 모델이 본 적 없습니다. 학습 데이터로 데모하면 정답이 그대로 나와 "잘 된다"고 착각하게 되고, 실제 일반화 성능을 보여주지 못합니다.

- `ground_truth.json`: 각 이미지의 정답 JSON(`{menu: [{name, count, price}]}`)과 원본 인덱스.
- **항목 수가 적은 것을 골랐습니다.** 생성 토큰 수가 곧 추론 시간입니다(약 40ms/토큰, L4 실측). `train` split의 첫 영수증은 메뉴 22개/592토큰으로 추론에 ~24초가 걸립니다.
- 긴 변을 1024로 축소하고 JPEG q88로 저장했습니다. 추론 품질에는 영향이 없고(payload 크기와 추론 시간은 무관, 실측) repository 용량만 줄입니다.

## 출처와 라이선스

**[naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)** (CORD: Consolidated Receipt Dataset), **CC BY 4.0**. 출처를 표시하면 재배포할 수 있습니다. 원본 데이터셋에서 이미 개인정보(상점명, 주소 등)가 마스킹된 상태입니다.

재생성이 필요하면 `ground_truth.json`의 `source_split`과 `source_index`로 원본에서 다시 뽑을 수 있습니다.
