"""
_build_notebooks.py — 멀티모달 추출(이미지→JSON) 트랙 노트북 생성기.

텍스트 트랙과 다른 점:
  - 합성 데이터 단계 없음(이미지 합성은 별개) → 01은 시드 탐색만.
  - 학습은 train_mm.py(AutoModelForImageTextToText + processor, vision freeze + language LoRA).
  - 서빙은 멀티모달 그대로(텍스트 재-export 안 함) — vLLM이 이미지 입력을 받음.
셀 헬퍼(header/md/code/_notebook)는 _shared_build에서 재사용.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tracks/ 를 path에

from _shared_build import _notebook, header, md, code, SETUP_PATH  # noqa: E402

TRACK_KEY = "mm_extraction"
ENDPOINT_PREFIX = "gemma-mm-extraction"
MAX_LEN = 2048


def write(name: str, cells: list[dict]) -> None:
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_notebook(cells), f, ensure_ascii=False, indent=1)
    print(f"✅ {name}")


# ---------------------------------------------------------------------------
def build_00():
    from _shared_build import _c00, TrackSpec
    spec = TrackSpec(
        key=TRACK_KEY, dir_name="05_multimodal_extraction", title="멀티모달 추출(이미지→JSON)",
        endpoint_prefix=ENDPOINT_PREFIX, max_seq_length=MAX_LEN, use_qlora=True,
        eval_kind="extraction", tool_name="extract_receipt_json", tool_doc="", agent_system="",
        smoke_user="", deploy_smoke_user="",
    )
    write("00_setup.ipynb", _c00(spec))


def build_01():
    cells = [
        header(
            "01 · 시드 이미지 데이터 탐색 — 멀티모달 추출(이미지→JSON)",
            "영수증 이미지→구조화 JSON 시드 데이터셋(cord-v2)을 로드해 이미지와 정답 JSON을 눈으로 확인합니다.",
            "멀티모달 학습은 '이미지→타깃 텍스트' 쌍이 필요합니다. 텍스트 트랙과 달리 합성 생성 단계 없이 "
            "공개 라벨 데이터를 직접 씁니다(이미지 합성은 별개 문제).",
            "이미지 태스크는 permissive 라이선스 데이터가 드뭅니다. cord-v2(cc-by-4.0, ungated)는 영수증→JSON "
            "라벨이 갖춰져 있어 이미지→구조화 추출 데모에 적합합니다.",
        ),
        code(SETUP_PATH),
        md(
            "## 시드 데이터셋: naver-clova-ix/cord-v2\n"
            "- **라이선스**: cc-by-4.0, ungated.\n"
            "- **원본 포맷**: `image`(영수증 PNG) + `ground_truth`(문자열 JSON, `gt_parse.menu[]` 등).\n"
            "- **이 트랙의 파싱**(`track_data.py`): 이미지 → 별도 `images` 컬럼(TRL VLM collator가 "
            "이미지 자리표시자를 주입), `gt_parse.menu` → `{menu:[{name,count,price}]}` 타깃 JSON.\n"
            "- **성공 기준**: 영수증 이미지에서 메뉴/수량/가격을 정확한 JSON으로 추출.\n\n"
            "**원본 row 예시**:\n"
            "```text\n"
            "image:        <영수증 PNG, 예 864x1296>\n"
            'ground_truth: {"gt_parse": {"menu": [{"nm":"Nasi Campur Bali","cnt":"1 x","price":"75,000"}, ...]}}\n'
            "```\n"
            "→ 파싱 후 타깃: `{\"menu\": [{\"name\":\"Nasi Campur Bali\",\"count\":\"1 x\",\"price\":\"75,000\"}, ...]}`"
        ),
        code(
            "import importlib, track_data as td; importlib.reload(td)\n"
            "from common import config\n"
            "seeds = td.load_seed_examples(3, token=config.get_hf_token())\n"
            "ex = seeds[0]\n"
            "# TRL VLM 포맷: images 컬럼(별도) + messages(텍스트만). collator가 이미지 placeholder를 주입.\n"
            "print('columns     :', [k for k in ex if not k.startswith('_')])\n"
            "print('user text   :', ex['messages'][0]['content'][:120])\n"
            "print('target JSON :', ex['messages'][1]['content'][:300])\n"
            "print('num images  :', len(ex['images']))"
        ),
        code(
            "# 이미지 미리보기 (노트북에서 렌더)\n"
            "img = ex['_image']   # load_seed_examples가 편의로 담아 준 원본 PIL\n"
            "print('image size:', img.size)\n"
            "img   # Jupyter가 이미지를 렌더링"
        ),
        md("✅ 이미지와 타깃 JSON을 확인했습니다. 다음은 **02_train_mm_sagemaker.ipynb**로 멀티모달 SFT를 실행합니다. "
           "(이 트랙은 합성 데이터 단계가 없습니다.)"),
    ]
    write("01_data_explore.ipynb", cells)


def build_02():
    cells = [
        header(
            "02 · SageMaker 멀티모달 SFT (이미지→JSON) — 멀티모달 추출",
            "`scripts/train_mm.py`를 SageMaker ModelTrainer로 실행합니다. gemma-4를 vision 포함 로드하고, "
            "vision tower는 얼린 채 language LoRA만 학습합니다.",
            "이미지→텍스트 파인튜닝은 processor(이미지 전처리)+멀티모달 모델 클래스가 필요합니다. train_mm.py가 "
            "TRL SFTTrainer에 processor를 넘겨 이미지를 자동 처리하고, vision tower를 freeze해 안정적으로 학습합니다.",
            "멀티모달 base에 무작정 all-linear LoRA를 붙이면 vision proj(ClippableLinear)에서 크래시합니다. "
            "language_model 한정 regex target으로 이를 피합니다(실측 검증).",
        ),
        code(SETUP_PATH),
        code(
            "import importlib, boto3\n"
            "from common import config, dlc, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "from sagemaker.train.model_trainer import ModelTrainer\n"
            "from sagemaker.core.training.configs import SourceCode, Compute, StoppingCondition\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            "%store -r role\n"
            "if 'role' not in dir() or not role or ':role/' not in str(role):\n"
            "    role = config.resolve_sagemaker_role(sess)\n"
            "TRACK = config.TRACKS['mm_extraction']\n"
            "print('role:', role, '| seed:', TRACK.seed_dataset, '| multimodal:', TRACK.multimodal)"
        ),
        md(
            "## 학습 구성\n"
            "🔴 이 트랙은 시드 이미지 데이터셋(cord-v2)을 컨테이너 안에서 직접 로드합니다(합성/업로드 단계 없음). "
            "`train_mm.py`가 `--seed_dataset`에서 이미지를 받아 학습합니다. gemma-4는 apache-2.0/ungated라 HF 토큰이 필요 없습니다."
        ),
        code(
            "MAX_TRAIN_SAMPLES = 200   # 멀티모달은 무거우니 작게 시작. 정식은 None(전체).\n"
            "MAX_RUNTIME_HOURS = 4   # 생략 시 SDK 기본 1시간 → 머지 중 강제 중단(docs/03 §4.5)\n"
            "hyperparameters = {\n"
            "    'model_id': config.DEFAULT_MODEL_ID,   # gemma-4 (멀티모달)\n"
            "    'seed_dataset': TRACK.seed_dataset,\n"
            "    'epochs': 2, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8,\n"
            "    'learning_rate': 2e-4,\n"
            f"    'max_seq_length': {MAX_LEN},\n"
            "    'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05,\n"
            "    'use_qlora': True, 'freeze_vision': True, 'merge_adapter': True,\n"
            "}\n"
            "if MAX_TRAIN_SAMPLES:\n"
            "    hyperparameters['max_train_samples'] = MAX_TRAIN_SAMPLES\n"
            "environment = {'HF_TOKEN': config.get_hf_token()} if config.get_hf_token() else {}\n"
            "image_uri = dlc.resolve_training_image(config.AWS_REGION)\n"
            "assert image_uri, 'DLC 이미지 해석 실패 — DLC_IMAGE_URI env로 지정: ' + dlc.AVAILABLE_IMAGES_URL\n"
            "trainer = ModelTrainer(\n"
            "    training_image=image_uri,\n"
            "    source_code=SourceCode(source_dir='scripts', entry_script='train_mm.py',\n"
            "                           requirements='requirements.txt'),   # torchvision 포함\n"
            "    compute=Compute(instance_type=config.TRAIN_INSTANCE_TYPE, instance_count=1),\n"
            "    hyperparameters=hyperparameters,\n"
            "    environment=environment,\n"
            "    role=role,\n"
            "    sagemaker_session=sess,\n"
            f"    base_job_name='{ENDPOINT_PREFIX}-train',\n"
            "    stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_HOURS * 3600),\n"
            ")"
        ),
        md(
            "## 학습 시작 (비동기) — 데이터는 컨테이너가 직접 로드하므로 input_data 채널 불필요\n"
            "이미지 시드를 컨테이너 안에서 `load_dataset`으로 받으므로 별도 train 채널을 붙이지 않습니다."
        ),
        code(
            "trainer.train(wait=False, logs=False)\n"
            "from IPython.display import display\n"
            "job = trainer._latest_training_job\n"
            "print('training job:', job.training_job_name)\n"
            "display(aws_utils.cw_links(config.AWS_REGION, training_job=job.training_job_name))"
        ),
        md(
            "### 진행 상태 확인 (이 셀만 반복 실행)\n"
            "필요할 때마다 이 셀을 다시 실행해 진행 단계를 봅니다. "
            "`Starting → Pending(용량 대기) → Downloading(이미지 pull) → Training(코드 실행)` 순으로 진행되며, "
            "**Training 단계부터 CloudWatch 로그가 생깁니다**. 멀티모달 학습은 텍스트보다 오래 걸립니다."
        ),
        code(
            "aws_utils.training_job_status(job.training_job_name, config.AWS_REGION)"
        ),
        md(
            "### 세션이 끊겼을 때 잡에 다시 붙기 (재접속)\n"
            "🔴 제출한 학습 잡은 SageMaker 서버에서 돌기 때문에 노트북 커널이 끊겨도 계속 진행됩니다. "
            "다시 붙을 때 **train 셀을 재실행하면 GPU 잡이 중복 제출되어 비용이 두 배로 듭니다** — 대신 아래 셀로 "
            "잡 이름을 조회해 `job` 변수만 복구하세요. 위쪽 설정 셀만 실행한 상태에서 바로 쓸 수 있습니다."
        ),
        code(
            "from sagemaker.core.resources import TrainingJob\n"
            "# 방법 A: 잡 이름을 알면 바로 붙기 (가장 확실 — 위 train 셀 출력에서 복사)\n"
            "# job = TrainingJob.get('<여기에 잡 이름>')\n"
            "# 방법 B: 이름을 잊었으면 base_job_name으로 최근 잡 찾기 (get_all은 최신순)\n"
            f"jobs = list(TrainingJob.get_all(name_contains='{ENDPOINT_PREFIX}-train'))\n"
            "assert jobs, '이 base_job_name으로 제출된 잡이 없습니다. 위 train 셀을 먼저 실행하세요.'\n"
            "job = TrainingJob.get(jobs[0].get_name())\n"
            "job.refresh()\n"
            "print('reattached to:', job.training_job_name)\n"
            "print('status       :', job.training_job_status, '/', job.secondary_status)\n"
            "from IPython.display import display\n"
            "display(aws_utils.cw_links(config.AWS_REGION, training_job=job.training_job_name))"
        ),
        md(
            "## 학습 완료 대기 → 모델 아티팩트\n"
            "잡이 끝나야 모델 아티팩트(S3)가 생깁니다. 아래 셀은 완료까지 폴링합니다. "
            "지금 기다리지 않아도 되며, 나중에 위 재접속 셀로 `job`을 복구한 뒤 다시 실행하면 됩니다."
        ),
        code(
            "import time\n"
            "# `job`은 위 train 셀 또는 재접속 셀에서 정의됩니다(trainer 객체에 의존하지 않음).\n"
            "assert 'job' in dir() and job is not None, (\n"
            "    \"job이 없습니다 — 위 train 셀이나 '세션이 끊겼을 때 잡에 다시 붙기' 셀을 먼저 실행하세요.\")\n"
            "while True:\n"
            "    job.refresh(); st = job.training_job_status; print('status:', st)\n"
            "    if st in ('Completed','Failed','Stopped'): break\n"
            "    time.sleep(30)\n"
            "assert st == 'Completed', f'잡이 {st} 상태입니다. CloudWatch 로그 확인.'\n"
            "model_data = job.model_artifacts.s3_model_artifacts\n"
            "print('MM model artifact:', model_data)\n"
            "md_mm_extraction = model_data   # 트랙 전용 키(전역은 다른 트랙이 덮어씀)\n"
            "%store model_data\n"
            "%store md_mm_extraction"
        ),
        md("✅ 멀티모달 학습이 끝났습니다. 산출물은 **멀티모달 그대로**(vision 포함) 저장됩니다. "
           "다음은 **03_deploy_mm_endpoint.ipynb**로 이미지 입력을 받는 endpoint를 배포합니다."),
    ]
    write("02_train_mm_sagemaker.ipynb", cells)


def build_03():
    cells = [
        header(
            "03 · 멀티모달 Endpoint 배포 & 이미지 추론 — 멀티모달 추출",
            "학습한 멀티모달 모델을 vLLM DLC로 배포하고, 실제 영수증 이미지를 보내 JSON 추출을 확인합니다.",
            "gemma-4 멀티모달은 vLLM(≥0.19)이 이미지 입력(OpenAI 호환 image_url)을 지원합니다. 텍스트 트랙과 달리 "
            "🔴 이미지/오디오 입력을 막지 않고 그대로 서빙합니다(재-export 안 함).",
            "텍스트 트랙에서 쓰던 `--limit-mm-per-prompt image=0` 을 여기서 쓰면 이미지가 막힙니다 — 멀티모달 트랙에선 "
            "그 플래그를 쓰지 않습니다.",
        ),
        code(SETUP_PATH),
        code(
            "import importlib, boto3\n"
            "from common import config, dlc, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.helper.session_helper import Session\n"
            "from sagemaker.serve import ModelBuilder\n"
            "import time\n"
            "sess = Session(boto3.Session(region_name=config.AWS_REGION))\n"
            "%store -r md_mm_extraction\n"
            "%store -r model_data\n%store -r role\n"
            "model_data = globals().get('md_mm_extraction') or globals().get('model_data')\n"
            "if 'role' not in dir() or not role or ':role/' not in str(role):\n"
            "    role = config.resolve_sagemaker_role(sess)\n"
            "\n"
            "# 리전 가드: %store 값이 옛 리전을 가리키면 자동 교체.\n"
            f"model_data = aws_utils.ensure_model_data_in_region(\n"
            f"    locals().get('model_data'), config.AWS_REGION, job_prefix='{ENDPOINT_PREFIX}-train')\n"
            "md_mm_extraction = model_data\n"
            "%store model_data\n"
            "%store md_mm_extraction\n"
            "print('model_data:', model_data)\n"
            "print('role      :', role)"
        ),
        md(
            "## vLLM / SGLang / DJL LMI 로 멀티모달 배포\n"
            "🔴 이미지 입력을 허용해야 하므로 이미지 개수 제한을 1 이상으로 둡니다(텍스트 트랙은 0으로 막지만 여기는 반대). "
            "gemma-4 서빙엔 vLLM ≥ 0.19가 필요하고 이 킷의 기본 이미지가 이를 충족합니다.\n\n"
            "엔진은 `.env`의 `SERVING_ENGINE`이 결정하고 이미지 URI도 거기서 해석합니다 — "
            "텍스트 트랙(`03_deploy_endpoint`)과 같은 방식입니다.\n\n"
            "| `SERVING_ENGINE` | 이미지 허용 옵션 | 비고 |\n"
            "|---|---|---|\n"
            "| `vllm` (기본) | `SM_VLLM_LIMIT_MM_PER_PROMPT` | 접두사를 떼고 `--limit-mm-per-prompt`로 전달 |\n"
            "| `sglang` | (기본 허용) | SGLang은 멀티모달 입력을 기본 허용 |\n"
            "| `lmi` | `OPTION_LIMIT_MM_PER_PROMPT` | LMI는 `OPTION_*`를 **vLLM EngineArguments로 pass-through** |\n\n"
            "> LMI도 멀티모달을 지원합니다 — 공식 vLLM user guide에 Qwen3-VL 예시와 "
            "`OPTION_LIMIT_MM_PER_PROMPT=\"{\\\"image\\\": 4, \\\"video\\\": 0}\"`가 명시돼 있고, "
            "\"LMI supports all additional vLLM EngineArguments in Pass-Through mode\"라고 문서화돼 있습니다.\n"
            "⚠️ 단 LMI는 **번들 vLLM 버전에 종속**됩니다 — gemma-4에는 vLLM ≥ 0.19가 필요하니 최신 LMI 태그를 쓰세요."
        ),
        md(
            "### 🔴 24GB GPU에서 배포가 `Failed`로 끝날 때 — CUDA OOM (실측 2026-07-31)\n"
            "멀티모달 아티팩트는 **vision tower를 포함**하므로 텍스트 트랙보다 무겁습니다(실측 가중치 15.18 GiB). "
            "여기에 vLLM 기본값 두 개가 겹치면 L4 24GB에서 엔진 초기화가 실패합니다.\n\n"
            "**증상** — endpoint가 `Failed`, 이유는 `did not pass the ping health check`뿐입니다. "
            "실제 원인은 CloudWatch 로그 안에 있습니다:\n"
            "```\n"
            "Available KV cache memory: 4.69 GiB\n"
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.\n"
            "  GPU 0 has a total capacity of 21.96 GiB of which 147.12 MiB is free.\n"
            "  ...in flashinfer_sample -> top_k_mask_logits -> torch.empty_like(logits)\n"
            "```\n\n"
            "**왜 하필 256 MiB인가** — 이게 결정적 단서입니다. 샘플러의 logits 버퍼 크기가\n"
            "`max_num_seqs × vocab_size × 4B = 256 × 262,144 × 4B = 정확히 256 MiB`입니다. "
            "즉 **모델이 커서가 아니라, 동시 시퀀스 기본값(256)이 실습 규모에 비해 과하게 잡혀서** 터진 것입니다.\n\n"
            "예산을 보면 왜 아슬아슬한지 보입니다(한도 = 21.96 × 0.92 = 20.21 GiB):\n\n"
            "| 항목 | 크기 |\n|---|---|\n"
            "| 가중치(vision 포함) | 15.18 GiB |\n"
            "| KV 캐시 (vLLM이 자동 배정) | 4.69 GiB |\n"
            "| **남은 여유** | **0.34 GiB** |\n"
            "| 실제로 더 필요한 양 (활성 0.27 + 비torch 0.07 + CUDAGraph 0.78) | 1.12 GiB → **0.78 GiB 부족** |\n\n"
            "vLLM 자신도 로그에서 `--kv-cache-memory=3.76 GiB`를 권고합니다 — **KV를 4.69로 과대 배정한 것**입니다.\n\n"
            "**대응** (아래 셀의 기본값):\n"
            "- `MAX_NUM_SEQS=32` — logits 버퍼가 256 MiB → 32 MiB로 줄어듭니다. 실습은 동시 요청이 1~2건이라 손실이 없습니다.\n"
            "- `GPU_MEM_UTIL=0.90` — KV 과대 배정을 막아 여유를 남깁니다.\n\n"
            "🔴 **GPU를 바꿀 필요는 없습니다** — 실측으로 L4와 같은 절대 예산(20.2 GiB)으로 제한한 L40S에서 "
            "이 설정으로 **로드 + 이미지 추론이 정상 동작**했습니다(KV 3.36 GiB, 여유 1.54 GiB). "
            "다만 동시 처리량이 필요하거나 `MAX_LEN`을 크게 늘릴 때는 `ml.g6e.2xlarge`(L40S 45GB)가 여유롭습니다 — "
            "그 경우 `MAX_NUM_SEQS`를 다시 올리세요.\n"
            "> ⚠️ 이 값들은 **컨테이너의 vLLM 버전에 따라 민감도가 다릅니다**. 같은 예산에서 0.25.1은 KV를 3.36 GiB로, "
            "0.26.0은 4.69 GiB로 잡았습니다(실측). 그래서 버전에 기대지 않고 명시적으로 낮춰 둡니다."
        ),
        code(
            "import json\n"
            "# 🔴 엔진/이미지는 env가 결정합니다(SERVING_ENGINE, *_IMAGE_URI). 텍스트 트랙과 동일한 해석 경로.\n"
            "ENGINE = config.SERVING_ENGINE\n"
            "for name, uri in dlc.serving_image_table(config.AWS_REGION).items():\n"
            "    print(('→ ' if name == ENGINE else '   ') + f'{name:8s} {uri}')\n"
            "print()\n"
            f"endpoint_name = f'{ENDPOINT_PREFIX}-{{ENGINE}}-{{int(time.time())}}'\n"
            "serve_image = dlc.resolve_serving_image(config.AWS_REGION, ENGINE)\n"
            "assert serve_image, f'{ENGINE} 이미지 해석 실패 — .env의 *_IMAGE_URI 를 확인하세요.'\n"
            "print(f'{ENGINE} DLC image:', serve_image)\n"
            "\n"
            "# 엔진별 env 키는 dlc.serving_env()가 관리. mm_limit=이미지 허용, max_num_seqs/mem_util=OOM 방지.\n"
            "serve_env = dlc.serving_env(\n"
            "    ENGINE,\n"
            f"    max_model_len={MAX_LEN},\n"
            "    max_num_seqs=32,\n"
            "    gpu_memory_utilization='0.90',\n"
            "    mm_limit=json.dumps({'image': 1}),\n"
            "    hf_token=config.get_serving_hf_token(),   # gated 모델일 때만 채워집니다\n"
            ")\n"
            "print('serve_env:', serve_env)\n"
            "mb = ModelBuilder(image_uri=serve_image, s3_model_data_url=model_data,\n"
            "                  env_vars=serve_env, role_arn=role, sagemaker_session=sess,\n"
            "                  instance_type=config.INFER_INSTANCE_TYPE)\n"
            "mb.build()\n"
            "endpoint = mb.deploy(endpoint_name=endpoint_name, initial_instance_count=1,\n"
            "                     instance_type=config.INFER_INSTANCE_TYPE, wait=False)\n"
            "ep_mm_extraction = endpoint_name   # 트랙 전용 키(전역은 다른 트랙이 덮어씀)\n"
            "%store endpoint_name\n"
            "%store ep_mm_extraction\n"
            "from IPython.display import display\n"
            "print('deploying:', endpoint_name)\n"
            "display(aws_utils.cw_links(config.AWS_REGION, endpoint_name=endpoint_name))"
        ),
        md(
            "## InService 대기 → 이미지 추론\n"
            "endpoint가 InService가 되면, 영수증 이미지를 base64 data URL로 실어 OpenAI 호환 chat 스키마로 호출합니다.\n\n"
            "### ⏸️ 세션이 끊겼다면 — 이 셀부터 이어서 실행\n"
            "커널을 재시작했어도 **endpoint는 서버에 살아 있습니다.** 위 배포 셀을 다시 돌릴 필요 없이 "
            "이 셀만 실행하면 호출에 필요한 것(경로·import·`endpoint_name`)이 복구됩니다."
        ),
        code(
            "# ── 세션 재개도 겸하는 셀 (이것만 실행하면 아래 추론 셀이 바로 동작) ──\n"
            "import os, sys, importlib\n"
            "REPO = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
            "for p in (REPO, os.getcwd()):\n"
            "    if p not in sys.path:\n"
            "        sys.path.insert(0, p)\n"
            "from common import config, aws_utils; importlib.reload(config)\n"
            "from sagemaker.core.resources import Endpoint\n"
            "\n"
            "%store -r ep_mm_extraction\n"
            "%store -r endpoint_name\n"
            "endpoint_name = globals().get('ep_mm_extraction') or globals().get('endpoint_name')\n"
            "assert endpoint_name, (\n"
            "    'endpoint_name 이 없습니다. 위에서 배포하거나 직접 지정하세요:\\n'\n"
            "    \"    endpoint_name = 'gemma-mm-extraction-vllm-...'\")\n"
            "\n"
            "ep = Endpoint.get(endpoint_name); ep.refresh()\n"
            "if ep.endpoint_status != 'InService':\n"
            "    print('waiting for InService (', ep.endpoint_status, ')...'); ep.wait_for_status(target_status='InService')\n"
            "print('InService:', endpoint_name)"
        ),
        code(
            "import base64, io, json, time\n"
            "import importlib, track_data as td; importlib.reload(td)\n"
            "from common import config, aws_utils\n"
            "from common.display_utils import show_image_inference\n"
            "\n"
            "MAX_TOKENS = 768   # 정답 JSON 최대 592토큰(실측 100건) — 512로는 잘림\n"
            "\n"
            "# samples/ 에 넣어 둔 영수증 2장을 즉시 로드합니다(데이터셋 다운로드 없음 — 실측 0.0초).\n"
            "#    전량이 필요하면 td.load_seed_examples(n) — 캐시가 없으면 ~40초 걸립니다.\n"
            "samples = td.load_sample_receipts()\n"
            "sample = samples[0]        # samples[1] 로 바꾸면 메뉴가 더 많은 영수증\n"
            "img = sample['image']\n"
            "print(f\"{sample['name']} | 메뉴 {sample['menu_items']}개 | {img.size}\")\n"
            "\n"
            "# JPEG로 전송 — payload가 PNG의 1/8(추론 시간은 동일)\n"
            "buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)\n"
            "data_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()\n"
            "messages = [{'role':'user','content':[\n"
            "    {'type':'image_url','image_url':{'url': data_url}},\n"
            "    {'type':'text','text': td.INSTRUCTION}]}]\n"
            "\n"
            "t0 = time.time()\n"
            "out = aws_utils.invoke_sagemaker_chat(endpoint_name, messages, region=config.AWS_REGION,\n"
            "                                      max_tokens=MAX_TOKENS, temperature=0.1)\n"

            "print(f'추론 {time.time() - t0:.1f}s')\n"
            "assert out, '빈 응답 — CloudWatch endpoint 로그를 확인하세요.'\n"
            "show_image_inference(img, out, title='영수증 → JSON 추출')\n"
            "\n"
            "# samples/ 에는 정답도 함께 있어 눈으로 바로 대조할 수 있습니다.\n"
            "print('정답(ground truth):', json.dumps(sample['ground_truth'], ensure_ascii=False))"
        ),
        md("✅ 이미지→JSON 추출을 확인했습니다. 정량 평가는 held-out 이미지로 JSON 필드 정확도를 재면 됩니다. "
           "🔴 실습을 마치면 **99_cleanup.ipynb**로 endpoint를 삭제하세요(과금 중단)."),
    ]
    write("03_deploy_mm_endpoint.ipynb", cells)


def build_99():
    from _shared_build import _c99, TrackSpec
    spec = TrackSpec(
        key=TRACK_KEY, dir_name="05_multimodal_extraction", title="멀티모달 추출(이미지→JSON)",
        endpoint_prefix=ENDPOINT_PREFIX, max_seq_length=MAX_LEN, use_qlora=True,
        eval_kind="extraction", tool_name="", tool_doc="", agent_system="",
        smoke_user="", deploy_smoke_user="",
        # 🔴 이 트랙엔 02b(로컬 서빙 검증)가 없다 → 99_cleanup의 '로컬 모델 정리' 섹션을 넣지 않는다.
        has_local_serve=False,
    )
    write("99_cleanup.ipynb", _c99(spec))


if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
    build_03()
    build_99()
    print("done (00,01,02,03,99) — multimodal track")
