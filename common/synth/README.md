# 합성 데이터 생성: 경로 선택

fine-tuning 데이터가 부족할 때 seed에 **grounded**된 합성 데이터를 만듭니다. 이 프로젝트는
외부 SDG 라이브러리가 없는 기본 구현을 사용하고, 필요하면 유지보수 중인 open source 대안을 연결합니다.

> 검증: synth-toolkit-recon 워크플로우(2026-07-19), GitHub/PyPI 실측 + 적대적 검증.
> 🔴 **distilabel은 사용하지 않는다**: 개발 정체 확인(마지막 릴리스 v1.5.3 @ 2025-01-28, 2026년 릴리스 0건).

---

## ✅ 기본 경로 (권장): `bedrock_synth.py`
**boto3 Bedrock Converse + critique/refine 루프 (외부 SDG 라이브러리 0개).**
- 프로젝트가 직접 관리하는 코드라 외부 라이브러리 중단 위험이 없습니다. production 환경의 기본 구현입니다.
- IAM, VPC, Guardrails 같은 AWS native 기능을 그대로 사용합니다.
- 생성 → groundedness/relevance critique → 통과분만 채택 → PII/중복 필터.
- 모델 ID는 `config.BEDROCK_CLAUDE_MODEL_ID`(env), 하드코딩 금지.

```python
from common.synth import bedrock_synth as bs
synth = bs.generate_grounded(
    task_instruction=..., seed_texts=..., n_total=500,
    model_id=cfg.BEDROCK_CLAUDE_MODEL_ID, region=cfg.BEDROCK_REGION,
    to_messages=track_adapter,
)
```

---

## 대안 라이브러리 (distilabel 대체)

### 1순위: **Kiln** (`kiln-ai`), native Bedrock
- 유일하게 **native Amazon Bedrock 지원**을 코드로 확인 (`ModelProviderName.amazon_bedrock`).
- v1.0.4(2026-07-16). GUI와 Python 라이브러리, SFT 데이터 생성, 평가, fine-tuning orchestration을 제공합니다.
- `pip install kiln-ai`  ⚠️ core lib(`kiln-ai`)는 MIT이지만 repository root는 커스텀 라이선스입니다.
- repo: github.com/Kiln-AI/Kiln

### code-first 대안: **Bespoke Labs Curator**, via LiteLLM
- code-first SDG 라이브러리. 구조화 출력(Pydantic), chaining, caching, 장애 복구를 지원합니다.
- Bedrock은 **LiteLLM 경유**(`bedrock/...` 모델 문자열 + AWS 자격증명). native 커넥터는 아님.
- `pip install bespokelabs-curator` (PyPI 0.1.29 @ 2026-07-13, 활발). Apache-2.0.
- repo: github.com/bespokelabsai/curator
- 이 프로젝트의 `common/llm_gateway.py`(LiteLLM)와 라우팅 규약이 같습니다.

### 문서 grounded QA 특화 (참고)
- **meta-llama/synthetic-data-kit**: 문서(PDF/HTML 등)→QA/CoT 생성. 다만 2025-10 이후 개발 속도가 느려졌고,
  Bedrock은 미문서화(OpenAI-호환/vLLM 백엔드) → LiteLLM 프록시 필요.
- **Augmentoolkit**: grounded factual datagen 특화. 커밋은 활발하지만 config/CLI 중심이고 Bedrock은 문서화되지 않았습니다.

### ⚠️ 배제 / 주의
- **DeepFabric**(구 promptwright): LiteLLM 미사용, Bedrock 미지원(초기 조사 결과를 재검증해 정정).
- **NVIDIA NeMo Curator**: 활발하나 Bedrock 호출 connector 없음(자체 NIM/vLLM endpoint 호스팅용), GPU curation 중심.
- **DataDreamer / fabricator**: 정체(dormant).
- **distilabel**: 유지보수 정체, 사용 금지.

> 대안 라이브러리를 쓰더라도 **grounded + critique 원칙**은 동일하게 적용하고, 출력은 이 프로젝트의
> `messages` JSONL 포맷으로 변환해 `train.py`에 넣습니다.
