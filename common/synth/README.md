# 합성 데이터 생성 — 경로 선택

파인튜닝 데이터가 부족할 때 seed에 **grounded**된 합성 데이터를 만든다. 이 kit은 **기본 경로(무의존성)**를
쓰고, 필요 시 **활발히 유지보수되는 오픈 라이브러리 대안**을 붙일 수 있게 한다.

> 검증: synth-toolkit-recon 워크플로우(2026-07-19), GitHub/PyPI 실측 + 적대적 검증.
> 🔴 **distilabel은 사용하지 않는다** — 개발 정체 확인(마지막 릴리스 v1.5.3 @ 2025-01-28, 2026년 릴리스 0건).

---

## ✅ 기본 경로 (권장) — `bedrock_synth.py`
**boto3 Bedrock Converse + critique/refine 루프 (외부 SDG 라이브러리 0개).**
- 우리 코드라 **라이브러리 노후화 리스크 없음** — production kit의 기본값.
- IAM/VPC/Guardrails 등 AWS 네이티브 거버넌스를 그대로 사용.
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

## 대안 라이브러리 (활발히 유지보수 — distilabel 대체)

### 1순위: **Kiln** (`kiln-ai`) — native Bedrock
- 유일하게 **native Amazon Bedrock 지원**을 코드로 확인 (`ModelProviderName.amazon_bedrock`).
- 가장 활발 (v1.0.4 @ 2026-07-16). GUI + Python 라이브러리, SFT 데이터 생성·평가·파인튜닝 오케스트레이션.
- `pip install kiln-ai`  ⚠️ core lib(kiln-ai)는 MIT이나 리포 루트는 커스텀 라이선스 — 라이선스 확인.
- repo: github.com/Kiln-AI/Kiln

### 코드-우선 대안: **Bespoke Labs Curator** — via LiteLLM
- 코드-우선 SDG 라이브러리, 구조화 출력(Pydantic)·체이닝·캐싱·장애복구. 대량 실행에 강함.
- Bedrock은 **LiteLLM 경유**(`bedrock/...` 모델 문자열 + AWS 자격증명). native 커넥터는 아님.
- `pip install bespokelabs-curator` (PyPI 0.1.29 @ 2026-07-13, 활발). Apache-2.0.
- repo: github.com/bespokelabsai/curator
- → 이 kit의 `common/llm_gateway.py`(LiteLLM)와 라우팅 규약 일치.

### 문서-그라운디드 QA 특화 (참고)
- **meta-llama/synthetic-data-kit** — 문서(PDF/HTML 등)→QA/CoT 생성. 단 케이던스 둔화(2025-10 이후 정체),
  Bedrock은 미문서화(OpenAI-호환/vLLM 백엔드) → LiteLLM 프록시 필요.
- **Augmentoolkit** — grounded factual datagen 특화, 커밋 활발하나 config/CLI 중심(라이브러리성 낮음), Bedrock 미문서화.

### ⚠️ 배제 / 주의
- **DeepFabric**(구 promptwright): LiteLLM 미사용·Bedrock 미지원(정찰의 초기 주장은 반증됨).
- **NVIDIA NeMo Curator**: 활발하나 Bedrock **소비** 커넥터 없음(자체 NIM/vLLM 엔드포인트 호스팅용), GPU 큐레이션 지향.
- **DataDreamer / fabricator**: 정체(dormant).
- **distilabel**: 정체 — 사용 금지.

> 대안 라이브러리를 쓰더라도 **grounded + critique 원칙**은 동일하게 적용하고, 출력은 이 kit의
> `messages` JSONL 포맷으로 변환해 `train.py`에 넣는다.
