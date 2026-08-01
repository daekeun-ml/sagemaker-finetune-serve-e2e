"""
build_all_tracks.py — 3개 트랙(분류/요약/QA) 노트북 생성 + 공용 train.py/requirements 복사.

플래그십(01_extraction_to_json)은 이미 생성됨. 여기서는 02/03/04를 공유 빌더로 생성한다.
train.py 는 트랙 무관(self-contained)이라 01의 것을 복사해 재사용.
"""
from __future__ import annotations

import os
import shutil

from _shared_build import TrackSpec, build_track

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGSHIP_SCRIPTS = os.path.join(HERE, "01_extraction_to_json", "scripts")

# 🔴 02b §3-D 비교 셀은 "같은 모델 · 같은 질문 · 다른 프롬프트 구성"이 통제 조건이다.
#    그 셀의 A/B는 deploy_smoke_user를, C는 serve_example_user를 쓰므로 두 값이 **같은 문서/질문**에서
#    나와야 한다. 서로 다른 질문을 주면 학습자가 보는 차이가 '프롬프트 구성' 때문인지 '질문이 달라서'인지
#    구분되지 않는다. 그래서 아래처럼 한 소스에서 파생시킨다.

# 03_summarization — 요약할 문서 본문. 학습 입력은 본문만(track_data 참조)이라
# serve_example_user=본문, deploy_smoke_user=본문 + '맨' 지시("Summarize:")로 준다.
_SUM_DOC = (
    "SECTION 1. SHORT TITLE. This Act may be cited as the 'Rural Broadband Investment Act'. "
    "SEC. 2. FUNDING. Section 12 of the Communications Act is amended to increase funding for "
    "rural broadband deployment by $2,000,000,000 over five fiscal years, with oversight by the "
    "Federal Communications Commission. SEC. 3. REPORTING. The Commission shall submit an annual "
    "report to Congress detailing the allocation and outcomes of funds disbursed under this Act."
)

# 04_domain_qa — 질문 + 선택적 context. 학습 입력 포맷은 track_data._compose_input 과 동일하게
# "{instruction}\n\n[Context]\n{context}" (지시문 먼저, 대괄호 헤더) 여야 한다.
_QA_QUESTION = "What is the capital of Australia and why is it not Sydney?"
_QA_CONTEXT = (
    "Canberra was selected as the site for Australia's capital in 1908 as a compromise "
    "between rival cities Sydney and Melbourne, which had both sought the honour."
)

SPECS = [
    TrackSpec(
        key="classification", dir_name="02_classification", title="텍스트 분류(intent)",
        endpoint_prefix="gemma-classification", max_seq_length=512, use_qlora=True,
        eval_kind="classification", grpo_reward_kind="classification",
        tool_name="classify_intent",
        tool_doc="Classify a banking customer message into a single intent label using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. When the user gives a customer message, call classify_intent to get "
                      "the intent label, then explain the routing decision."),
        smoke_user="A customer writes: 'My card still hasn't arrived after two weeks.' Classify the intent and suggest next action.",
        deploy_smoke_user="My new card hasn't arrived yet, what should I do?",
        # 🔴 02b 로컬 서빙 예시: 학습 형태(고객 메시지 → 단일 intent 라벨)와 같게 준다.
        serve_example_user="My new card hasn't arrived yet, what should I do?",
        dataset_blurb=(
            "**시드 데이터셋**: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) "
            "(mit, ungated) — 은행 고객 문의 intent 분류 셋(BANKING77).\n"
            "- **원본 포맷**: `text`(고객 메시지) + `label`(정수) + `label_text`(intent 이름, 예: `card_arrival`).\n"
            "- **이 트랙의 파싱**: `text` → `input`, intent 라벨명 → `output`. JumpStart엔 분류 헤드가 없어 "
            "**라벨을 텍스트로 생성**하는 instruction 방식으로 학습합니다.\n"
            "- **성공 기준**: 라벨 정확 일치(accuracy / macro-F1).\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "text:       I am still waiting on my card?\n"
            "label:      11\n"
            "label_text: card_arrival        # 77개 intent 중 하나\n"
            "```\n"
            "→ 파싱 후: `input`=고객 메시지, `output`=`card_arrival`(라벨 이름 텍스트)\n\n"
            "🔴 **왜 원본 `PolyAI/banking77`이 아닌가**: 그 리포는 **스크립트 기반**(`banking77.py`)이라 "
            "이 kit이 핀한 `datasets>=5.0.0`에서 `Dataset scripts are no longer supported` 로 **로드 자체가 실패**합니다"
            "(실측 2026-07-30). parquet 자동변환본도 없어 되살릴 수 없으므로, 내용이 동일한 parquet 미러를 씁니다.\n"
            "🔴 **셔플이 필수인 이유**: banking77의 train 스플릿은 **라벨 정렬 순서**입니다. 앞에서부터 뽑으면 "
            "300건에 클래스가 3개뿐이고 평가 held-out이 단일 라벨로 붕괴합니다(실측). "
            "`load_seed_examples`가 고정 시드(42)로 셔플하므로 재현성은 유지됩니다."
        ),
    ),
    TrackSpec(
        key="summarization", dir_name="03_summarization", title="문서 요약",
        endpoint_prefix="gemma-summarization", max_seq_length=2048, use_qlora=True,
        # 🔴 요약 트랙은 '긴 입력 + 긴 정답' 이라 서빙/생성 길이를 학습 길이와 분리해야 한다(실측 2026-07-30).
        #    held-out 프롬프트: median 1370 / max 2006 토큰 → 학습값 2048을 서빙 컨텍스트로 쓰면
        #    (2006 + 생성 256) > 2048 이라 vLLM이 400(context length exceeded)으로 거부한다.
        #    정답 요약: median 209 / p90 475 / max 964 토큰 → 256으로 자르면 held-out 40%가 절단돼
        #    ROUGE-L이 구조적으로 과소 측정된다.
        serve_max_model_len=4096,   # 프롬프트 max 2006 + 생성 512 에 여유
        gen_max_tokens=512,         # p90(475)을 덮음. 전량을 덮으려면 1024(정답 max 964)
        eval_kind="summarization",
        tool_name="summarize_document",
        tool_doc="Summarize a document concisely and faithfully using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. When the user provides a long document, call summarize_document, "
                      "then optionally refine or extract key action items."),
        smoke_user="Summarize this: 'The bill amends section 12 to increase funding for rural broadband by $2B over 5 years, with oversight by the FCC...'",
        # 🔴 02b §3-D의 A/B가 쓰는 '맨' 입력 — serve_example_user와 **같은 문서**여야 비교가 성립한다.
        #    학습 형태와의 차이는 "Summarize:" 지시를 앞에 붙였다는 점뿐이다.
        deploy_smoke_user="Summarize: " + _SUM_DOC,
        # 🔴 02b 로컬 서빙 예시: 학습 형태(문서 본문 → 요약)에 맞춰 '요약해달라'는 지시 없이 본문만 준다.
        #    SYSTEM_PROMPT가 이미 "요약기"라고 지시하므로 본문만 주는 것이 학습 데이터와 같다.
        serve_example_user=_SUM_DOC,
        dataset_blurb=(
            "**시드 데이터셋**: [`FiscalNote/billsum`](https://huggingface.co/datasets/FiscalNote/billsum) "
            "(cc0-1.0, public domain) — 미국 법안 요약 셋.\n"
            "- **원본 포맷**: `text`(법안 본문) + `summary`(사람이 작성한 요약) + `title`.\n"
            "- **이 트랙의 파싱**: `text` → `input`, `summary` → `output`. **긴 문서 → 간결 요약** 쌍.\n"
            "- **성공 기준**: ROUGE-L(자동) + Bedrock LLM-judge(groundedness/coverage). "
            "대화체 요약은 완전 permissive 공개셋이 부족해 문서요약 시드 + grounded 합성으로 확장합니다.\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "title:   A bill to limit the civil liability of business entities providing use of facilities...\n"
            "text:    SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES TO NONPROFIT... (긴 법안 본문)\n"
            "summary: Shields a business entity from civil liability relating to any injury or death occurring...\n"
            "```\n"
            "→ 파싱 후: `input`=법안 본문(`text`), `output`=사람이 쓴 요약(`summary`)"
        ),
    ),
    TrackSpec(
        key="domain_qa", dir_name="04_domain_qa", title="도메인 QA / instruction",
        endpoint_prefix="gemma-domainqa", max_seq_length=1024, use_qlora=True,
        # 🔴 dolly는 길이 분포가 넓다(실측 2026-07-30, 150건):
        #    학습 전체(입력+정답) median 141 / p90 420 / max 1945 → 1024면 4건(2.7%) 절단
        #    추론 프롬프트 median 58 / p90 291 / max 1140,  정답 median 39 / p90 218 / max 1781
        #    학습 길이는 1024로 두되(대부분이 훨씬 짧아 메모리 낭비를 피함), 서빙·생성 길이는 넉넉히.
        serve_max_model_len=2048,   # 프롬프트 max 1140 + 생성 512 여유 (1024면 1건 초과)
        gen_max_tokens=512,         # 256이면 정답 13건(8.7%)이 잘려 지표가 과소 측정됨
        eval_kind="domain_qa",
        tool_name="answer_domain_question",
        tool_doc="Answer a domain question (optionally grounded in provided context) using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. For domain questions, call answer_domain_question, then verify the "
                      "answer is grounded and add citations if context was provided."),
        smoke_user="Using the SLM, answer: 'What is the capital of Australia and why is it not Sydney?'",
        # 🔴 02b §3-D의 A/B가 쓰는 '맨' 질문 — serve_example_user와 **같은 질문**이어야 비교가 성립한다.
        #    학습 형태와의 차이는 context가 없다는 점뿐이다.
        deploy_smoke_user=_QA_QUESTION,
        # 🔴 02b 로컬 서빙 예시: 학습 형태(instruction + 선택적 context)에 맞춰 context를 함께 준다.
        #    포맷은 track_data._compose_input 과 동일하게 '지시문 먼저 + [Context] 헤더' 로 맞춘다
        #    (04 평가·05 agentic이 보내는 ex['input']과 같은 표면형이어야 "학습과 같은 형태"라는 말이 성립).
        serve_example_user=f"{_QA_QUESTION}\n\n[Context]\n{_QA_CONTEXT}",
        dataset_blurb=(
            "**시드 데이터셋**: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) "
            "(cc-by-sa-3.0, ungated) — 사람이 작성한 instruction-following 셋.\n"
            "- **원본 포맷**: `instruction` + `context`(선택) + `response` + `category`(예: open_qa, closed_qa, summarization).\n"
            "- **이 트랙의 파싱**: `instruction`(+`context`) → `input`, `response` → `output`.\n"
            "- **성공 기준**: Bedrock LLM-judge(correctness/helpfulness/groundedness) + ROUGE-L proxy.\n"
            "- ⚠️ **CC-BY-SA**: 파생물 share-alike 의무 — 학습·배포 산출물에 라이선스가 전파됩니다.\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "instruction: When did Virgin Australia start operating?\n"
            "context:     Virgin Australia... commenced services on 31 August 2000... (선택 필드, 없을 수도)\n"
            "response:    Virgin Australia commenced services on 31 August 2000 as Virgin Blue...\n"
            "category:    closed_qa\n"
            "```\n"
            "→ 파싱 후: `input`=instruction(+context), `output`=response"
        ),
    ),
]


def main() -> None:
    for spec in SPECS:
        build_track(spec, tracks_root=HERE)
        # train.py + requirements 복사 (self-contained, 트랙 무관)
        dst = os.path.join(HERE, spec.dir_name, "scripts")
        os.makedirs(dst, exist_ok=True)
        for fn in ("train.py", "train_grpo.py", "requirements.txt", "serve_local_vllm.sh",
                   "bench_local_vllm.sh", "cleanup_local.sh", "inference.py"):
            shutil.copy(os.path.join(FLAGSHIP_SCRIPTS, fn), os.path.join(dst, fn))
        print(f"   ↳ scripts/ 복사 완료 ({spec.dir_name})")
    print("done: 3 tracks")


if __name__ == "__main__":
    main()
