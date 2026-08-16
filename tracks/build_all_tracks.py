"""분류, 요약, QA 트랙의 노트북과 공통 스크립트를 생성합니다."""
from __future__ import annotations

import os
import shutil

from _shared_build import TrackSpec, build_track

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGSHIP_SCRIPTS = os.path.join(HERE, "01_extraction_to_json", "scripts")

# 로컬 서빙 비교 셀은 같은 입력에 다른 프롬프트 구성을 적용해야 합니다.

# 요약 트랙은 같은 문서에 지시문 유무만 다르게 적용합니다.
_SUM_DOC = (
    "SECTION 1. SHORT TITLE. This Act may be cited as the 'Rural Broadband Investment Act'. "
    "SEC. 2. FUNDING. Section 12 of the Communications Act is amended to increase funding for "
    "rural broadband deployment by $2,000,000,000 over five fiscal years, with oversight by the "
    "Federal Communications Commission. SEC. 3. REPORTING. The Commission shall submit an annual "
    "report to Congress detailing the allocation and outcomes of funds disbursed under this Act."
)

# QA 트랙은 track_data._compose_input과 같은 입력 형식을 사용합니다.
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
        # 학습 데이터와 같은 고객 메시지 형식을 사용합니다.
        serve_example_user="My new card hasn't arrived yet, what should I do?",
        dataset_blurb=(
            "**시드 데이터셋**: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) "
            "(MIT, ungated). 은행 고객 문의 intent 분류 데이터셋입니다.\n"
            "- **원본 포맷**: `text`(고객 메시지) + `label`(정수) + `label_text`(intent 이름, 예: `card_arrival`).\n"
            "- **이 트랙의 파싱**: `text`를 `input`, intent 라벨명을 `output`으로 변환합니다. JumpStart엔 분류 헤드가 없어 "
            "**라벨을 텍스트로 생성**하는 instruction 방식으로 학습합니다.\n"
            "- **성공 기준**: 라벨 정확 일치(accuracy / macro-F1).\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "text:       I am still waiting on my card?\n"
            "label:      11\n"
            "label_text: card_arrival        # 77개 intent 중 하나\n"
            "```\n"
            "파싱 후: `input`=고객 메시지, `output`=`card_arrival`\n\n"
            "**데이터셋 선택**: `PolyAI/banking77`은 스크립트 기반이라 `datasets>=5.0.0`에서 "
            "불러올 수 없어 내용이 같은 parquet 미러를 사용합니다.\n"
            "**셔플**: train 스플릿이 라벨 순서로 정렬되어 있어 고정 시드로 섞습니다. "
            "`load_seed_examples`가 고정 시드(42)로 셔플하므로 재현성은 유지됩니다."
        ),
    ),
    TrackSpec(
        key="summarization", dir_name="03_summarization", title="문서 요약",
        endpoint_prefix="gemma-summarization", max_seq_length=2048, use_qlora=True,
        # 긴 입력과 출력을 위해 서빙 컨텍스트와 생성 길이를 학습 길이보다 크게 둡니다.
        serve_max_model_len=4096,
        gen_max_tokens=512,
        eval_kind="summarization",
        tool_name="summarize_document",
        tool_doc="Summarize a document concisely and faithfully using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. When the user provides a long document, call summarize_document, "
                      "then optionally refine or extract key action items."),
        smoke_user="Summarize this: 'The bill amends section 12 to increase funding for rural broadband by $2B over 5 years, with oversight by the FCC...'",
        # 같은 문서에 "Summarize:" 지시만 추가합니다.
        deploy_smoke_user="Summarize: " + _SUM_DOC,
        # 학습 데이터와 같은 문서 본문만 전달합니다.
        serve_example_user=_SUM_DOC,
        dataset_blurb=(
            "**시드 데이터셋**: [`FiscalNote/billsum`](https://huggingface.co/datasets/FiscalNote/billsum) "
            "(CC0-1.0, public domain). 미국 법안 요약 데이터셋입니다.\n"
            "- **원본 포맷**: `text`(법안 본문) + `summary`(사람이 작성한 요약) + `title`.\n"
            "- **이 트랙의 파싱**: `text`를 `input`, `summary`를 `output`으로 변환합니다.\n"
            "- **성공 기준**: ROUGE-L(자동) + Bedrock LLM-judge(groundedness/coverage). "
            "대화체 요약은 완전 permissive 공개셋이 부족해 문서요약 시드 + grounded 합성으로 확장합니다.\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "title:   A bill to limit the civil liability of business entities providing use of facilities...\n"
            "text:    SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES TO NONPROFIT... (긴 법안 본문)\n"
            "summary: Shields a business entity from civil liability relating to any injury or death occurring...\n"
            "```\n"
            "파싱 후: `input`=법안 본문, `output`=사람이 쓴 요약"
        ),
    ),
    TrackSpec(
        key="domain_qa", dir_name="04_domain_qa", title="도메인 QA / instruction",
        endpoint_prefix="gemma-domainqa", max_seq_length=1024, use_qlora=True,
        # Dolly의 긴 응답을 수용하도록 서빙 컨텍스트와 생성 길이를 별도로 둡니다.
        serve_max_model_len=2048,
        gen_max_tokens=512,
        eval_kind="domain_qa",
        tool_name="answer_domain_question",
        tool_doc="Answer a domain question (optionally grounded in provided context) using the fine-tuned Gemma SLM.",
        agent_system=("You orchestrate. For domain questions, call answer_domain_question, then verify the "
                      "answer is grounded and add citations if context was provided."),
        smoke_user="Using the SLM, answer: 'What is the capital of Australia and why is it not Sydney?'",
        # 같은 질문에 context 유무만 다르게 적용합니다.
        deploy_smoke_user=_QA_QUESTION,
        # 학습 데이터와 같은 instruction 및 context 형식을 사용합니다.
        serve_example_user=f"{_QA_QUESTION}\n\n[Context]\n{_QA_CONTEXT}",
        dataset_blurb=(
            "**시드 데이터셋**: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) "
            "(CC-BY-SA-3.0, ungated). 사람이 작성한 instruction-following 데이터셋입니다.\n"
            "- **원본 포맷**: `instruction` + `context`(선택) + `response` + `category`(예: open_qa, closed_qa, summarization).\n"
            "- **이 트랙의 파싱**: `instruction`과 `context`를 `input`, `response`를 `output`으로 변환합니다.\n"
            "- **성공 기준**: Bedrock LLM-judge(correctness/helpfulness/groundedness) + ROUGE-L proxy.\n"
            "- **라이선스**: CC-BY-SA의 share-alike 의무가 파생물에 적용됩니다.\n\n"
            "**원본 row 예시** (raw):\n"
            "```text\n"
            "instruction: When did Virgin Australia start operating?\n"
            "context:     Virgin Australia... commenced services on 31 August 2000... (선택 필드, 없을 수도)\n"
            "response:    Virgin Australia commenced services on 31 August 2000 as Virgin Blue...\n"
            "category:    closed_qa\n"
            "```\n"
            "파싱 후: `input`=instruction(+context), `output`=response"
        ),
    ),
]


def main() -> None:
    for spec in SPECS:
        build_track(spec, tracks_root=HERE)
        # 공통 학습 스크립트를 각 트랙에 복사합니다.
        dst = os.path.join(HERE, spec.dir_name, "scripts")
        os.makedirs(dst, exist_ok=True)
        for fn in ("train.py", "train_grpo.py", "requirements.txt", "serve_local_vllm.sh",
                   "bench_local_vllm.sh", "cleanup_local.sh", "inference.py"):
            shutil.copy(os.path.join(FLAGSHIP_SCRIPTS, fn), os.path.join(dst, fn))
        print(f"   scripts 복사 완료: {spec.dir_name}")
    print("완료: 트랙 3개")


if __name__ == "__main__":
    main()
