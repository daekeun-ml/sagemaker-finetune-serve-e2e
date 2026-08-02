"""
pipelines/run_benchmark.py — 배포된 endpoint 의 속도를 잰다 (TTFT / TPOT / ITL / E2EL).

    python pipelines/run_benchmark.py --course extraction          # 상태 파일의 endpoint 를 잰다
    python pipelines/run_benchmark.py --endpoint-name my-endpoint   # 이름을 직접 주고 잰다
    python pipelines/run_benchmark.py --course extraction -- --goodput ttft:200 tpot:50

🔴 이것은 run_<course>.py 와 **다른 종류의 실행**이다. 그래서 스테이지가 아니라 별도 진입점이다.
   · run_<course>.py 는 모델을 만들어 배포하는 흐름이다(data→train→deploy→eval). 순서가 있고,
     앞 단계가 뒤 단계의 선행조건이다.
   · 벤치마크는 **이미 있는 endpoint** 를 재는 일이다. 선행조건이 endpoint 하나뿐이고, 설정을
     바꿔 몇 번씩 다시 돌리는 성격이다(동시성을 올려 가며 한계를 찾는 식).
   둘을 한 파이프라인에 두면 "배포를 다시 해야 벤치마크를 하나?" 를 매번 되묻게 된다.

측정은 sm-endpoint-bmt 가 한다(https://github.com/daekeun-ml/sm-endpoint-bmt).
지표 공식·필드명·출력 표를 `vllm bench serve` 와 맞춘 도구이고, 로컬 vLLM 과 같은 부하를 재생해
대조 검증한 이력이 있다. 여기서 다시 구현하지 않는 이유가 그것이다 — 같은 지표를 두 곳에서
구현하면 숫자가 갈리는 순간 어느 쪽이 맞는지 판단할 근거가 없어진다.

이 파일이 하는 일은 셋뿐이다:
  1. endpoint 이름을 찾는다(--endpoint-name 또는 코스 상태 파일)
  2. config.yaml 의 benchmark 섹션을 sm-endpoint-bmt 의 CLI 인자로 옮긴다
  3. 그 도구를 부른다. `--` 뒤에 준 인자는 그대로 전달되어 위 설정을 덮는다

sm-endpoint-bmt 는 코어 의존성이라 `uv sync` 로 함께 설치된다. 단 그 도구는 Python 3.12+ 이고
kit 은 3.10 부터 지원하므로, 3.10/3.11 에서는 마커에 걸려 빠진다(pyproject 주석 참고).
"""
from __future__ import annotations

import os
import sys

# `python pipelines/run_benchmark.py` 로 직접 실행하면 sys.path[0] 이 pipelines/ 라 `import pipelines`
# 가 안 된다 → 리포 루트를 넣어 준다.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse  # noqa: E402

from pipelines._common import (  # noqa: E402
    STATE_DIR,
    StateStore,
    load_courses,
)
from pipelines._config import load_config  # noqa: E402

_INSTALL_HINT = (
    "sm-endpoint-bmt 를 import 할 수 없습니다.\n"
    f"  현재 파이썬: {sys.version.split()[0]}\n"
    "  🔴 이 도구는 Python 3.12 이상을 요구합니다. kit 은 3.10 부터 지원하므로 pyproject 에\n"
    "     python_version >= '3.12' 마커가 달려 있고, 그 아래 버전에서는 설치되지 않습니다.\n"
    "     3.12+ 환경에서 다시 실행하세요:  uv venv --python 3.12 && uv sync\n"
    "  3.12 인데도 없다면 동기화가 안 된 것입니다:  uv sync\n"
    "  리포: https://github.com/daekeun-ml/sm-endpoint-bmt"
)


def _resolve_endpoint(args, cfg) -> str:
    """--endpoint-name 이 있으면 그것을, 없으면 코스 상태 파일에서 찾는다.

    🔴 SystemExit 를 쓰지 않는다. main() 에서 잡아 메시지를 다시 찍으면 종료 코드가 0 이 되어
       (SystemExit.code 가 문자열이면 파이썬은 그걸 stderr 로 찍고 1 을 쓰지만, 우리가 잡아
       print 하면 그 정보가 사라진다) 스크립트·CI 가 성공으로 읽는다. ValueError 로 올린다.
    """
    if args.endpoint_name:
        return args.endpoint_name
    if not args.course:
        raise ValueError(
            "--endpoint-name 또는 --course 중 하나는 필요합니다.\n"
            "  --course 를 주면 그 코스의 deploy 가 상태 파일에 남긴 endpoint 를 씁니다."
        )
    courses = load_courses()
    if args.course not in courses:
        raise ValueError(f"알 수 없는 코스 {args.course!r} — 허용: {list(courses)}")
    state = StateStore(args.course, state_dir=args.state_dir,
                       dry_run=cfg.runtime.dry_run, path=args.state)
    name = state.get("endpoint_name")
    if not name:
        raise ValueError(
            f"상태 파일에 endpoint_name 이 없습니다({state.path}).\n"
            f"  → 먼저 배포하세요: python pipelines/run_{args.course}.py --stages deploy\n"
            "  → 이미 도는 endpoint 가 있으면 --endpoint-name 으로 직접 주세요."
        )
    return name


def _result_dir(args, cfg) -> str | None:
    """결과 JSON 을 둘 곳. 코스를 알면 그 코스 데이터 디렉터리, 모르면 ./bench_results.

    🔴 지정하지 않으면 도구가 **현재 디렉터리**에 쓴다. 리포 루트에서 돌리는 것이 보통이라
       sagemaker-infqps-concurrency4-<endpoint>-<타임스탬프>.json 이 루트에 쌓인다(실측: 8개).
       파일명에 endpoint 이름이 들어가므로 커밋하면 안 되는 것이기도 하다.
    """
    if not args.course:
        return os.path.join(_REPO, "bench_results")
    from pipelines._common import data_dir, load_courses

    courses = load_courses()
    if args.course not in courses:
        return os.path.join(_REPO, "bench_results")
    return data_dir(courses[args.course], cfg)


def _bench_argv(endpoint_name: str, cfg, passthrough: list[str],
                result_dir: str | None = None) -> list[str]:
    """config.yaml 의 benchmark 섹션을 sm-endpoint-bmt CLI 인자로 옮긴다.

    🔴 passthrough 를 **뒤에** 붙인다. argparse 는 같은 플래그가 두 번 오면 뒤의 값을 쓰므로,
       `-- --num-prompts 500` 이 config.yaml 값을 덮는다. 앞에 붙이면 설정이 사용자 지시를
       이겨서, 명령줄에 쓴 값이 조용히 무시된다.
    """
    b = cfg.benchmark
    n = b.dry_run_num_prompts if cfg.runtime.dry_run else b.num_prompts
    # 🔴 동시성이 건수보다 크면 의미가 없다(보낼 요청이 없어 그 슬롯은 비어 있다). dry-run 은
    #    건수가 작아 이 상황이 쉽게 생기므로 값을 낮추고, 낮췄다는 사실을 알린다.
    conc = min(b.max_concurrency, n)
    if conc < b.max_concurrency:
        print(f"⚠️  max_concurrency {b.max_concurrency} → {conc} (요청 {n}건보다 클 수 없습니다)")
    argv = [
        "--endpoint-name", endpoint_name,
        "--region", os.environ.get("AWS_REGION") or "us-east-1",
        # 이 kit 의 서빙 컨테이너는 셋 다 messages 스키마(OpenAI chat)를 받는다.
        "--endpoint-type", "openai-chat",
        "--dataset-name", "random",
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--random-input-len", str(b.input_len),
        "--random-output-len", str(b.output_len),
        "--num-warmups", str(b.num_warmups),
        # 지표 4종 전부와 백분위를 낸다. 평균만 보면 꼬리 지연을 놓친다.
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", ",".join(
            str(int(q)) if float(q).is_integer() else str(q) for q in b.percentiles),
    ]
    if b.request_rate != "inf":
        argv += ["--request-rate", str(b.request_rate)]
    if b.save_results:
        argv += ["--save-result", "--save-detailed"]
        if result_dir:
            argv += ["--result-dir", result_dir]
    return argv + passthrough


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="배포된 endpoint 의 지연·처리량 측정 (sm-endpoint-bmt 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python pipelines/run_benchmark.py --course extraction\n"
            "  python pipelines/run_benchmark.py --endpoint-name my-endpoint\n"
            "  python pipelines/run_benchmark.py --course extraction --dry-run\n"
            "\n"
            "`--` 뒤의 인자는 sm-endpoint-bmt 에 그대로 전달되며 config.yaml 값을 덮습니다:\n"
            "  ... --course extraction -- --num-prompts 500 --max-concurrency 32\n"
            "  ... --course extraction -- --goodput ttft:200 tpot:50\n"
            "  ... --course extraction -- --ramp-up-strategy linear "
            "--ramp-up-start-rps 1 --ramp-up-end-rps 20\n"
            "  전체 옵션: python -m sagemaker_benchmark --help\n"
            "\n"
            "설정: config.yaml 의 benchmark 섹션 (건수·동시성·부하율·백분위·저장 여부)\n"
            "🔴 endpoint 는 삭제 전까지 시간당 과금됩니다. 측정이 끝나면:\n"
            "     python pipelines/run_<course>.py --stages cleanup\n"),
    )
    p.add_argument("--course", default=None,
                   help="코스 키. 그 코스의 상태 파일에서 endpoint 이름을 읽습니다 "
                        "(extraction/classification/summarization/domain_qa/mm_extraction)")
    p.add_argument("--endpoint-name", default=None, metavar="NAME",
                   help="측정할 endpoint 이름. 주면 --course 없이도 됩니다")
    p.add_argument("--config", default=None, help="config.yaml 경로(기본 <repo>/config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="건수를 benchmark.dry_run_num_prompts 로 줄여 돕니다. "
                        "🔴 endpoint 는 실제로 호출합니다 — 호출 자체가 과금이 아닌 것은 아닙니다")
    p.add_argument("--state-dir", default=STATE_DIR, help=f"상태 파일 디렉토리(기본 {STATE_DIR})")
    p.add_argument("--state", default=None, metavar="PATH", help="상태 파일 경로를 통째로 지정")
    p.add_argument("--print-command", action="store_true",
                   help="실행할 sm-endpoint-bmt 명령만 출력하고 종료합니다(측정하지 않습니다)")
    args, passthrough = p.parse_known_args(argv)
    # argparse 는 `--` 를 남겨 주므로 여기서 떼어 낸다.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    try:
        cfg = load_config(args.config, dry_run=True if args.dry_run else None)
    except (ValueError, RuntimeError, FileNotFoundError, TypeError, AttributeError) as e:
        print(f"🔴 {e}")
        return 2

    try:
        endpoint_name = _resolve_endpoint(args, cfg)
    except ValueError as e:
        print(f"🔴 {e}")
        return 2

    bench_argv = _bench_argv(endpoint_name, cfg, passthrough,
                             result_dir=_result_dir(args, cfg))

    if args.print_command:
        print("python -m sagemaker_benchmark " + " ".join(bench_argv))
        return 0

    try:
        import sagemaker_benchmark
    except ImportError:
        print(f"🔴 {_INSTALL_HINT}")
        return 2

    from common.logging_utils import setup_logging

    setup_logging(cfg.runtime.log_level)
    print(f"endpoint  : {endpoint_name}")
    print(f"config    : {cfg.source_path or '(config.yaml 없음 — 내장 기본값)'}")
    print("tool      : sm-endpoint-bmt (vllm bench serve 지표 규약)")
    print("command   : python -m sagemaker_benchmark " + " ".join(bench_argv))
    print("=" * 78)

    # 🔴 sagemaker_benchmark.main() 은 sys.argv 를 직접 읽는다(인자를 받지 않는다).
    #    그래서 argv 를 바꿔 끼운 뒤 되돌린다 — subprocess 로 띄우면 같은 venv 를 쓰는지,
    #    예외를 어떻게 전달할지가 다시 문제가 되므로 같은 프로세스에서 부른다.
    saved = sys.argv
    sys.argv = ["sagemaker_benchmark"] + bench_argv
    try:
        result = sagemaker_benchmark.main()
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 130
    except (ValueError, RuntimeError) as e:
        print(f"\n🔴 벤치마크 실패: {e}")
        return 1
    finally:
        sys.argv = saved

    # 🔴 표가 찍혔다고 성공이 아니다. 요청이 전부 실패해도 도구는 0 이 채워진 표를 찍고 정상
    #    반환한다(실측: endpoint 이름이 틀려 4건 전부 ValidationError 인데 종료 코드 0). 그러면
    #    CI 와 스크립트가 이것을 성공으로 읽는다. completed/failed 를 보고 종료 코드를 정한다.
    completed = (result or {}).get("completed") or 0
    failed = (result or {}).get("failed") or 0
    if not completed:
        print(f"\n🔴 성공한 요청이 없습니다(실패 {failed}건).")
        print("   endpoint 이름·리전·상태(InService)를 확인하세요. 위 Error breakdown 이 원인입니다.")
        return 1
    if failed:
        print(f"\n⚠️  {failed}건이 실패했습니다 — 위 지표는 성공한 {completed}건만의 값입니다.")

    print("\n" + "=" * 78)
    print("🔴 endpoint 는 삭제 전까지 시간당 과금됩니다. 측정이 끝났으면:")
    print(f"   python pipelines/run_{args.course or '<course>'}.py --stages cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
