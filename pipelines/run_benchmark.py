"""배포된 엔드포인트의 TTFT, TPOT, ITL, E2EL을 측정합니다.

엔드포인트 이름은 직접 지정하거나 코스 상태 파일에서 읽습니다. `config.yaml`의 benchmark
설정을 `sm-endpoint-bmt` 인자로 변환하며, `--` 뒤의 인자는 해당 설정을 덮어씁니다.
"""
from __future__ import annotations

import os
import sys

# 파일을 직접 실행해도 pipelines 패키지를 찾도록 저장소 루트를 추가합니다.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse  # noqa: E402

from pipelines._common import (  # noqa: E402
    STATE_DIR,
    StateStore,
    _region,
    load_courses,
)
from pipelines._config import load_config  # noqa: E402

_INSTALL_HINT = (
    "sm-endpoint-bmt 를 import 할 수 없습니다.\n"
    f"  현재 파이썬: {sys.version.split()[0]}\n"
    "  코어 의존성이므로 동기화하면 설치됩니다:  uv sync\n"
    "  리포: https://github.com/daekeun-ml/sm-endpoint-bmt"
)


def _resolve_endpoint(args, cfg) -> str:
    """엔드포인트 이름을 인자 또는 코스 상태 파일에서 찾습니다."""
    if args.endpoint_name:
        return args.endpoint_name
    if not args.course:
        raise ValueError(
            "--endpoint-name 또는 --course 중 하나는 필요합니다.\n"
            "  --course 를 주면 그 코스의 deploy 가 상태 파일에 남긴 endpoint 를 씁니다."
        )
    courses = load_courses()
    if args.course not in courses:
        raise ValueError(f"알 수 없는 코스 {args.course!r}. 허용값: {list(courses)}")
    state = StateStore(args.course, state_dir=args.state_dir,
                       dry_run=cfg.runtime.dry_run, path=args.state)
    name = state.get("endpoint_name")
    if not name:
        raise ValueError(
            f"상태 파일에 endpoint_name이 없습니다({state.path}).\n"
            f"먼저 배포하세요: python pipelines/run_{args.course}.py --stages deploy\n"
            "기존 엔드포인트는 --endpoint-name으로 직접 지정할 수 있습니다."
        )
    return name


def _result_dir(args, cfg) -> str | None:
    """결과 JSON을 코스 데이터 디렉터리 또는 bench_results에 저장합니다."""
    if not args.course:
        return os.path.join(_REPO, "bench_results")
    from pipelines._common import data_dir, load_courses

    courses = load_courses()
    if args.course not in courses:
        return os.path.join(_REPO, "bench_results")
    return data_dir(courses[args.course], cfg)


def _diagnose_all_failed(endpoint_name: str, region: str) -> None:
    """모든 요청이 실패하면 엔드포인트 이름과 리전 불일치를 확인합니다."""
    import boto3

    print(f"   요청 리전: {region}, 엔드포인트: {endpoint_name}")
    try:
        boto3.client("sagemaker", region_name=region).describe_endpoint(
            EndpointName=endpoint_name)
        print("   이 리전에 엔드포인트가 있습니다. 상태와 CloudWatch 로그를 확인하세요.")
        return
    except Exception:  # noqa: BLE001. 존재하지 않거나 조회 권한이 없을 수 있습니다.
        print(f"   {region}에서 엔드포인트를 찾지 못했습니다.")

    # 흔히 사용하는 리전만 확인합니다.
    for other in ("us-west-2", "us-east-1", "ap-northeast-2", "eu-west-1"):
        if other == region:
            continue
        try:
            st = boto3.client("sagemaker", region_name=other).describe_endpoint(
                EndpointName=endpoint_name)["EndpointStatus"]
        except Exception:                  # noqa: BLE001
            continue
        print(f"\n   {other}에서 찾았습니다(상태 {st}). AWS_REGION을 맞추세요:")
        print(f"        AWS_REGION={other} python pipelines/run_benchmark.py "
              f"--endpoint-name {endpoint_name}")
        print("   셸의 AWS_REGION은 .env보다 우선합니다.")
        print("   셸 값이 잘못됐다면 `unset AWS_REGION` 후 다시 실행하세요.")
        return

    print("   위 오류 내역에서 이름, InService 상태, 자격증명을 확인하세요.")


def _bench_argv(endpoint_name: str, cfg, passthrough: list[str], region: str,
                result_dir: str | None = None) -> list[str]:
    """benchmark 설정을 sm-endpoint-bmt 인자로 변환합니다."""
    b = cfg.benchmark
    n = b.dry_run_num_prompts if cfg.runtime.dry_run else b.num_prompts
    # 동시 요청 수는 전체 요청 수를 넘지 않도록 조정합니다.
    conc = min(b.max_concurrency, n)
    if conc < b.max_concurrency:
        print(f"max_concurrency를 {b.max_concurrency}에서 {conc}로 낮춥니다. 요청 수: {n}")
    argv = [
        "--endpoint-name", endpoint_name,
        # load_config()가 확정한 리전을 사용합니다.
        "--region", region,
        # 모든 서빙 엔진이 OpenAI chat messages 스키마를 사용합니다.
        "--endpoint-type", "openai-chat",
        "--dataset-name", "random",
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--random-input-len", str(b.input_len),
        "--random-output-len", str(b.output_len),
        "--num-warmups", str(b.num_warmups),
        # 평균과 꼬리 지연을 함께 확인합니다.
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
        description="배포된 엔드포인트의 지연과 처리량 측정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python pipelines/run_benchmark.py --course extraction\n"
            "  python pipelines/run_benchmark.py --endpoint-name my-endpoint\n"
            "  python pipelines/run_benchmark.py --course extraction --dry-run\n"
            "\n"
            "`--` 뒤의 인자는 sm-endpoint-bmt에 전달되며 config.yaml 값을 덮습니다:\n"
            "  ... --course extraction -- --num-prompts 500 --max-concurrency 32\n"
            "  ... --course extraction -- --goodput ttft:200 tpot:50\n"
            "  ... --course extraction -- --ramp-up-strategy linear "
            "--ramp-up-start-rps 1 --ramp-up-end-rps 20\n"
            "  전체 옵션: python -m sagemaker_benchmark --help\n"
            "\n"
            "설정: config.yaml의 benchmark 섹션\n"
            "엔드포인트는 삭제할 때까지 시간당 과금됩니다. 측정이 끝나면:\n"
            "     python pipelines/run_<course>.py --stages cleanup\n"),
    )
    p.add_argument("--course", default=None,
                   help="코스 상태 파일에서 엔드포인트 이름을 읽고 결과를 해당 data/에 저장합니다 "
                        "(extraction/classification/summarization/domain_qa/mm_extraction)")
    p.add_argument("--endpoint-name", default=None, metavar="NAME",
                   help="측정할 엔드포인트 이름을 직접 지정합니다 "
                        "(--course 없이 이것만으로 충분)")
    p.add_argument("--config", default=None, help="config.yaml 경로(기본 <repo>/config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="건수를 benchmark.dry_run_num_prompts로 줄입니다. 엔드포인트는 실제로 호출합니다")
    p.add_argument("--state-dir", default=STATE_DIR, help=f"상태 파일 디렉토리(기본 {STATE_DIR})")
    p.add_argument("--state", default=None, metavar="PATH", help="상태 파일 경로를 통째로 지정")
    p.add_argument("--print-command", action="store_true",
                   help="실행할 sm-endpoint-bmt 명령만 출력하고 종료합니다(측정하지 않습니다)")
    args, passthrough = p.parse_known_args(argv)
    # argparse가 남긴 구분자 `--`를 제거합니다.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    try:
        cfg = load_config(args.config, dry_run=True if args.dry_run else None)
    except (ValueError, RuntimeError, FileNotFoundError, TypeError, AttributeError) as e:
        print(e)
        return 2

    try:
        endpoint_name = _resolve_endpoint(args, cfg)
    except ValueError as e:
        print(e)
        return 2

    region = _region()
    bench_argv = _bench_argv(endpoint_name, cfg, passthrough, region,
                             result_dir=_result_dir(args, cfg))

    if args.print_command:
        print("python -m sagemaker_benchmark " + " ".join(bench_argv))
        return 0

    try:
        import sagemaker_benchmark
    except ImportError:
        print(_INSTALL_HINT)
        return 2

    from common.logging_utils import setup_logging

    setup_logging(cfg.runtime.log_level)
    print(f"endpoint  : {endpoint_name}")
    print(f"config    : {cfg.source_path or '(config.yaml 없음, 내장 기본값 사용)'}")
    print("tool      : sm-endpoint-bmt (vllm bench serve 지표 규약)")
    print("command   : python -m sagemaker_benchmark " + " ".join(bench_argv))
    print("=" * 78)

    # sagemaker_benchmark.main()이 sys.argv를 읽으므로 호출 동안만 값을 바꿉니다.
    saved = sys.argv
    sys.argv = ["sagemaker_benchmark"] + bench_argv
    try:
        result = sagemaker_benchmark.main()
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 130
    except (ValueError, RuntimeError) as e:
        print(f"\n벤치마크 실패: {e}")
        return 1
    finally:
        sys.argv = saved

    # completed와 failed를 확인해 실제 성공 여부를 판단합니다.
    completed = (result or {}).get("completed") or 0
    failed = (result or {}).get("failed") or 0
    if not completed:
        print(f"\n성공한 요청이 없습니다. 실패: {failed}건")
        _diagnose_all_failed(endpoint_name, region)
        return 1
    if failed:
        print(f"\n요청 {failed}건이 실패했습니다. 지표는 성공한 {completed}건만 포함합니다.")

    print("\n" + "=" * 78)
    print("엔드포인트는 삭제할 때까지 시간당 과금됩니다. 측정이 끝나면 정리하세요:")
    print(f"   python pipelines/run_{args.course or '<course>'}.py --stages cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
