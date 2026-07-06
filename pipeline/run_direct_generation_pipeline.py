import argparse
from .run_01_environment_analyzer_md import run as run_env
from .run_02_build_expert_context import run as run_rag
from .run_03_direct_reward_generator import run as run_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default="mock_run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validation-retry", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.validation_retry:
        run_env(args.config, args.run_name, mock=args.mock)
        run_rag(args.config, args.run_name)
    run_reward(
        args.config,
        args.run_name,
        mock=args.mock,
        seed=args.seed,
        validation_retry=args.validation_retry,
    )


if __name__ == "__main__":
    main()
