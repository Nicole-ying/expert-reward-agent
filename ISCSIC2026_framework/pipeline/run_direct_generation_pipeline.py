import argparse
import shutil
from pathlib import Path
from .common import load_config
from .run_01_environment_analyzer_md import run as run_env
from .run_02_build_expert_context import run as run_rag
from .run_03_direct_reward_generator import run as run_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default="mock_run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validation-retry", default=None)
    ap.add_argument("--reuse-context-from", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not args.validation_retry:
        use_expert_context = cfg.get("initial_generation", {}).get("use_expert_context", True)
        if args.reuse_context_from:
            source_dir = Path(args.reuse_context_from).resolve()
            target_dir = (Path(cfg["experiment"]["run_root"]) / args.run_name).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            source_card = source_dir / "environment_card.md"
            if not source_card.exists():
                raise FileNotFoundError(f"Reusable environment card not found: {source_card}")
            shutil.copy2(source_card, target_dir / "environment_card.md")
            (target_dir / "environment_card.provenance.md").write_text(
                "# Environment Card Provenance\n\n"
                f"Reused unchanged from `{source_card}`. Reward search restarts must not resample environment semantics.\n",
                encoding="utf-8",
            )
            if use_expert_context:
                source_expert = source_dir / "expert_reward_context.md"
                if source_expert.exists():
                    shutil.copy2(source_expert, target_dir / "expert_reward_context.md")
                else:
                    run_rag(args.config, args.run_name)
        else:
            run_env(args.config, args.run_name, mock=args.mock)
            if use_expert_context:
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
