import argparse
from pathlib import Path
from .common import load_config, read_text, write_text
from rag.direct_context_builder import build_expert_reward_context


def run(config_path, run_name):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / run_name
    if cfg.get("ablation", {}).get("disable_expert_rag", False):
        write_text(
            run_dir / "expert_reward_context.md",
            "# Expert Schema Disabled\n\n"
            "This run is the w/o Expert Schema ablation. Design from environment facts only; "
            "do not use the fixed expert task templates or formula-operator library.\n",
        )
        print(run_dir / "expert_reward_context.md")
        return
    env_md = read_text(run_dir / "environment_card.md")
    route_id, expert_md = build_expert_reward_context(
        env_md,
        cfg.get("rag", {}).get("generation_chunks_path"),
        # Fixed Expert Schema is intentionally kept complete; do not truncate it
        # with legacy RAG context length settings.
        max_chars=10_000_000,
    )
    write_text(run_dir / "expert_reward_context.md", expert_md)
    print(run_dir / "expert_reward_context.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default="mock_run")
    args = ap.parse_args()
    run(args.config, args.run_name)
