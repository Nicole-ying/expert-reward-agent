import argparse
import subprocess
from pathlib import Path

from .common import load_config


def run_cmd(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pad_iter(i):
    return f"{i:02d}"


def validate_rounds(rounds):
    if rounds < 1:
        raise ValueError("iteration.total_rounds must be >= 1")


def maybe_reset_memory(memory_path, reset):
    p = Path(memory_path)
    if reset and p.exists():
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)


def build_paths(cfg, prefix, iteration_index, seed):
    iter_pad = pad_iter(iteration_index)
    seed_dir = f"seed_{seed}"
    run_root = Path(cfg["experiment"]["run_root"])
    experiment_root = Path(cfg.get("iteration", {}).get("experiment_root", str(run_root / "experiments"))) / seed_dir
    iter_root = experiment_root / prefix / f"iter_{iter_pad}"
    gen_run_name = f"experiments/{seed_dir}/{prefix}/iter_{iter_pad}/generation"
    train_run_name = f"experiments/{seed_dir}/{prefix}/iter_{iter_pad}/training"
    train_dir = run_root / "training_runs" / train_run_name
    context_path = iter_root / "iteration_context.md"
    return {
        "iter_pad": iter_pad,
        "seed_dir": seed_dir,
        "iter_root": iter_root,
        "gen_run_name": gen_run_name,
        "train_run_name": train_run_name,
        "train_dir": train_dir,
        "context_path": context_path,
    }


def reward_path_for(cfg, gen_run_name, version):
    return Path(cfg["experiment"]["run_root"]) / gen_run_name / f"reward_v{version}.py"


def validation_path_for(cfg, gen_run_name, version):
    return Path(cfg["experiment"]["run_root"]) / gen_run_name / "validations" / f"reward_v{version}.validation.json"


def check_reward_valid(cfg, gen_run_name, version, stop_on_invalid):
    import json
    path = validation_path_for(cfg, gen_run_name, version)
    if not path.exists():
        raise FileNotFoundError(f"Missing validation file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if stop_on_invalid and not data.get("valid", False):
        raise RuntimeError(f"Reward v{version} failed validation: {path}")


def run_iterative_experiment(config_path, prefix=None, rounds=None, total_timesteps=None, eval_episodes=None, mock=None, seed=0):
    cfg = load_config(config_path)
    iter_cfg = cfg.get("iteration", {})
    train_cfg = cfg.get("training", {})
    rag_cfg = cfg.get("rag", {})

    prefix = prefix or iter_cfg.get("experiment_prefix", "exp_iter")
    rounds = int(rounds if rounds is not None else iter_cfg.get("total_rounds", 3))
    validate_rounds(rounds)

    total_timesteps = int(total_timesteps if total_timesteps is not None else train_cfg.get("total_timesteps", 1000000))
    eval_episodes = int(eval_episodes if eval_episodes is not None else train_cfg.get("eval_episodes", 10))
    use_mock = bool(iter_cfg.get("use_mock_llm", False) if mock is None else mock)
    stop_on_invalid = bool(iter_cfg.get("stop_on_invalid_reward", True))
    cards_top_k = int(iter_cfg.get("cards_top_k", 4))

    memory_path = iter_cfg.get("memory_path", "runs/env_001/memory/reward_memory.md")
    base_memory = Path(memory_path)
    memory_path = str(base_memory.parent / f"seed_{seed}" / base_memory.name)
    cards_path = rag_cfg.get("reward_misalignment_cards_path", "knowledge_base/iteration/reward_misalignment_cards.md")
    maybe_reset_memory(memory_path, bool(iter_cfg.get("reset_memory_at_start", True)))

    print("=" * 60)
    print("Config-driven iterative reward experiment")
    print("=" * 60)
    print(f"config          : {config_path}")
    print(f"prefix          : {prefix}")
    print(f"seed            : {seed}")
    print(f"rounds          : {rounds}")
    print(f"total_timesteps : {total_timesteps}")
    print(f"eval_episodes   : {eval_episodes}")
    print(f"memory_path     : {memory_path}")
    print(f"cards_path      : {cards_path}")
    print(f"cards_top_k     : {cards_top_k}")
    print(f"mock_llm        : {use_mock}")

    previous_reward = None

    for iteration_index in range(1, rounds + 1):
        version = iteration_index
        paths = build_paths(cfg, prefix, iteration_index, seed)
        mock_args = ["--mock"] if use_mock else []

        print("\n" + "-" * 60)
        print(f"Iteration {iteration_index}/{rounds}")
        print("-" * 60)

        if iteration_index == 1:
            run_cmd([
                "python", "-m", "pipeline.run_direct_generation_pipeline",
                "--config", config_path,
                "--run-name", paths["gen_run_name"],
                *mock_args,
            ])
            current_reward = reward_path_for(cfg, paths["gen_run_name"], 1)
        else:
            prev_paths = build_paths(cfg, prefix, iteration_index - 1, seed)
            run_cmd([
                "python", "-m", "pipeline.run_04_build_iteration_context",
                "--train-run-dir", str(prev_paths["train_dir"]),
                "--memory", memory_path,
                "--cards", cards_path,
                "--top-k", str(cards_top_k),
                "--out", str(paths["context_path"]),
            ])
            run_cmd([
                "python", "-m", "pipeline.run_05_reward_revision",
                "--config", config_path,
                "--previous-reward", str(previous_reward),
                "--iteration-context", str(paths["context_path"]),
                "--out-run-name", paths["gen_run_name"],
                "--reward-version", f"v{version}",
                *mock_args,
            ])
            current_reward = reward_path_for(cfg, paths["gen_run_name"], version)

        check_reward_valid(cfg, paths["gen_run_name"], version, stop_on_invalid)

        run_cmd([
            "python", "-m", "training.train_sb3_wrapper",
            "--config", config_path,
            "--reward", str(current_reward),
            "--run-name", paths["train_run_name"],
            "--total-timesteps", str(total_timesteps),
            "--eval-episodes", str(eval_episodes),
            "--seed", str(seed),
        ])

        run_cmd([
            "python", "-m", "pipeline.run_06_update_reward_memory",
            "--iter", str(iteration_index),
            "--train-run-dir", str(paths["train_dir"]),
            "--memory", memory_path,
        ])

        previous_reward = current_reward

    print("\n" + "=" * 60)
    print("Done. Key files")
    print("=" * 60)
    for iteration_index in range(1, rounds + 1):
        paths = build_paths(cfg, prefix, iteration_index, seed)
        version = iteration_index
        print(f"iter_{paths['iter_pad']}:")
        print(f"  reward   : {reward_path_for(cfg, paths['gen_run_name'], version)}")
        print(f"  feedback : {paths['train_dir'] / 'training_feedback.md'}")
        if iteration_index > 1:
            print(f"  context  : {paths['context_path']}")
    print(f"memory: {memory_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    mock = True if args.mock else None
    run_iterative_experiment(
        config_path=args.config,
        prefix=args.prefix,
        rounds=args.rounds,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        mock=mock,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
