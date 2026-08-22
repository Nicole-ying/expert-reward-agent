"""pv5 experiment — direct Python runner, no subprocess nesting."""
import os, sys, time

# ── MUST be set before ANY imports that touch httpx/openai ──
os.environ["SSL_CERT_FILE"] = "C:/ProgramData/miniconda3/envs/eure/Lib/site-packages/certifi/cacert.pem"
os.environ["DEEPSEEK_API_KEY"] = "sk-4c875777051e4ec5b0c03ee4955aefdc"

CONFIG = "configs/env001_paper_v5.yaml"
PREFIX = "pv5"
ROUNDS = 10
TOTAL_TIMESTEPS = 1_000_000
EVAL_EPISODES = 20
FROZEN_CONTEXT = "runs/env_001/pv5_frozen_context"
SEEDS = [0, 1, 2, 3, 4]

from pipeline.run_iterative_experiment import run_iterative_experiment

def main():
    t_start = time.time()
    print(f"=== pv5 experiment ===")
    print(f"seeds: {SEEDS}")
    print(f"rounds: {ROUNDS}")
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"{'='*60}")
        print(f" SEED {seed}  ({i+1}/{len(SEEDS)})  —  {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        run_iterative_experiment(
            config_path=CONFIG,
            prefix=PREFIX,
            seed=seed,
            rounds=ROUNDS,
            total_timesteps=TOTAL_TIMESTEPS,
            eval_episodes=EVAL_EPISODES,
            frozen_context_dir=FROZEN_CONTEXT,
        )
        elapsed = time.time() - t0
        print(f" SEED {seed} DONE in {elapsed/60:.1f}min  —  {time.strftime('%H:%M:%S')}")
        print()

    total = time.time() - t_start
    print(f"=== ALL DONE in {total/60:.1f}min ===")

if __name__ == "__main__":
    main()
