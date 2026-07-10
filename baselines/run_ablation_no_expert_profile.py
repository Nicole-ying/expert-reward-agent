#!/usr/bin/env python3
"""Ablation: sections 1-8 only (no expert profile), 5 seeds x 1 iter each."""

import json, sys, time, os
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common import load_config
from pipeline.run_iterative_experiment import run_iterative_experiment

CONFIG = "configs/env002_ablation_no_expert_profile.yaml"
PREFIX = "ablation_no_expert_profile_v1"
NUM_SEEDS = 5
ROUNDS = 1

OUTPUT_DIR = Path(f"runs/env_002/{PREFIX}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        try: self.file.write(data)
        except ValueError: pass
        self.stdout.write(data)
    def flush(self):
        try: self.file.flush()
        except ValueError: pass
        self.stdout.flush()

tee = Tee(OUTPUT_DIR / "ablation.log")
sys.stdout = tee

start_time = datetime.now()
print("=" * 60)
print("ABLATION: No Expert Profile (sections 1-8 only)")
print(f"start: {start_time.isoformat()}")
print(f"config: {CONFIG}")
print(f"seeds: {NUM_SEEDS} x {ROUNDS} iter each")
print("=" * 60)
print()

results = {}

for seed in range(NUM_SEEDS):
    print(f"\n{'#'*60}")
    print(f" SEED {seed}  ({seed+1}/{NUM_SEEDS})")
    print(f"{'#'*60}")
    t0 = time.time()
    try:
        run_iterative_experiment(
            config_path=CONFIG,
            prefix=PREFIX,
            rounds=ROUNDS,
            seed=seed,
        )
    except Exception as e:
        print(f"SEED {seed} FAILED: {e}")

    # Read score from training_summary.json
    summary_path = OUTPUT_DIR / f"seed_{seed}" / "iter_01" / "training" / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        score = summary.get("external_eval", {}).get("mean_eval_reward", None)
        train_sec = summary.get("train_duration_sec", None)
    else:
        score = None
        train_sec = None

    elapsed = (time.time() - t0) / 60
    results[f"seed_{seed}"] = {
        "score": score,
        "train_min": round(elapsed, 1),
        "train_duration_sec": train_sec,
    }
    print(f"SEED {seed}: score={score}, time={elapsed:.1f} min")

end_time = datetime.now()
print(f"\n{'='*60}")
print(f"ABLATION RESULTS (No Expert Profile)")
print(f"{'='*60}")
print(f"{'seed':<8} {'score':>10}")
scores = []
for seed in range(NUM_SEEDS):
    s = results[f"seed_{seed}"]["score"]
    scores.append(s)
    print(f"{seed:<8} {s if s else 'FAILED':>10}")

valid = [s for s in scores if s is not None]
if valid:
    print(f"\nmean: {sum(valid)/len(valid):.1f}  |  min: {min(valid):.1f}  |  max: {max(valid):.1f}  |  std: {(sum((x-sum(valid)/len(valid))**2 for x in valid)/len(valid))**0.5:.1f}")
    print(f"solved >= 300: {sum(1 for s in valid if s >= 300)}/{len(valid)}")

print(f"\nstart: {start_time.isoformat()}")
print(f"end:   {end_time.isoformat()}")

json.dump({
    "experiment": "Ablation: No Expert Profile (sections 1-8 only)",
    "config": CONFIG,
    "start": start_time.isoformat(),
    "end": end_time.isoformat(),
    "seeds": results,
    "scores": scores,
}, open(OUTPUT_DIR / "ablation_results.json", "w"), indent=2, ensure_ascii=False)

tee.file.close()
