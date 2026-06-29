import argparse
from pathlib import Path
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--reward", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reward_fn = load_reward_function(args.reward)
    train_cfg = cfg["training"]
    env = RewardOverrideWrapper(gym.make(train_cfg["runner_env_id"]), reward_fn, train_cfg.get("max_training_steps_for_progress", train_cfg["total_timesteps"]))
    model = PPO("MlpPolicy", env, verbose=1, device=train_cfg.get("device", "auto"))
    model.learn(total_timesteps=int(train_cfg["total_timesteps"]))

if __name__ == "__main__":
    main()
