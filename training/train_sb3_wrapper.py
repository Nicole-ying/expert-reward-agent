import argparse
from pathlib import Path
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function


def build_env(env_id, reward_fn, max_progress_steps, seed, rank, monitor_dir):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env = RewardOverrideWrapper(env, reward_fn, max_training_steps_for_progress=max_progress_steps)
        env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{rank}.csv"))
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--reward", required=True)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    reward_fn = load_reward_function(args.reward)

    seed = int(train_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0)))
    set_random_seed(seed)

    total_timesteps = int(float(train_cfg.get("total_timesteps", 100000)))
    n_envs = int(train_cfg.get("n_envs", 1))
    run_name = args.run_name or train_cfg.get("run_name", "ppo_reward_run")
    save_dir = Path(train_cfg.get("save_dir", "runs/env_001/training_runs")) / run_name
    monitor_dir = save_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([
        build_env(
            train_cfg["runner_env_id"],
            reward_fn,
            int(train_cfg.get("max_training_steps_for_progress", total_timesteps)),
            seed,
            i,
            monitor_dir,
        )
        for i in range(n_envs)
    ])

    ppo_args = {
        "policy": train_cfg.get("policy", "MlpPolicy"),
        "env": env,
        "verbose": int(train_cfg.get("verbose", 1)),
        "device": train_cfg.get("device", "auto"),
        "seed": seed,
    }
    for key in ["n_steps", "batch_size", "gae_lambda", "gamma", "n_epochs", "ent_coef", "learning_rate", "clip_range", "vf_coef", "max_grad_norm"]:
        if key in train_cfg:
            ppo_args[key] = train_cfg[key]
    if train_cfg.get("tensorboard_log"):
        ppo_args["tensorboard_log"] = train_cfg["tensorboard_log"]

    model = PPO(**ppo_args)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)
    model.save(str(save_dir / "model.zip"))
    env.close()

    summary = {
        "runner_env_id": train_cfg["runner_env_id"],
        "reward_path": args.reward,
        "run_name": run_name,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "ppo_args": {k: str(v) for k, v in ppo_args.items() if k != "env"},
        "model_path": str(save_dir / "model.zip"),
        "monitor_dir": str(monitor_dir),
        "tensorboard_log": train_cfg.get("tensorboard_log"),
    }
    (save_dir / "train_config_used.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Training finished. Model saved to: {save_dir / 'model.zip'}")
    print(f"Monitor logs: {monitor_dir}")


if __name__ == "__main__":
    main()
