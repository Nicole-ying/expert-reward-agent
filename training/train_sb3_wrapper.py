import argparse
import json
from pathlib import Path
from statistics import mean
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function


def build_env(env_id, reward_fn, max_progress_steps, seed, rank, monitor_dir, reward_clip, error_fallback):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env = RewardOverrideWrapper(
            env,
            reward_fn,
            max_training_steps_for_progress=max_progress_steps,
            reward_clip=reward_clip,
            error_fallback=error_fallback,
        )
        env = Monitor(
            env,
            filename=str(Path(monitor_dir) / f"monitor_{rank}.csv"),
            info_keywords=("original_env_reward", "generated_reward", "reward_error_count"),
        )
        return env
    return _init


def evaluate_model_on_original_env(model, env_id, eval_episodes, seed):
    env = gym.make(env_id)
    episode_rewards = []
    episode_lengths = []
    for episode_id in range(eval_episodes):
        obs, _ = env.reset(seed=seed + episode_id)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
    env.close()
    return {
        "eval_episodes": eval_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_eval_reward": mean(episode_rewards) if episode_rewards else 0.0,
        "mean_episode_length": mean(episode_lengths) if episode_lengths else 0.0,
        "min_eval_reward": min(episode_rewards) if episode_rewards else 0.0,
        "max_eval_reward": max(episode_rewards) if episode_rewards else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--reward", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=float, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = dict(cfg["training"])
    reward_fn = load_reward_function(args.reward)

    seed = int(train_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0)))
    set_random_seed(seed)

    total_timesteps = int(float(args.total_timesteps if args.total_timesteps is not None else train_cfg.get("total_timesteps", 100000)))
    n_envs = int(train_cfg.get("n_envs", 1))
    run_name = args.run_name or train_cfg.get("run_name", "ppo_reward_run")
    save_dir = Path(train_cfg.get("save_dir", "runs/env_001/training_runs")) / run_name
    monitor_dir = save_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    reward_clip = train_cfg.get("reward_clip", 20.0)
    error_fallback = train_cfg.get("error_fallback", "zero")
    max_progress_steps = int(train_cfg.get("max_training_steps_for_progress", total_timesteps))

    env = DummyVecEnv([
        build_env(
            train_cfg["runner_env_id"],
            reward_fn,
            max_progress_steps,
            seed,
            i,
            monitor_dir,
            reward_clip,
            error_fallback,
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

    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.get("eval_episodes", 10))
    eval_result = evaluate_model_on_original_env(
        model,
        train_cfg["runner_env_id"],
        eval_episodes=eval_episodes,
        seed=seed + 10000,
    )

    summary = {
        "runner_env_id": train_cfg["runner_env_id"],
        "reward_path": args.reward,
        "run_name": run_name,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "reward_clip": reward_clip,
        "error_fallback": error_fallback,
        "ppo_args": {k: str(v) for k, v in ppo_args.items() if k != "env"},
        "model_path": str(save_dir / "model.zip"),
        "monitor_dir": str(monitor_dir),
        "tensorboard_log": train_cfg.get("tensorboard_log"),
        "external_eval": eval_result,
    }
    (save_dir / "train_config_used.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (save_dir / "eval_result.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training finished. Model saved to: {save_dir / 'model.zip'}")
    print(f"Monitor logs: {monitor_dir}")
    print(f"External eval mean reward: {eval_result['mean_eval_reward']:.3f}")


if __name__ == "__main__":
    main()
