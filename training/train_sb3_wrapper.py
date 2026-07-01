import argparse
import json
from pathlib import Path
from statistics import mean
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function


class RewardComponentStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.stats = {}
        self.reward_error_count_max = 0
        self.steps_seen = 0

    def _update_value(self, name, value):
        try:
            v = float(value)
        except Exception:
            return
        item = self.stats.setdefault(name, {
            "count": 0,
            "sum": 0.0,
            "abs_sum": 0.0,
            "nonzero_count": 0,
            "min": v,
            "max": v,
        })
        item["count"] += 1
        item["sum"] += v
        item["abs_sum"] += abs(v)
        if abs(v) > 1e-12:
            item["nonzero_count"] += 1
        item["min"] = min(item["min"], v)
        item["max"] = max(item["max"], v)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        self.steps_seen += len(infos)
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "generated_reward" in info:
                self._update_value("generated_reward", info["generated_reward"])
            if "original_env_reward" in info:
                self._update_value("original_env_reward", info["original_env_reward"])
            if "reward_error_count" in info:
                try:
                    self.reward_error_count_max = max(self.reward_error_count_max, int(info["reward_error_count"]))
                except Exception:
                    pass
            terms = info.get("reward_terms", {})
            if isinstance(terms, dict):
                for key, value in terms.items():
                    self._update_value(f"component.{key}", value)
        return True

    def summary(self):
        out = {}
        for name, item in sorted(self.stats.items()):
            count = max(1, item["count"])
            out[name] = {
                "count": item["count"],
                "mean": item["sum"] / count,
                "abs_mean": item["abs_sum"] / count,
                "nonzero_rate": item["nonzero_count"] / count,
                "min": item["min"],
                "max": item["max"],
            }
        return {
            "steps_seen": self.steps_seen,
            "reward_error_count_max": self.reward_error_count_max,
            "component_stats": out,
        }


def _make_env(env_id, reward_fn, max_progress_steps, seed, rank, monitor_dir, reward_clip, error_fallback):
    """Module-level env factory for SubprocVecEnv (must be picklable)."""
    import gymnasium as gym
    from training.reward_wrapper import RewardOverrideWrapper
    from stable_baselines3.common.monitor import Monitor
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


def build_env_fns(env_id, reward_fn, max_progress_steps, seed, n_envs, monitor_dir, reward_clip, error_fallback):
    """Return list of env factory functions for SubprocVecEnv."""
    from functools import partial
    fns = []
    for rank in range(n_envs):
        fns.append(partial(_make_env, env_id, reward_fn, max_progress_steps, seed, rank, str(monitor_dir), reward_clip, error_fallback))
    return fns


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


def _fmt_float(value):
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_eval_result_md(path, eval_result):
    lines = []
    lines.append("# External Evaluation Result")
    lines.append("")
    lines.append("Evaluation uses the original environment reward, not the generated training reward.")
    lines.append("")
    lines.append(f"- eval_episodes: {eval_result['eval_episodes']}")
    lines.append(f"- mean_eval_reward: {_fmt_float(eval_result['mean_eval_reward'])}")
    lines.append(f"- mean_episode_length: {_fmt_float(eval_result['mean_episode_length'])}")
    lines.append(f"- min_eval_reward: {_fmt_float(eval_result['min_eval_reward'])}")
    lines.append(f"- max_eval_reward: {_fmt_float(eval_result['max_eval_reward'])}")
    lines.append("")
    lines.append("## Episodes")
    lines.append("")
    lines.append("| episode | reward | length |")
    lines.append("|---:|---:|---:|")
    for i, (reward, length) in enumerate(zip(eval_result["episode_rewards"], eval_result["episode_lengths"])):
        lines.append(f"| {i} | {_fmt_float(reward)} | {length} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_component_stats_md(path, component_summary):
    lines = []
    lines.append("# Reward Component Training Statistics")
    lines.append("")
    lines.append(f"- steps_seen: {component_summary['steps_seen']}")
    lines.append(f"- reward_error_count_max: {component_summary['reward_error_count_max']}")
    lines.append("")
    lines.append("| name | mean | abs_mean | nonzero_rate | min | max | count |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in component_summary["component_stats"].items():
        lines.append(
            f"| {name} | {_fmt_float(item['mean'])} | {_fmt_float(item['abs_mean'])} | "
            f"{_fmt_float(item['nonzero_rate'])} | {_fmt_float(item['min'])} | {_fmt_float(item['max'])} | {item['count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_feedback_md(path, summary, eval_result, component_summary):
    stats = component_summary.get("component_stats", {})
    lines = []
    lines.append("# Training Feedback")
    lines.append("")
    lines.append("## Training outcome")
    lines.append(f"score={_fmt_float(eval_result['mean_eval_reward'])}, len={_fmt_float(eval_result['mean_episode_length'])}, errors={component_summary['reward_error_count_max']}")
    lines.append("")
    lines.append("## Component evidence")
    lines.append("")
    lines.append("| component | mean | abs_mean | nonzero_rate | min | max |")
    lines.append("|-----------|------|----------|-------------|-----|-----|")
    for name in sorted(stats.keys()):
        if name.startswith("component.original_env"):
            continue
        item = stats[name]
        short = name.replace("component.", "", 1)
        lines.append(
            f"| {short} | {_fmt_float(item.get('mean'))} | {_fmt_float(item.get('abs_mean'))} | "
            f"{_fmt_float(item.get('nonzero_rate'))} | {_fmt_float(item.get('min'))} | {_fmt_float(item.get('max'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--reward", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=float, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = dict(cfg["training"])
    reward_fn = load_reward_function(args.reward)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(train_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0)))
    set_random_seed(seed)

    total_timesteps = int(float(args.total_timesteps if args.total_timesteps is not None else train_cfg.get("total_timesteps", 100000)))
    n_envs = int(train_cfg.get("n_envs", 1))
    run_name = args.run_name or train_cfg.get("run_name", "ppo_reward_run")
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(train_cfg.get("save_dir", "runs/env_001/training_runs")) / run_name
    monitor_dir = save_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    reward_clip = train_cfg.get("reward_clip", 20.0)
    error_fallback = train_cfg.get("error_fallback", "zero")
    max_progress_steps = int(train_cfg.get("max_training_steps_for_progress", total_timesteps))

    env_fns = build_env_fns(
        train_cfg["runner_env_id"],
        reward_fn,
        max_progress_steps,
        seed,
        n_envs,
        monitor_dir,
        reward_clip,
        error_fallback,
    )
    env = SubprocVecEnv(env_fns)

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

    component_callback = RewardComponentStatsCallback()
    model = PPO(**ppo_args)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=component_callback)
    model.save(str(save_dir / "model.zip"))
    env.close()

    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.get("eval_episodes", 10))
    eval_result = evaluate_model_on_original_env(
        model,
        train_cfg["runner_env_id"],
        eval_episodes=eval_episodes,
        seed=seed + 10000,
    )
    component_summary = component_callback.summary()

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
        "component_summary": component_summary,
    }
    (save_dir / "train_config_used.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (save_dir / "eval_result.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_eval_result_md(save_dir / "eval_result.md", eval_result)
    write_component_stats_md(save_dir / "component_stats.md", component_summary)
    write_training_feedback_md(save_dir / "training_feedback.md", summary, eval_result, component_summary)

    print(f"Training finished. Model saved to: {save_dir / 'model.zip'}")
    print(f"Monitor logs: {monitor_dir}")
    print(f"External eval mean reward: {eval_result['mean_eval_reward']:.3f}")
    print(f"LLM feedback file: {save_dir / 'training_feedback.md'}")


if __name__ == "__main__":
    main()
