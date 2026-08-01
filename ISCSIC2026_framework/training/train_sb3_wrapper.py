import argparse
import json
from pathlib import Path
from statistics import mean
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function
from pipeline.common import load_config


def _parse_schedule_value(value):
    """Convert RL Zoo's lin_<initial> notation to an SB3 schedule."""
    if not isinstance(value, str) or not value.startswith("lin_"):
        return value
    initial_value = float(value.removeprefix("lin_"))

    def linear_schedule(progress_remaining):
        return progress_remaining * initial_value

    return linear_schedule


class RewardComponentStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.stats = {}
        self.reward_error_count_max = 0
        self.steps_seen = 0
        self.episode_component_sums = []
        self._active_episode_component_sums = []

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
            "active_sum": 0.0,
            "active_abs_sum": 0.0,
            "min": v,
            "max": v,
        })
        item["count"] += 1
        item["sum"] += v
        item["abs_sum"] += abs(v)
        if abs(v) > 1e-12:
            item["nonzero_count"] += 1
            item["active_sum"] += v
            item["active_abs_sum"] += abs(v)
        item["min"] = min(item["min"], v)
        item["max"] = max(item["max"], v)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        while len(self._active_episode_component_sums) < len(infos):
            self._active_episode_component_sums.append({})
        self.steps_seen += len(infos)
        for env_index, info in enumerate(infos):
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
                    try:
                        episode_sums = self._active_episode_component_sums[env_index]
                        episode_sums[key] = episode_sums.get(key, 0.0) + float(value)
                    except (TypeError, ValueError):
                        pass
            if env_index < len(dones) and dones[env_index]:
                self.episode_component_sums.append(dict(self._active_episode_component_sums[env_index]))
                self._active_episode_component_sums[env_index] = {}
        return True

    def summary(self):
        out = {}
        for name, item in sorted(self.stats.items()):
            count = max(1, item["count"])
            active_count = max(1, item["nonzero_count"])
            out[name] = {
                "count": item["count"],
                "mean": item["sum"] / count,
                "abs_mean": item["abs_sum"] / count,
                "nonzero_rate": item["nonzero_count"] / count,
                "mean_when_active": item["active_sum"] / active_count,
                "abs_mean_when_active": item["active_abs_sum"] / active_count,
                "min": item["min"],
                "max": item["max"],
            }
        episode_component_sum_stats = {}
        component_names = sorted({name for episode in self.episode_component_sums for name in episode})
        for name in component_names:
            values = [episode.get(name, 0.0) for episode in self.episode_component_sums]
            episode_component_sum_stats[name] = {
                "count": len(values),
                "mean": sum(values) / max(1, len(values)),
                "abs_mean": sum(abs(value) for value in values) / max(1, len(values)),
                "min": min(values),
                "max": max(values),
            }
        return {
            "steps_seen": self.steps_seen,
            "reward_error_count_max": self.reward_error_count_max,
            "component_stats": out,
            "episode_component_sum_stats": episode_component_sum_stats,
        }


def _make_env(env_id, reward_fn, max_progress_steps, seed, rank, monitor_dir, reward_clip, error_fallback):
    """Module-level env factory for SubprocVecEnv (must be picklable)."""
    import gymnasium as gym
    from training.reward_wrapper import RewardOverrideWrapper
    from stable_baselines3.common.monitor import Monitor
    env = gym.make(env_id)
    env.reset(seed=seed + rank)
    env.action_space.seed(seed + rank)
    info_keywords = ()
    if reward_fn is not None:
        env = RewardOverrideWrapper(
            env,
            reward_fn,
            max_training_steps_for_progress=max_progress_steps,
            reward_clip=reward_clip,
            error_fallback=error_fallback,
        )
        info_keywords = ("original_env_reward", "generated_reward", "reward_error_count")
    env = Monitor(
        env,
        filename=str(Path(monitor_dir) / f"monitor_{rank}.csv"),
        info_keywords=info_keywords,
    )
    return env


def build_env_fns(env_id, reward_fn, max_progress_steps, seed, n_envs, monitor_dir, reward_clip, error_fallback):
    """Return list of env factory functions for SubprocVecEnv."""
    from functools import partial
    fns = []
    for rank in range(n_envs):
        fns.append(partial(_make_env, env_id, reward_fn, max_progress_steps, seed, rank, str(monitor_dir), reward_clip, error_fallback))
    return fns


def evaluate_model_on_original_env(
    model,
    env_id,
    eval_episodes,
    seed,
    eval_seed_offset=10000,
    reward_fn=None,
    observation_normalizer=None,
):
    """Evaluate using a fixed set of seeds for reproducibility and paired comparison.

    All evaluations use the SAME set of seeds (seed_offset + 0, seed_offset + 1, ..., seed_offset + N-1).
    This means parent and child evaluations with the same seed_offset are directly comparable
    on an episode-by-episode basis (paired comparison).

    Parameters
    ----------
    model : PPO model
    env_id : str, gym environment id
    eval_episodes : int, number of evaluation episodes
    seed : int, training seed (used for reproducibility logging, not directly for eval seeds)
    eval_seed_offset : int, base offset for eval seeds. The eval seeds used are:
                       eval_seed_offset + 0, eval_seed_offset + 1, ..., eval_seed_offset + eval_episodes - 1.
                       Parent and child should use the same eval_seed_offset for paired comparison.
    """
    env = gym.make(env_id)
    episode_rewards = []
    episode_lengths = []
    episode_terminated = []  # True = terminated (env-defined end), False = truncated (time limit)
    component_episode_sums = {}
    component_episode_abs_sums = {}
    component_active_steps = {}
    component_total_steps = 0
    component_error_count = 0
    for episode_id in range(eval_episodes):
        eval_seed = eval_seed_offset + episode_id
        obs, _ = env.reset(seed=eval_seed)
        done = False
        total_reward = 0.0
        length = 0
        was_terminated = False
        episode_component_sums = {}
        episode_component_abs_sums = {}
        while not done:
            policy_obs = obs
            if observation_normalizer is not None:
                # VecNormalize expects a batch even though this evaluator uses one raw env.
                policy_obs = observation_normalizer.normalize_obs(obs[None, ...])[0]
            action, _state = model.predict(policy_obs, deterministic=True)
            previous_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
            if terminated:
                was_terminated = True
            if reward_fn is not None:
                safe_info = dict(info or {})
                safe_info["terminated"] = bool(terminated)
                safe_info["truncated"] = bool(truncated)
                safe_info["done"] = done
                try:
                    reward_out = reward_fn(
                        previous_obs,
                        action,
                        obs,
                        reward,
                        safe_info,
                        training_progress=1.0,
                    )
                    terms = reward_out[1] if isinstance(reward_out, (tuple, list)) and len(reward_out) == 2 else {}
                    if not isinstance(terms, dict):
                        terms = {}
                except Exception:
                    component_error_count += 1
                    terms = {}
                component_total_steps += 1
                for name, value in terms.items():
                    if name in {"total_reward", "generated_reward", "original_env_reward"}:
                        continue
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    episode_component_sums[name] = episode_component_sums.get(name, 0.0) + numeric_value
                    episode_component_abs_sums[name] = episode_component_abs_sums.get(name, 0.0) + abs(numeric_value)
                    if abs(numeric_value) > 1e-12:
                        component_active_steps[name] = component_active_steps.get(name, 0) + 1
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        episode_terminated.append(was_terminated)
        known_names = set(component_episode_sums) | set(episode_component_sums)
        for name in known_names:
            component_episode_sums.setdefault(name, [0.0] * episode_id).append(
                episode_component_sums.get(name, 0.0)
            )
            component_episode_abs_sums.setdefault(name, [0.0] * episode_id).append(
                episode_component_abs_sums.get(name, 0.0)
            )
    env.close()
    component_evaluation = {}
    all_names = sorted(set(component_episode_sums) | set(component_episode_abs_sums))
    total_abs_contribution = sum(
        mean(component_episode_abs_sums[name]) for name in all_names
    )
    for name in all_names:
        sums = component_episode_sums[name]
        abs_sums = component_episode_abs_sums[name]
        sum_mean = mean(sums)
        abs_sum_mean = mean(abs_sums)
        component_evaluation[name] = {
            "episode_sum_mean": sum_mean,
            "episode_abs_sum_mean": abs_sum_mean,
            "signed_contribution_share": (
                sum_mean / total_abs_contribution if total_abs_contribution > 1e-12 else 0.0
            ),
            "magnitude_share": (
                abs_sum_mean / total_abs_contribution if total_abs_contribution > 1e-12 else 0.0
            ),
            "active_rate": (
                component_active_steps.get(name, 0) / component_total_steps
                if component_total_steps else 0.0
            ),
        }
    return {
        "eval_episodes": eval_episodes,
        "eval_seed_offset": eval_seed_offset,
        "eval_seeds": [eval_seed_offset + i for i in range(eval_episodes)],
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_terminated": episode_terminated,
        "mean_eval_reward": mean(episode_rewards) if episode_rewards else 0.0,
        "mean_episode_length": mean(episode_lengths) if episode_lengths else 0.0,
        "min_eval_reward": min(episode_rewards) if episode_rewards else 0.0,
        "max_eval_reward": max(episode_rewards) if episode_rewards else 0.0,
        "termination_breakdown": {
            "terminated": sum(1 for t in episode_terminated if t),
            "truncated": sum(1 for t in episode_terminated if not t),
        },
        "final_policy_component_evaluation": component_evaluation,
        "final_policy_component_error_count": component_error_count,
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
    lines.append("All evaluations use the same fixed seed set for reproducible paired comparison.")
    lines.append("")
    lines.append(f"- eval_episodes: {eval_result['eval_episodes']}")
    lines.append(f"- eval_seed_offset: {eval_result.get('eval_seed_offset', 'N/A')}")
    lines.append(f"- mean_eval_reward: {_fmt_float(eval_result['mean_eval_reward'])}")
    lines.append(f"- mean_episode_length: {_fmt_float(eval_result['mean_episode_length'])}")
    lines.append(f"- min_eval_reward: {_fmt_float(eval_result['min_eval_reward'])}")
    lines.append(f"- max_eval_reward: {_fmt_float(eval_result['max_eval_reward'])}")
    tb = eval_result.get("termination_breakdown", {})
    if tb:
        lines.append(f"- termination: {tb.get('terminated', '?')} terminated, {tb.get('truncated', '?')} truncated")
    lines.append("")
    lines.append("## Episodes")
    lines.append("")
    term_flags = eval_result.get("episode_terminated", [])
    lines.append("| episode | eval_seed | reward | length | end |")
    lines.append("|---:|---:|---:|---:|---|")
    for i, (reward, length) in enumerate(zip(eval_result["episode_rewards"], eval_result["episode_lengths"])):
        end_type = "terminated" if (i < len(term_flags) and term_flags[i]) else "truncated"
        eval_seed = eval_result.get("eval_seeds", [])[i] if i < len(eval_result.get("eval_seeds", [])) else "?"
        lines.append(f"| {i} | {eval_seed} | {_fmt_float(reward)} | {length} | {end_type} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_component_stats_md(path, component_summary):
    lines = []
    lines.append("# Reward Component Training Statistics")
    lines.append("")
    lines.append(f"- steps_seen: {component_summary['steps_seen']}")
    lines.append(f"- reward_error_count_max: {component_summary['reward_error_count_max']}")
    lines.append("")
    lines.append("| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in component_summary["component_stats"].items():
        lines.append(
            f"| {name} | {_fmt_float(item['mean'])} | {_fmt_float(item['abs_mean'])} | "
            f"{_fmt_float(item['nonzero_rate'])} | {_fmt_float(item['mean_when_active'])} | "
            f"{_fmt_float(item['abs_mean_when_active'])} | {_fmt_float(item['min'])} | "
            f"{_fmt_float(item['max'])} | {item['count']} |"
        )
    episode_stats = component_summary.get("episode_component_sum_stats", {})
    if episode_stats:
        lines.extend(["", "## Per-episode component sums", "", "| component | mean | abs_mean | min | max | episodes |", "|---|---:|---:|---:|---:|---:|"])
        for name, item in episode_stats.items():
            lines.append(
                f"| {name} | {_fmt_float(item['mean'])} | {_fmt_float(item['abs_mean'])} | "
                f"{_fmt_float(item['min'])} | {_fmt_float(item['max'])} | {item['count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_feedback_legacy_md(path, summary, eval_result, component_summary):
    stats = component_summary.get("component_stats", {})
    lines = []
    lines.append("# Training Feedback")
    lines.append("")
    lines.append("## Training outcome")
    lines.append(f"score={_fmt_float(eval_result['mean_eval_reward'])}, len={_fmt_float(eval_result['mean_episode_length'])}, errors={component_summary['reward_error_count_max']}")
    lines.append("`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。")
    lines.append("")
    # Find progress reference: component whose short name starts with "progress"
    # The LLM is instructed to name its main learning signal "progress_reward".
    # Fallback: component with largest abs_mean (excluding summary/total/original).
    progress_ref = 1.0
    progress_name = "progress_reward"
    progress_found = False
    _skip_ref = {"generated_reward", "total_reward", "original_env_reward"}
    for name in sorted(stats.keys()):
        short = name.replace("component.", "", 1)
        if short in _skip_ref:
            continue
        if short.startswith("progress"):
            progress_ref = max(float(stats[name].get("abs_mean", 1.0)), 0.001)
            progress_name = short
            progress_found = True
            break
    # Fallback: if no progress-named component, use the one with largest abs_mean
    if not progress_found:
        best_abs = 0.0
        for name in sorted(stats.keys()):
            short = name.replace("component.", "", 1)
            if short in _skip_ref:
                continue
            val = abs(float(stats[name].get("abs_mean", 0.0)))
            if val > best_abs:
                best_abs = val
                progress_name = short
        progress_ref = max(best_abs, 0.001)

    lines.append("## Component evidence")
    lines.append("")
    lines.append(f"`ratio_to_{progress_name}` = mean_of_component / abs_mean_of_{progress_name}. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.")
    lines.append("")
    lines.append(f"| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_{progress_name} |")
    lines.append(f"|-----------|------|----------|-------------|------------------|--------------------------|")
    for name in sorted(stats.keys()):
        item = stats[name]
        short = name.replace("component.", "", 1)
        if short in {"generated_reward", "original_env_reward"}:
            continue
        ratio_val = float(item.get("mean", 0.0)) / progress_ref
        lines.append(
            f"| {short} | {_fmt_float(item.get('mean'))} | {_fmt_float(item.get('abs_mean'))} | "
            f"{_fmt_float(item.get('nonzero_rate'))} | {_fmt_float(item.get('mean_when_active'))} | "
            f"{_fmt_float(ratio_val)} |"
        )
    # Show original_env_reward for alignment reference
    orig = stats.get("original_env_reward") or stats.get("component.original_env_reward")
    if orig:
        orig_ratio = float(orig.get("mean", 0.0)) / progress_ref
        lines.append(f"| original_env_reward | {_fmt_float(orig.get('mean'))} | {_fmt_float(orig.get('abs_mean'))} | {_fmt_float(orig.get('nonzero_rate'))} | {_fmt_float(orig.get('mean_when_active'))} | {_fmt_float(orig_ratio)} |")
    lines.append("")
    lines.append(f"> `ratio_to_{progress_name}` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。")

    episode_stats = component_summary.get("episode_component_sum_stats", {})
    if episode_stats:
        lines.extend(["", "## Per-episode component contribution", "", "| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |", "|---|---:|---:|---:|---:|---:|"])
        for name, item in episode_stats.items():
            lines.append(
                f"| {name} | {_fmt_float(item.get('mean'))} | {_fmt_float(item.get('abs_mean'))} | "
                f"{_fmt_float(item.get('min'))} | {_fmt_float(item.get('max'))} | {item.get('count', 0)} |"
            )

    # Distribution summary
    mean_len = float(eval_result.get("mean_episode_length", 0))
    mean_score = float(eval_result.get("mean_eval_reward", 0))
    lines.append("")
    lines.append("## Distribution")
    early_term = "N/A"
    ep_rewards = eval_result.get("episode_rewards", [])
    ep_lengths = eval_result.get("episode_lengths", [])
    if ep_lengths and ep_rewards:
        early_count = sum(1 for i in range(len(ep_lengths)) if ep_lengths[i] < 150 and ep_rewards[i] < -50)
        early_term = f"{early_count}/{len(ep_lengths)} ({100*early_count/len(ep_lengths):.0f}%)"
    lines.append(f"- score: mean={_fmt_float(mean_score)}, min={_fmt_float(eval_result.get('min_eval_reward', 0))}, max={_fmt_float(eval_result.get('max_eval_reward', 0))}")
    lines.append(f"- episode_length: mean={_fmt_float(mean_len)}")
    lines.append(f"- early_terminal (<150 steps + score<-50): {early_term}")
    lines.append(f"- errors: {component_summary['reward_error_count_max']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_feedback_md(path, summary, eval_result, component_summary):
    """Write the compact evidence view consumed by the reflection agent."""
    lines = ["# Training Feedback", "", "## Final-policy outcome"]
    termination = eval_result.get("termination_breakdown", {})
    lines.append(
        f"score={_fmt_float(eval_result.get('mean_eval_reward', 0.0))}, "
        f"len={_fmt_float(eval_result.get('mean_episode_length', 0.0))}, "
        f"terminated={termination.get('terminated', 0)}/{eval_result.get('eval_episodes', 0)}, "
        f"truncated={termination.get('truncated', 0)}/{eval_result.get('eval_episodes', 0)}, "
        f"reward_errors={eval_result.get('final_policy_component_error_count', 0)}"
    )
    lines.append(
        f"score_range=[{_fmt_float(eval_result.get('min_eval_reward', 0.0))}, "
        f"{_fmt_float(eval_result.get('max_eval_reward', 0.0))}]"
    )

    component_eval = eval_result.get("final_policy_component_evaluation", {})
    visible_components = {
        name: item for name, item in component_eval.items()
        if name not in {"total_reward", "generated_reward", "original_env_reward"}
    }
    lines.extend([
        "",
        "## Final-policy reward composition",
        "",
        "These statistics come from the same fixed evaluation episodes as `score`. "
        "Shares describe observed reward composition, not causal influence.",
        "",
        "| component | episode_sum_mean | signed_share | magnitude_share | active_rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in sorted(
        visible_components.items(),
        key=lambda pair: float(pair[1].get("magnitude_share", 0.0)),
        reverse=True,
    ):
        lines.append(
            f"| {name} | {_fmt_float(item.get('episode_sum_mean'))} | "
            f"{100.0 * float(item.get('signed_contribution_share', 0.0)):.1f}% | "
            f"{100.0 * float(item.get('magnitude_share', 0.0)):.1f}% | "
            f"{100.0 * float(item.get('active_rate', 0.0)):.1f}% |"
        )
    if not visible_components:
        lines.append("| (no component data) | 0 | 0% | 0% | 0% |")

    episode_rewards = eval_result.get("episode_rewards", [])
    episode_lengths = eval_result.get("episode_lengths", [])
    early_count = sum(
        1 for reward, length in zip(episode_rewards, episode_lengths)
        if length < 150 and reward < -50
    )
    lines.extend([
        "",
        "## Evaluation distribution",
        f"- fixed_eval_seeds: {eval_result.get('eval_seed_offset', 0)}.."
        f"{eval_result.get('eval_seed_offset', 0) + max(0, eval_result.get('eval_episodes', 0) - 1)}",
        f"- early_terminal (<150 steps and score<-50): {early_count}/{len(episode_rewards)}",
        f"- training_reward_errors_max: {component_summary.get('reward_error_count_max', 0)}",
        "- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--reward", default=None)
    ap.add_argument("--use-original-reward", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=float, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-seed-offset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_cfg = dict(cfg["training"])
    if args.use_original_reward and args.reward:
        raise ValueError("Use either --reward or --use-original-reward, not both")
    if not args.use_original_reward and not args.reward:
        raise ValueError("--reward is required unless --use-original-reward is set")
    reward_fn = None if args.use_original_reward else load_reward_function(args.reward)

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
    normalize_obs = bool(train_cfg.get("normalize_obs", False))
    normalize_reward = bool(train_cfg.get("normalize_reward", False))
    vec_normalize = None
    if normalize_obs or normalize_reward:
        vec_normalize = VecNormalize(
            env,
            training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=float(train_cfg.get("clip_obs", 10.0)),
            clip_reward=float(train_cfg.get("clip_normalized_reward", 10.0)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            epsilon=float(train_cfg.get("normalize_epsilon", 1e-8)),
        )
        env = vec_normalize

    ppo_args = {
        "policy": train_cfg.get("policy", "MlpPolicy"),
        "env": env,
        "verbose": int(train_cfg.get("verbose", 1)),
        "device": train_cfg.get("device", "auto"),
        "seed": seed,
    }
    for key in ["n_steps", "batch_size", "gae_lambda", "gamma", "n_epochs", "ent_coef", "learning_rate", "clip_range", "vf_coef", "max_grad_norm"]:
        if key in train_cfg:
            value = train_cfg[key]
            if key in {"learning_rate", "clip_range"}:
                value = _parse_schedule_value(value)
            ppo_args[key] = value
    if train_cfg.get("policy_kwargs"):
        import torch.nn as nn
        policy_kwargs = dict(train_cfg["policy_kwargs"])
        activation_name = policy_kwargs.get("activation_fn")
        if isinstance(activation_name, str):
            activation_types = {
                "ReLU": nn.ReLU,
                "Tanh": nn.Tanh,
                "ELU": nn.ELU,
                "GELU": nn.GELU,
            }
            if activation_name not in activation_types:
                raise ValueError(f"Unsupported policy activation_fn: {activation_name}")
            policy_kwargs["activation_fn"] = activation_types[activation_name]
        ppo_args["policy_kwargs"] = policy_kwargs
    if train_cfg.get("tensorboard_log"):
        ppo_args["tensorboard_log"] = train_cfg["tensorboard_log"]

    component_callback = RewardComponentStatsCallback()
    model = PPO(**ppo_args)
    # Use short, path-safe name for tensorboard (Windows can't handle long nested names)
    tb_log_name = f"s{seed}_i{run_name.split('/')[-1].replace('training', 't')}"
    # Ensure tensorboard dir exists before SB3 tries to write
    if train_cfg.get("tensorboard_log"):
        from pathlib import Path as _Path
        _Path(train_cfg["tensorboard_log"]).mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=component_callback)
    model.save(str(save_dir / "model.zip"))
    vec_normalize_path = None
    if vec_normalize is not None:
        vec_normalize_path = save_dir / "vecnormalize.pkl"
        # Freeze running statistics. External evaluation still uses raw environment rewards.
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        vec_normalize.save(str(vec_normalize_path))

    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.get("eval_episodes", 20))
    eval_seed_offset = int(args.eval_seed_offset if args.eval_seed_offset is not None else train_cfg.get("eval_seed_offset", 10000))
    eval_result = evaluate_model_on_original_env(
        model,
        train_cfg["runner_env_id"],
        eval_episodes=eval_episodes,
        seed=seed,
        eval_seed_offset=eval_seed_offset,
        reward_fn=reward_fn,
        observation_normalizer=vec_normalize if normalize_obs else None,
    )
    env.close()
    component_summary = component_callback.summary()

    summary = {
        "runner_env_id": train_cfg["runner_env_id"],
        "reward_path": args.reward,
        "reward_source": "official_environment" if args.use_original_reward else "generated_function",
        "run_name": run_name,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "reward_clip": reward_clip,
        "error_fallback": error_fallback,
        "ppo_args": {k: str(v) for k, v in ppo_args.items() if k != "env"},
        "model_path": str(save_dir / "model.zip"),
        "monitor_dir": str(monitor_dir),
        "tensorboard_log": train_cfg.get("tensorboard_log"),
        "normalization": {
            "normalize_obs": normalize_obs,
            "normalize_reward": normalize_reward,
            "clip_obs": float(train_cfg.get("clip_obs", 10.0)),
            "clip_normalized_reward": float(train_cfg.get("clip_normalized_reward", 10.0)),
            "vecnormalize_path": str(vec_normalize_path) if vec_normalize_path else None,
        },
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
