import argparse
import json
import re
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def read_best_iteration(seed_dir: Path) -> int:
    summary = (seed_dir / "best" / "best_summary.md").read_text(encoding="utf-8")
    match = re.search(r"best_iter:\s*(\d+)", summary)
    if not match:
        raise ValueError(f"Cannot find best_iter in {seed_dir}")
    return int(match.group(1))


def run_episode(model: PPO, env_id: str, seed: int, capture: bool, frame_stride: int,
                observation_normalizer=None):
    env = gym.make(env_id, render_mode="rgb_array" if capture else None)
    obs, _ = env.reset(seed=seed)
    frames = []
    score = 0.0
    length = 0
    terminated = truncated = False
    if capture:
        frames.append(env.render())
    while not (terminated or truncated):
        policy_obs = obs
        if observation_normalizer is not None:
            policy_obs = observation_normalizer.normalize_obs(obs[None, ...])[0]
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        length += 1
        if capture and length % frame_stride == 0:
            frames.append(env.render())
    if capture and length % frame_stride:
        frames.append(env.render())
    env.close()
    return score, length, frames


def record_experiment(experiment_dir: Path, env_id: str, seed_count: int, eval_seed_offset: int,
                      candidate_episodes: int, frame_stride: int, fps: int):
    gif_dir = experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for policy_seed in range(seed_count):
        seed_dir = experiment_dir / f"seed_{policy_seed}"
        best_iter = read_best_iteration(seed_dir)
        model_path = seed_dir / f"iter_{best_iter:02d}" / "training" / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = PPO.load(model_path, device="cpu")
        observation_normalizer = None
        normalization_env = None
        training_summary_path = seed_dir / "best" / "best_training_summary.json"
        if training_summary_path.exists():
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            normalization = training_summary.get("normalization", {})
            if normalization.get("normalize_obs"):
                stats_path = Path(normalization["vecnormalize_path"])
                normalization_env = DummyVecEnv([lambda: gym.make(env_id)])
                observation_normalizer = VecNormalize.load(str(stats_path), normalization_env)
                observation_normalizer.training = False
                observation_normalizer.norm_reward = False

        candidates = []
        for episode_id in range(candidate_episodes):
            eval_seed = eval_seed_offset + episode_id
            score, length, _ = run_episode(
                model, env_id, eval_seed, False, frame_stride, observation_normalizer
            )
            candidates.append((score, length, eval_seed))
        score, length, selected_seed = max(candidates, key=lambda row: row[0])
        replay_score, replay_length, frames = run_episode(
            model, env_id, selected_seed, True, frame_stride, observation_normalizer
        )

        gif_path = gif_dir / f"seed_{policy_seed}_best_iter_{best_iter:02d}.gif"
        imageio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)
        record = {
            "policy_seed": policy_seed,
            "best_iter": best_iter,
            "model_path": str(model_path),
            "selection": f"best original-environment return among {candidate_episodes} fixed seeds",
            "candidate_seed_range": [eval_seed_offset, eval_seed_offset + candidate_episodes - 1],
            "selected_eval_seed": selected_seed,
            "original_env_score": replay_score,
            "episode_length": replay_length,
            "frame_stride": frame_stride,
            "fps": fps,
            "gif_path": str(gif_path),
        }
        manifest.append(record)
        print(f"saved {gif_path} score={replay_score:.3f} length={replay_length}")
        if observation_normalizer is not None:
            observation_normalizer.close()

    (gif_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed-count", required=True, type=int)
    parser.add_argument("--eval-seed-offset", type=int, default=10000)
    parser.add_argument("--candidate-episodes", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    record_experiment(
        args.experiment_dir,
        args.env_id,
        args.seed_count,
        args.eval_seed_offset,
        args.candidate_episodes,
        args.frame_stride,
        args.fps,
    )


if __name__ == "__main__":
    main()
