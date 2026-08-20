# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_5\exp_iter\iter_09\generation\reward_v9.py
- train_run: experiments/seed_5/exp_iter/iter_09/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: 219.676378
- mean_episode_length: 511.200000
- min_eval_reward: 163.808548
- max_eval_reward: 251.849040

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.031382, nonzero_rate: 0.999866
- stability_penalty mean: -0.003188, abs_mean: 0.003188

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
