# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_4\exp_iter\iter_08\generation\reward_v8.py
- train_run: experiments/seed_4/exp_iter/iter_08/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: 107.082362
- mean_episode_length: 1000.000000
- min_eval_reward: 74.600503
- max_eval_reward: 128.657430

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.006183, nonzero_rate: 0.999210
- stability_penalty mean: -0.000781, abs_mean: 0.000781

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
