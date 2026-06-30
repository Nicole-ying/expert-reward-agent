# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\exp_fixed\seed_0\iter_08\generation\reward_v8.py
- train_run: exp_fixed/seed_0/iter_08/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: -18.550812
- mean_episode_length: 1000.000000
- min_eval_reward: -52.128165
- max_eval_reward: 12.658302

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.776811, nonzero_rate: 0.999941
- stability_penalty mean: -0.005922, abs_mean: 0.005922

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
