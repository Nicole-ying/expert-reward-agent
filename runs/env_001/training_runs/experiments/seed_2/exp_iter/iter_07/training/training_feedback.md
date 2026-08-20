# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_2\exp_iter\iter_07\generation\reward_v7.py
- train_run: experiments/seed_2/exp_iter/iter_07/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: 131.085970
- mean_episode_length: 875.600000
- min_eval_reward: 73.078246
- max_eval_reward: 223.457171

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.021849, nonzero_rate: 0.999065
- stability_penalty mean: -0.000884, abs_mean: 0.000884

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
