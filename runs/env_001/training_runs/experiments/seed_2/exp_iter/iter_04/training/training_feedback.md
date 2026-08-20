# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_2\exp_iter\iter_04\generation\reward_v4.py
- train_run: experiments/seed_2/exp_iter/iter_04/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: -5.483078
- mean_episode_length: 1000.000000
- min_eval_reward: -37.712499
- max_eval_reward: 29.699580

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.018344, nonzero_rate: 0.999819
- stability_penalty mean: -0.010608, abs_mean: 0.010608

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
