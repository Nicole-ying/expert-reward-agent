# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_1\exp_iter\iter_07\generation\reward_v7.py
- train_run: experiments/seed_1/exp_iter/iter_07/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: 259.522516
- mean_episode_length: 335.400000
- min_eval_reward: 236.828402
- max_eval_reward: 286.871097

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- stability_penalty mean: -0.013326, abs_mean: 0.013326

## 5. Preliminary failure hints

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
