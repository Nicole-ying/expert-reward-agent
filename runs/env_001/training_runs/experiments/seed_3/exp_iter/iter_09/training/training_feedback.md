# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\experiments\seed_3\exp_iter\iter_09\generation\reward_v9.py
- train_run: experiments/seed_3/exp_iter/iter_09/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: -480.614044
- mean_episode_length: 65.800000
- min_eval_reward: -641.231990
- max_eval_reward: -185.582067

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
