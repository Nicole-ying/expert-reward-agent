# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs\env_001\exp_fixed\seed_0\iter_01\generation\reward_v1.py
- train_run: exp_fixed/seed_0/iter_01/training
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: -112.273186
- mean_episode_length: 74.100000
- min_eval_reward: -122.738419
- max_eval_reward: -98.103413

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.161493, nonzero_rate: 0.999988
- stability_penalty mean: -0.341700, abs_mean: 0.341700
- soft_landing_bonus mean: 0.010822, trigger_rate: 0.005411

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
- likely_issue: soft_landing_bonus is too sparse or never reached
- likely_issue: stability penalty may dominate progress signal

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
