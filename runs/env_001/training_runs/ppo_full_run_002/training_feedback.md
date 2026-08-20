# Training Feedback for Reward Revision

This file is the preferred LLM input for the next diagnosis/reflection step.
It is intentionally short and evidence-focused.

## 1. Run
- reward_path: runs/env_001/deepseek_full_run_002/reward_v1.py
- train_run: ppo_full_run_002
- total_timesteps: 1000000
- n_envs: 4
- reward_clip: 20.0
- error_fallback: zero

## 2. External evaluation
- mean_eval_reward: -111.544763
- mean_episode_length: 74.100000
- min_eval_reward: -121.825106
- max_eval_reward: -105.813061

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.159844, nonzero_rate: 0.999994
- speed_penalty mean: -0.525343, abs_mean: 0.525343, nonzero_rate: 0.999928
- stability_penalty mean: -0.024591, abs_mean: 0.024591
- soft_landing_bonus mean: 0.008542, trigger_rate: 0.004271
- total_reward mean: -0.381548, abs_mean: 0.397792
- generated_reward mean: -0.381548; original_env_reward mean: -1.543968

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
- likely_issue: speed_penalty dominates progress signal
- evidence: abs(speed_penalty mean)=0.525343 > abs(progress_reward mean)=0.159844
- likely_issue: soft_landing_bonus is too sparse or rarely reached
- evidence: trigger_rate=0.004271
- likely_issue: generated reward is negative on average during training

## 6. Do not directly change from this file alone
Use this feedback together with reward_v1.py, reward_v1.md, and expert failure-mode knowledge before generating reward_v2.
