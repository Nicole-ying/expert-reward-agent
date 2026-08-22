# Subagent Research Signal

**训练过程**: Early→late: avg_len 139→185, avg_score 272.5→504.5, crash_rate 82%→68%. Agent improved length and score, but original_env_reward only marginally improved (-96.8→-93.6/step).

**组件健康**: landing_reward active 11.3% steps (mean 150 when active). total_reward, original_env_reward, progress_reward, stability_penalty all 100% active. action_penalty 84.1% active. No dead components.

**奖励对齐**: Shaped reward (gen_reward) positive 2.28/step, original_env_reward negative -0.586/step. Gen_reward increased over training while original_env_reward barely changed. In eval, mean orig_reward=-98.9, 17/20 episodes truncated, indicating exploitation by extending episodes without consistent landing.

**异常检测**: Persistent reward-reality gap: original_env_reward always negative across all phases. Shaped reward promotes progress/landing but env penalizes heavily. Agent learned to max episode length instead of landing; no sudden divergence.

**置信度**: `high`
