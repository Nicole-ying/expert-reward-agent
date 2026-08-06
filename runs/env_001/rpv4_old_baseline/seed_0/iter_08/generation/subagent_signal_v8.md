# Subagent Research Signal

**训练过程**: No survival learning: crash_rate=100% all phases (early/mid/late). Avg episode length ~70, score 67-73, no clear trend. Agent never avoided crashes.

**组件健康**: Active: radial_reward, vel_penalty, angle_penalty, time_penalty (100% nonzero); descent_reward (93.7%). Low activity: engine_penalty (27.6%), proximity_bonus (21.1%). landing_bonus nearly dead (3.4% nonzero), mean 35.8 when active, reflecting rare crash/landing events.

**奖励对齐**: Severe misalignment: generated_reward/step ~4-5 positive, original_env_reward/step ~-100. Agent exploits shaped reward (high positive) while achieving negative true reward (crashes). Exploitation present.

**异常检测**: Crash rate stuck at 100%; episode length plateau ~70 suggests agent learns to extend flight but cannot land, possibly hitting time limit. No reward progress toward safe landing.

**置信度**: `high`
