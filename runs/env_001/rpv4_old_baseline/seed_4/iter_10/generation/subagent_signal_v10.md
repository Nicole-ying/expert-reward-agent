# Subagent Research Signal

**训练过程**: Agent improved over time: episode length grew from 219 to 305, score from 803 to 1475, crash rate fell from 64% to 44%. Early episodes were brief with frequent terminations; later episodes showed longer survival but still high crash rate.

**组件健康**: landing_success_reward active only 24.3% of steps (mean=300.9 when active); engine_penalty active 75.3%; angvel_penalty active 95.0%; all others active 100%. No dead components, but landing reward rarely triggered.

**奖励对齐**: Shaped reward diverged from original_env_reward. Training: generated_reward/step improved from -0.39 to +1.66, while original_env_reward/step stayed negative (-89.3 to -81.2). Final eval mean reward = -90.4, indicating shaped reward overestimates task success. Large gap persists.

**异常检测**: No sudden divergence or value explosion; steady improvement in metrics. Persistent reward-reality gap.

**置信度**: `high`
