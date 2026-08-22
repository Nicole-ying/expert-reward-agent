# Subagent Research Signal

**训练过程**: Agent improved across phases: avg_len 318→373→433, avg_score 1790→2035→2759, crash_rate 36%→20%→19%. Gen_reward/step 4.70→4.97→6.03, original_env_reward/step -85.1→-81.9→-74.6, indicating better stability but poor task completion.

**组件健康**: All components active: proximity, height, speed, angle at 100% nonzero. Contact_reward nonzero=33.7% (mean_when_active=0.37). Landing_bonus nonzero=19.1% (mean_when_active=200). Dominant component is landing_bonus when active, but per-step mean only 38.2.

**奖励对齐**: Large gap: per-step generated_reward=5.85 vs original_env_reward=-0.293. Training original_reward improved slightly, but evaluation mean score=-80.09, all episodes truncated at max steps, no successful landings. Reward shaping fails to incentivize terminal success; exploitation signs: agent learns to survive longer (higher avg_len, lower crash_rate) but does not optimize for landing bonus conditions.

**异常检测**: Despite training improvements, evaluation episodes never terminate (all 1000 steps) and scores remain negative. Landing bonus never fired during evaluation.

**置信度**: `high`
