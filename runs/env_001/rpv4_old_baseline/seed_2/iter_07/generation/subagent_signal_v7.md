# Subagent Research Signal

**训练过程**: Early→Mid: crash 89→79%, len 124→155, score 445→822. Mid→Late: crash 79→87%, len 155→130, score 822→516. Agent never achieved survival; crash>87% late, regression after mid-phase improvements.

**组件健康**: All penalties (horiz, vx, orient, angvel, time) active 100% with negative means. descent_shaping 100% active, mean=-0.297. contact_reward nonzero 13.6%, mean=3.31 when active. landing_bonus sparse 5.7% but mean=200.0 when active. Total generated_reward mean=0.972 from sparse bonuses.

**奖励对齐**: generated_reward mean=+0.97/step vs original_env_reward mean=-0.84/step. Shaped reward positive despite task failure; agent exploits sparse bonuses without learning to avoid crashes (crash>87%).

**异常检测**: Mid-to-late regression: crash rate rose, score and length dropped after earlier improvement. Suggests divergence or overfitting to shaped reward.

**置信度**: `high`
