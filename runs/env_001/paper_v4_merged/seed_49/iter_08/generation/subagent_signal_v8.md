# Subagent Research Signal

**Key Findings**: mean_eval_reward=-114.07, all 20 episodes truncated at 1000 steps, success_bonus never activated.

**Component Anomalies**: success_bonus is dead in eval (0% active); training summary shows 34% nonzero and mean 68.06, indicating possible distributional shift or overfitting.

**Training Dynamics**: no temporal snapshots provided, unable to report growth/decay.

**Signal Quality**: success_bonus dead, no episodes scored; progress is active but small magnitude; overspeed_penalty negligible; large gap between training mean reward (6.81) and eval reward (-114.07) suggests poor generalization.

**Evidence Confidence**: `medium`
