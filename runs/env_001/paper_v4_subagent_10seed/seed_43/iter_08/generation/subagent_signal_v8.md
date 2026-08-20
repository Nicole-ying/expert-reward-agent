# Subagent Research Signal

**Key Findings**: mean_eval_reward=-80.85, mean_ep_len=103.55, terminated=20/20. landing_soft_reward active 1.1% but 54.9% signed_share; safety_penalty active 9.5% with -19.9% share. Progress always on but only 12.7% share.

**Component Anomalies**: landing_soft_reward is dead (1.1% active) yet dominates positive share. safety_penalty rare (9.5%) with large negative share. angle_penalty constant tiny. action_cost moderate.

**Training Dynamics**: no temporal snapshots; unable to assess trends.

**Signal Quality**: landing reward threshold rarely crossed (next_left>0.5 & next_right>0.5). Sparse signals cause high variance. Missing consistent attractor for landing behavior.

**Evidence Confidence**: `medium`
