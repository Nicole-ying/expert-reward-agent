# Subagent Research Signal

**Key Findings**: mean_eval_reward=216.2, ep_len=554.2, terminated=20/20. Generated reward sum (~322.8) exceeds eval reward. soft_landing_progress dominates 94.2% signed share.

**Component Anomalies**: soft_landing_progress dominates (>94% share). angle_penalty dead (0% active, mean 0). engine_penalty minor.

**Training Dynamics**: No temporal snapshots; only final-policy reward composition available.

**Signal Quality**: angle_penalty dead (never activates). Generated reward overshoots eval score, suggesting misalignment. No coupling data.

**Evidence Confidence**: `medium`
