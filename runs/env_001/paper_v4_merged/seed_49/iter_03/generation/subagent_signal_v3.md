# Subagent Research Signal

**Key Findings**: Mean eval score -115.68, all episodes crash early at ~68 steps. Shaped reward per step +0.0074 but true env reward -1.6156, indicating severe proxy misalignment.

**Component Anomalies**: soft_landing dominates (56.6% signed share) yet active only 1.2% of steps. angle_penalty dead (0 mean, 0% active). angvel_penalty rare (1% active) but significant negative share (-21.7%).

**Training Dynamics**: No checkpoint snapshots; only final policy data. Temporal trends unavailable.

**Signal Quality**: Dead angle_penalty threshold never crossed. Sparse soft_landing reward (1.2% active) likely causes high variance. Positive shaped reward fails to reflect crashing outcome.

**Evidence Confidence**: `medium`
