# Subagent Research Signal

**Key Findings**: final mean score -117.88, all episodes terminated early (mean len 68.3). Progress_shaping dominates (signed share 90.4%) but original env reward -1.57, large misalignment.

**Component Anomalies**: progress_shaping dominates (97.1% mag), action_cost and angle_hinge nearly dead (active 4.9%, 0.1%). Positive sum (1.065) but agent fails.

**Training Dynamics**: no temporal snapshots; checkpoint evolution unknown.

**Signal Quality**: angle_hinge dead, threshold never crossed; action_cost negligible. Only progress_shaping active, no early-termination penalty. Missing success attractor.

**Evidence Confidence**: `medium`
