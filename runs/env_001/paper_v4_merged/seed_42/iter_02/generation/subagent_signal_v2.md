# Subagent Research Signal

**Key Findings**: score=-92.75, all episodes terminated early (len 68.75). Generated reward (goal_progress=1.12, stable_bonus=1.46) adds small offset but fails to offset original env reward (-1.42/step).

**Component Anomalies**: stable_bonus active 16.9% of steps but 53.9% magnitude share; sparse proximity/contact conditions dominate generated reward.

**Training Dynamics**: no checkpoint snapshots; cannot assess temporal trends.

**Signal Quality**: generated reward mean 0.036/step, all components low magnitude. stable_bonus gating rarely satisfied (proximity<0.5). Weak signal relative to negative env reward.

**Evidence Confidence**: `medium`
