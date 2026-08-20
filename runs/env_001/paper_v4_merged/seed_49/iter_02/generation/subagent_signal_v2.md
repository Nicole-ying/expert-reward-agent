# Subagent Research Signal

**Key Findings**: Mean shaped reward +0.0202/step but env return -113.7; all 20 episodes terminate early (<150 steps). Progress active 100% mean 1.12; soft_landing 0.97 mean but active only 0.9%.

**Component Anomalies**: angle_penalty dead (mean 0.0, active 0%); soft_landing high magnitude share (39.6%) but near-zero active rate; efficiency negative only 7% active.

**Training Dynamics**: No checkpoint snapshots; cannot assess temporal trends.

**Signal Quality**: Dead angle penalty; soft_landing gate rarely opens; progress always active, may dominate learning; no signal for collision avoidance.

**Evidence Confidence**: `medium`
