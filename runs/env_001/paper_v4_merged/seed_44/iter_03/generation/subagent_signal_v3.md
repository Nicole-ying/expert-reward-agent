# Subagent Research Signal

**Key Findings**: Score=144.3, all episodes truncated (len=1000). Progress (56.7% signed share) and landing_reward (32.7%) dominate reward; angle penalty negative (-6.9%); speed penalty dead (0.2% share, 0.1% active).

**Component Anomalies**: speed_penalty nearly dead (active 0.1%, negligible share). landing_reward active 94.3% but per-step mean tiny (0.0067). No component self-cancelling or >70% share.

**Training Dynamics**: No temporal snapshots provided; cannot assess trend.

**Signal Quality**: Dead gate: speed_penalty. angle_penalty always active (100% nonzero) so penalty never zero. No terminations, suggesting missing attractor for landing behavior.

**Evidence Confidence**: `medium`
