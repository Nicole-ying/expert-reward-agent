# Subagent Research Signal

**Key Findings**: Mean eval reward -45, all episodes truncated (1000 steps). Progress has largest signed share (42.2%) but is canceled by angle/speed penalties. Score highly negative despite positive progress.

**Component Anomalies**: Angle_penalty has high magnitude (28.7% signed share) yet active only 3.8% of steps—sparse large penalties. Completion and boundary_penalty dead (0 sum, 0% active). Contact_reward virtually dead in final eval.

**Training Dynamics**: No temporal monitor snapshots provided; cannot assess component growth/decay or checkpoint drift.

**Signal Quality**: Contact_reward active in training (mean 0.19, 50.2% nonzero) but zero in final eval—policy likely avoids contacts. Progress and penalty components self-cancelling, leaving no net progress towards completion. Completion metric never triggered.

**Evidence Confidence**: `medium`
