# Subagent Research Signal

**Key Findings**: Eval score -114.4, all 20 eps early terminated (<150 steps, score<-50). Our per-step reward 0.014 insufficient vs -1.65 env penalty.

**Component Anomalies**: angle_hinge_penalty dead (0% active, 0 mean). progress_shaping & shaped_progress dominate magnitude (89% combined) but tiny sum.

**Training Dynamics**: No checkpoint snapshots; cannot assess temporal trends.

**Signal Quality**: Angle hinge never activates; progress signals fail to prevent crash; all episodes early terminal.

**Evidence Confidence**: `low`
