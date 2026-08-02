# Subagent Research Signal

**Key Findings**: All episodes terminated early (len=68.35) with large negative score (-117.5). progress_shaping dominates (90% signed share) but fails to prevent crashing. danger_penalty never activates in evaluation (0% active).

**Component Anomalies**: danger_penalty dead (0% active). progress_shaping overwhelmingly dominant (97% magnitude share). angle_hinge and action_cost negligible.

**Training Dynamics**: No temporal snapshots available; only final policy observed, so no trend analysis possible.

**Signal Quality**: danger_penalty thresholds never triggered despite early terminations. progress_shaping provides positive signal but no attractor for safe landing. Dead gates, missing coupling to desired behavior.

**Evidence Confidence**: `medium`
