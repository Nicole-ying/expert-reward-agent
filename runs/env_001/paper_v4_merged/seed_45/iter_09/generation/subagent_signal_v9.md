# Subagent Research Signal

**Key Findings**: eval_score=-58.9, 0/20 terminated, all truncated at 1000 steps. Completion_bonus & angvel_penalty dead (active_rate=0). Progress dominates (79% mag share) but insufficient.

**Component Anomalies**: completion_bonus dead (never met all 5 conditions). angvel_penalty dead (threshold never crossed). angle_penalty active (-15.6% signed share).

**Training Dynamics**: No temporal snapshots provided; drift/growth not assessable.

**Signal Quality**: Dead gates: completion_bonus requires simultaneous proximity, velocity, angle, angvel, contact; never triggered. angvel_penalty threshold 0.5 never exceeded. No intermediate completion signal. Missing attractor for standing behavior.

**Evidence Confidence**: `medium`
