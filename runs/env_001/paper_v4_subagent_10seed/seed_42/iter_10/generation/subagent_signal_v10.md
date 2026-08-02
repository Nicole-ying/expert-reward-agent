# Subagent Research Signal

**Key Findings**: Eval score -16.77, episodes truncate at 1000 steps with no terminations. Proximity_reward dominates (864.1 ep sum, 100% signed share).

**Component Anomalies**: soft_landing dead (active 0%, 0 share). proximity_reward overwhelmingly dominant (>70% share). attitude_penalty negligible.

**Training Dynamics**: No temporal monitor data; checkpoint evolution unknown.

**Signal Quality**: Soft_landing threshold never crossed—dead gate. No coupling signals. Missing attractor for desired landing.

**Evidence Confidence**: `medium`
