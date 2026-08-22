# Subagent Research Signal

**Key Findings**: Mean eval reward 157.86, mean episode length 933.1, only 2/20 terminated. Landing bonus accounts for 97.8% signed share (mean 652.47). Progress is small (2.1%). Angle/angvel penalties near-zero.

**Component Anomalies**: Landing bonus dominates (>97% share). Angle and angvel penalties have negligible magnitude despite moderate active rates (100%/66.8%).

**Training Dynamics**: No temporal snapshots provided; cannot assess evolution across checkpoints.

**Signal Quality**: Angle penalty always active but contribution ~0.0%. Angvel penalty active 66.8% but near-zero. Landing bonus gated by leg contact, active 73.3%. Progress signal small. No immediate dead gates, but weak shaping outside landing.

**Evidence Confidence**: `low`
