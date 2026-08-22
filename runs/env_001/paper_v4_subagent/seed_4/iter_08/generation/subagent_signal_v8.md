# Subagent Research Signal

**Key Findings**: Landing bonus dominates reward (98.6% signed share). Mean eval reward=150.8, termination=55%, ep_len=813.7.

**Component Anomalies**: landing_bonus dominant (episode_sum_mean=1050.7, 98.6% share). Penalties negligible (mean -0.3, -0.07, share ~0%). progress only 1.2% share.

**Training Dynamics**: No temporal checkpoint data; dynamics unknown.

**Signal Quality**: Components highly active (landing_bonus 99.4%). Multiplicative attractor (proximity*speed*angle*contact) may cause sparse spikes. Dominance of landing_bonus may suppress progress signal.

**Evidence Confidence**: `medium`
