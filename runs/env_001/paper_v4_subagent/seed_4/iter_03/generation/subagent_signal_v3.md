# Subagent Research Signal

**Key Findings**: Mean reward -6.02, episodes 455 steps, 85% termination. Landing_proxy 82.7% signed share but only 25.7% active rate.

**Component Anomalies**: failure_penalty dead (0% nonzero). landing_proxy extremely sparse but dominates reward. progress always present but small (4.3% share).

**Training Dynamics**: No temporal snapshots provided.

**Signal Quality**: failure_penalty never activates. landing_proxy fires late, no pre-landing attractor. stability_penalty always negative.

**Evidence Confidence**: `medium`
