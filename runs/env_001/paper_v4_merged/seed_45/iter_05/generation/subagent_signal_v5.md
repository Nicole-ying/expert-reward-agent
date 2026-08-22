# Subagent Research Signal

**Key Findings**: Eval score 195.3±~74.8 (10/20 terminated). Completion_proxy dominates reward (98.3% signed share, magnitude 466.3 vs progress_delta 6.6). Penalties are negligible (active <5%). Ep length 754.65.

**Component Anomalies**: completion_proxy is dominating (>70% share) with 98.3% share. angle_penalty, angvel_penalty, speed_penalty are near-zero magnitude and low activity (0.4-4.2%). No self-cancelling components observed.

**Training Dynamics**: No temporal snapshots provided. Cannot assess growth/decay of components over training.

**Signal Quality**: Penalty thresholds (0.4,0.2,0.3) rarely exceeded (active rates 3.2%,0.4%,4.2%), making penalization ineffective. completion_proxy active only 64.5% of steps. No coupling data.

**Evidence Confidence**: `medium`
