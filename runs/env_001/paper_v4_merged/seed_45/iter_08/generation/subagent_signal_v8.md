# Subagent Research Signal

**Key Findings**: High mean reward (142.74) from landing_bonus (50.2%) and contact_reward (46.8%). Progress_delta contributes only 1.7% despite 99.9% active. Termination low (4/20), episode length 885.

**Component Anomalies**: Landing_bonus and contact_reward dominate (>97% combined signed share). Penalty components nearly inactive (active rate ≤4.1%).

**Training Dynamics**: No temporal snapshots provided. No data on component evolution across checkpoints.

**Signal Quality**: Penalty components (angle, angvel, speed, boundary) rarely cross thresholds (active 0.5-4.6%). Potential dead gates. Progress signal overshadowed by contact/landing.

**Evidence Confidence**: `medium`
