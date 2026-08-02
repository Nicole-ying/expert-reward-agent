# Subagent Research Signal

**Key Findings**: Mean eval score -53.83 despite 100% termination. Progress component mean_sum 1.24 (95.9% signed share), but original_env_reward mean -0.6413. Generated reward positive (0.0118) but true task reward negative.

**Component Anomalies**: angle_penalty and angvel_penalty dead (active_rate 0%). Progress dominates (magnitude_share 100%) but contributes minimal positive value. Safety constraints inactive, leaving crashes unpenalized.

**Training Dynamics**: No temporal snapshots available; cannot assess drift or scaffold progression.

**Signal Quality**: Dead safety gates: thresholds never crossed. Hinge penalties never applied; rewards disconnected from underlying crash risk. Progress is only signal, but insufficient to prevent unsafe behavior.

**Evidence Confidence**: `medium`
