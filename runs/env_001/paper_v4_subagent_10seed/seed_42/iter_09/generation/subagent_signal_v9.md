# Subagent Research Signal

**Key Findings**: Eval score -10.7, 0/20 terminated, all truncated at 1000 steps. Soft_landing dead (active_rate=0%, sum=0). Proximity_reward dominates.

**Component Anomalies**: Soft_landing dead in final eval despite 39.8% nonzero in train summary. Proximity_reward 100% share. Attitude_penalty negligible (-0.011 sum).

**Training Dynamics**: No temporal snapshots; cannot assess drift or plateau.

**Signal Quality**: Soft_landing threshold (dist<0.3) not regularly crossed; avg dist ~0.39. Reward uncorrelated with task (orig_env_reward -0.09). No effective attitude signal.

**Evidence Confidence**: `medium`
