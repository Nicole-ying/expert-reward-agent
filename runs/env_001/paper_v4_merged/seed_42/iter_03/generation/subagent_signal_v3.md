# Subagent Research Signal

**Key Findings**: mean_eval_reward=194.9, terminated=16/20, ep_len=661.9. Reward dominated by stable_bonus (81.7% signed share).

**Component Anomalies**: stable_bonus dominates (81.7% share, 82.8% active); approach_reward active 100% but only 16.1% share; fuel_penalty negative but negligible magnitude.

**Training Dynamics**: No temporal monitor data; only final-policy composition.

**Signal Quality**: stable_bonus sparse (proximity gate, contacts) yields high reward when active; goal_progress negligible (0.4%); original_env_reward mean negative (-0.012); coupling: fuel_penalty only with non-zero action, effect small.

**Evidence Confidence**: `medium`
