# Subagent Research Signal

**Key Findings**: mean_eval_reward=-222.06, ep_len=148.5, terminated=20/20. failure_penalty (-35.4% signed share) and efficiency (-32.2%) dominate negative reward.

**Component Anomalies**: failure_penalty: large episodic -1.5 mean sum but only 0.2% active (terminal spikes). efficiency: pervasive -0.013/step, 45.9% active. soft_landing and angvel_penalty dead (0% active, 0% share).

**Training Dynamics**: no temporal snapshots; cannot detect growth/decay or scaffold drift.

**Signal Quality**: soft_landing never triggered; failure_penalty threshold rarely crossed but dominates when it does; progress positive but minor (6.9% share).

**Evidence Confidence**: `medium`
