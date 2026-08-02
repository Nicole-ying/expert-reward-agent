# Subagent Research Signal

**Key Findings**: Final eval: score=222.88, term=13/20, ep_len=552.5, total_reward/step=0.0615. Contact_reward dominates signed_share (47.1%) despite low active_rate (44.1% in eval, 21.8% nonzero in training summary). Progress active 98.3% with 35.1% share.

**Component Anomalies**: contact_reward: high signed share but sparse activity, indicating large per-event rewards. angle_penalty: always active, small magnitude share (-8%). landing_reward: minimal contribution despite high active_rate.

**Training Dynamics**: No temporal snapshots provided; unable to detect growth/decay or drift.

**Signal Quality**: contact_reward gated by exponential proximity/speed; active_rate discrepancy (21.8% vs 44.1%) suggests thresholds may be too strict or misaligned with policy behavior. No dead gates observed.

**Evidence Confidence**: `medium`
