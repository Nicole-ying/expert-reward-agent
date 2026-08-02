# Subagent Research Signal

**Key Findings**: Env reward -117.8 (all episodes early-term <150 steps, score<-50) while shaped reward per ep sums to ~+1.26. Shaped reward mismatch: progress (64% share) and sparse landing reward (29% share, active 0.7% steps) dominate, but env outcome catastrophic.

**Component Anomalies**: boundary_penalty dead (zero always). landing_soft_reward active only 0.7% steps but 29% magnitude share. progress always active (100%) dominates 64% signed share. action_cost near-dead (4% active). angle_penalty always on but -3.4% share.

**Training Dynamics**: No checkpoint snapshots provided; unable to assess temporal trends.

**Signal Quality**: Dead gates: boundary_penalty never fires. Missing dense landing guide: landing reward triggers rarely. Progress gate may suppress near-target. No component signals steady approach to soft landing.

**Evidence Confidence**: `medium`
