# Subagent Research Signal

**Key Findings**: Mean eval reward -103.7, all 20 episodes early-termination (mean len 87.4). Reward components: speed_improvement dominates magnitude (41.6%), approach_delta 26.0%, both positive; fuel_penalty -10.4% signed share but active only 31% of steps; landing_proxy near-zero active (0.5%).

**Component Anomalies**: landing_proxy and x_penalty dead/near-dead (0.5% and 0% active). angle_improvement self-cancelling (mean -0.044, 100% active). No component >70% magnitude share.

**Training Dynamics**: No temporal checkpoint data available.

**Signal Quality**: Desired attractor landing_proxy barely activated; dead x_penalty; fuel_penalty sparse; overall reward scale leads to early crashes.

**Evidence Confidence**: `medium`
