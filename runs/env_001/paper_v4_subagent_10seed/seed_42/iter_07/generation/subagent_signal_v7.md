# Subagent Research Signal

**Key Findings**: Mean eval reward 194.6, 19/20 terminated, ep len 630.2. Proximity_reward dominates 89.7% signed share; soft_landing active only 9.7% of steps.

**Component Anomalies**: orientation_penalty dead (mean -0.0007, 0% magnitude share). proximity_reward dominating (89.7% share). soft_landing low active rate (9.7%).

**Training Dynamics**: No monitor snapshots; temporal trends unavailable.

**Signal Quality**: orientation_penalty signal negligible. soft_landing threshold (dist<0.3) rarely met. Proximity reward provides continuous gradient.

**Evidence Confidence**: `medium`
