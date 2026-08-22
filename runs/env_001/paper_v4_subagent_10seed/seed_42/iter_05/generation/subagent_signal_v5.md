# Subagent Research Signal

**Key Findings**: mean_eval_reward=146.4, ep_len=812.95, terminated=16/20. Reward dominated by soft_landing (98% signed share, active only 12.5% of steps).

**Component Anomalies**: soft_landing: dominating (98% share, mean=69.2) but sparse (12.5% active). orientation_penalty: 100% active, negligible magnitude (share -0.1%). safe_progress: 70.3% active, tiny share (2%).

**Training Dynamics**: No temporal checkpoint data available.

**Signal Quality**: Dead signal: orientation_penalty always present but effectively zero. Sparse gate: soft_landing fires rarely, creating sparse reward attractor. safe_progress provides weak signal despite frequent activation. Missing dense progress incentive.

**Evidence Confidence**: `medium`
