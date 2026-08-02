# Subagent Research Signal

**Key Findings**: All 20 episodes timeout at 1000 steps, zero terminations. landing_approach_reward dominates (98.8% signed share, episode sum 3138 vs 11 and 25). Per-step total_reward=2.26, but eval score -18.8 (discrepancy).

**Component Anomalies**: landing_approach_reward magnitude 100x others. soft_landing_penalty is near-zero positive, not a meaningful penalty. progress reward negligible.

**Training Dynamics**: No temporal data; only final policy composition and termination stats.

**Signal Quality**: approach_reward discourages motion (vx, vy penalized) at low height, causing hover without landing. penalty and progress too weak to counter. No terminal reward for touchdown.

**Evidence Confidence**: `medium`
