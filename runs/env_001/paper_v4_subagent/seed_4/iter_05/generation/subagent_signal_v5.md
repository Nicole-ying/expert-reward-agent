# Subagent Research Signal

**Key Findings**: Score -124.5, len 68.4, 100% early termination. Reward dominated by sparse failure_penalty (-52.8% signed, active 2.5%); success_reward rarely triggers (31.1% signed, active 0.1%).

**Component Anomalies**: failure_penalty dominates negative when triggered (magnitude 52.8%, active 2.5%). success_reward near-dead (active 0.1%). soft_landing dead (active 0.3%). progress always active but small share (10.4%).

**Training Dynamics**: No temporal snapshots provided.

**Signal Quality**: Dead success/soft_landing signals; failure_penalty sparse but massive; no intermediate shaping to guide toward success; progress signal exists but insufficient.

**Evidence Confidence**: `medium`
