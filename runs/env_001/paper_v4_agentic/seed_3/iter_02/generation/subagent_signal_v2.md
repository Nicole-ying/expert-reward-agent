# Subagent Research Signal

**Key Findings**: Automatic fallback after 5 turns without submit. Raw data: [inspect_previous_reward]: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]


**Component Anomalies**: Subagent exhausted turns without explicit submission.

**Training Dynamics**: No temporal analysis available.

**Signal Quality**: No signal quality assessment available.

**Evidence Confidence**: `low`
