# Response Record

```json
{"action": "rebuild", "target": "骨架重建：移除landing_shaping，强化progress_delta_reward，条件化stability_penalty", "reasoning": "1) 当前骨架已迭代3轮，最佳得分158.82远低于目标200，最近两轮得分大幅下降（-142.80, -190.67），趋势恶化。2) landing_shaping触发率仅0.2%（均值0.0097），几乎无学习信号，且与外部评分负相关。3) stability_penalty全触发（100%），均值-0.0675，绝对值大于progress_reward均值5.12的1%，存在penalty_dominance风险，抑制探索。4) progress_reward均值5.12但外部评分-190.67，说明progress_reward与外部目标不一致，需要大幅提高系数并调整形态。5) Expert Cards建议：early_failure_or_crash需添加平滑引导信号；goal_near_oscillation需避免纯距离奖励；stability_penalty_dominance需降低或条件化。6) 根据expert_reward_context.md，推荐使用progress_delta_reward作为主信号（系数>200），stability_penalty条件化（仅当接近目标时生效），移除无效的landing_shaping，添加轻量distance_reward作为anchor。", "expected_effect": "大幅提高progress_reward系数至500，使正向信号主导；stability_penalty条件化（仅当dist<2时生效），避免抑制探索；移除landing_shaping，消除无效信号；添加distance_reward（系数0.1）作为稳定anchor。期望外部评分从-190提升至>0，逐步接近200目标。", "risk_awareness": "1) progress_reward系数过高可能导致agent只追求接近目标而忽略稳定性，需配合条件化stability_penalty。2) 移除landing_shaping后，任务完成信号缺失，但当前触发率极低，移除风险可控。3) 条件化stability_penalty可能在远距离时无约束，但progress_reward主导下agent会主动接近目标，接近后stability_penalty生效。4) 若外部评分仍为负，需进一步调整progress_reward系数或添加其他引导信号。"}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables
    # Position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Velocity
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # Orientation
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (main learning signal, significantly increased) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Significantly increased coefficient to overcome penalty dominance and drive learning
    progress_reward = 500.0 * progress_delta
    
    # ========== Component 2: Distance Reward (light anchor, provides baseline gradient) ==========
    # Continuous distance-based reward to encourage being close to target
    # Small weight to avoid dominating progress signal
    distance_reward = -0.1 * dist_next
    
    # ========== Component 3: Conditional Stability Penalty (only when near target) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Only apply stability penalty when close to target (dist_next < 2.0)
    # This avoids suppressing exploration when far away
    near_target_condition = 1.0 if dist_next < 2.0 else 0.0
    
    angle_penalty = 0.05 * abs(body_angle) * near_target_condition
    angular_penalty = 0.02 * abs(angular_vel) * near_target_condition
    speed_penalty = 0.1 * speed * near_target_condition
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Total Reward ==========
    total_reward = progress_reward + distance_reward + stability_penalty
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
