```json
{
  "action": "rebuild",
  "target": "骨架从 progress_delta + stability_penalty + landing_shaping 切换为 distance_reward + progress_delta + conditional_stability_penalty",
  "reasoning": "当前骨架（progress_delta + stability_penalty + landing_shaping）已迭代4轮，最佳得分158.82未达目标200，且最近3轮得分持续下降（158.82→-142.80→-190.67→100.05）。landing_shaping触发率48%但未带来突破，progress_reward均值仅0.228，表明结构瓶颈。Expert Cards指出goal_near_oscillation风险（距离改善但episode length长，landing proxy触发不足）和stability_penalty_dominance（当前stability_penalty均值-0.022，绝对值小于progress_reward，但landing_shaping均值1.37主导了总奖励，导致agent可能停留在目标附近震荡而不完成）。Training Evidence显示landing_shaping的nonzero_rate仅48%，且其均值1.37远高于progress_reward的0.228，导致总奖励被landing_shaping主导，agent可能学会在目标附近低速震荡而非真正接近。Recommended Action明确建议rebuild。根据expert_reward_context.md，推荐使用distance_reward作为小权重anchor（-d(obs,goal)）提供持续梯度，配合progress_delta作为主信号，并条件化stability_penalty（仅在接近目标时生效）以避免保守。",
  "expected_effect": "distance_reward提供持续向目标移动的梯度，progress_delta驱动每一步接近，条件化stability_penalty在远离目标时允许高速移动，在接近目标时约束姿态和速度，从而打破震荡瓶颈，提升得分至200以上。",
  "risk_awareness": "distance_reward和progress_delta同时使用可能导致梯度冲突（一个鼓励接近，一个鼓励delta），需小权重distance_reward（如0.1倍）。条件化stability_penalty可能使agent在远离目标时姿态不稳定，但任务目标是接近目标，远离时姿态不重要。landing_shaping被移除，可能失去对最终着陆的引导，但当前landing_shaping触发率仅48%且未带来突破，移除后可让progress信号主导。"
}
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
    
    # ========== Component 1: Progress Delta Reward (main learning signal) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Strong coefficient to drive learning
    progress_reward = 50.0 * progress_delta
    
    # ========== Component 2: Distance Reward (small anchor, provides continuous gradient) ==========
    # Negative distance to target, scaled down to avoid dominance
    distance_reward = -0.1 * dist_next
    
    # ========== Component 3: Conditional Stability Penalty (only near target) ==========
    # Compute speed and angle
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Only penalize instability when close to target (dist_next < 0.5)
    # This allows high speed when far away, but requires stable approach near goal
    near_target_threshold = 0.5
    if dist_next < near_target_threshold:
        # Progressive penalty that increases as distance decreases
        proximity_factor = 1.0 - dist_next / near_target_threshold  # 0 at threshold, 1 at target
        angle_penalty = 0.1 * abs(body_angle) * proximity_factor
        angular_penalty = 0.05 * abs(angular_vel) * proximity_factor
        speed_penalty = 0.2 * speed * proximity_factor
        stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    else:
        stability_penalty = 0.0
    
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