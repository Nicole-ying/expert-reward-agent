```json
{"action": "rebuild", "target": "landing_shaping + progress_reward + stability_penalty", "reasoning": "当前骨架（progress_reward + soft_landing_bonus + stability_penalty）迭代3轮，最佳得分146.57低于历史最佳158.82。训练证据显示：progress_reward均值仅0.091（系数10.0过弱），stability_penalty均值-0.146主导总奖励（触发penalty_dominance信号），soft_landing_bonus触发率19.8%但稀疏。历史最佳骨架（iter3）使用landing_shaping连续塑形+progress_reward系数50.0+低stability_penalty，得分158.82。Expert Card建议：stability_penalty_dominance需降低惩罚系数，sparse_completion_proxy需将稀疏bonus替换为连续shaping。因此回退到最佳骨架并微调：progress_reward系数保持50.0，stability_penalty进一步降低系数，landing_shaping保持连续但调整权重。", "expected_effect": "progress_reward均值提升至0.2-0.3，stability_penalty均值降至-0.05以内，landing_shaping提供连续梯度引导，总奖励均值提升至0.5以上，得分突破200。", "risk_awareness": "progress_reward系数过高可能导致目标附近震荡；landing_shaping条件过宽可能奖励hacking；stability_penalty过低可能允许不稳定行为。需监控训练中是否出现high_reward_without_success或fast_crash_near_goal。"}
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
    
    # ========== Component 2: Stability Penalty (light constraint, further reduced) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Further reduced penalties to avoid dominance
    angle_penalty = 0.03 * abs(body_angle)
    angular_penalty = 0.01 * abs(angular_vel)
    speed_penalty = 0.05 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Shaping (continuous, replaces sparse bonus) ==========
    # Continuous shaping that rewards being near target, low speed, stable angle, and both contacts
    # This provides gradient instead of sparse binary bonus
    near_target_score = max(0.0, 1.0 - dist_next / 0.5)  # 1.0 when dist=0, 0.0 when dist>=0.5
    low_speed_score = max(0.0, 1.0 - speed / 0.3)  # 1.0 when speed=0, 0.0 when speed>=0.3
    stable_angle_score = max(0.0, 1.0 - abs(body_angle) / 0.2)  # 1.0 when angle=0, 0.0 when angle>=0.2
    both_contact_score = 1.0 if (left_contact > 0.5) and (right_contact > 0.5) else 0.0
    
    # Combined continuous shaping signal (product ensures all conditions matter)
    landing_shaping = 5.0 * near_target_score * low_speed_score * stable_angle_score * (0.5 + 0.5 * both_contact_score)
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```