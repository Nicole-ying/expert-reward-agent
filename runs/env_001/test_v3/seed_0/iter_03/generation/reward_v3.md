```json
{"action": "tune", "target": "progress_reward coefficient, stability_penalty coefficients, soft_landing_bonus -> continuous shaping", "reasoning": "Training evidence shows penalty dominance: stability_penalty mean (-0.331) > progress_reward mean (0.242) in absolute terms, causing negative total reward. soft_landing_bonus has only 0.5% trigger rate, making it effectively useless. Expert card 'stability_penalty_dominance' recommends reducing angle/angular penalties. Recommended action suggests increasing progress_reward coefficient to 50+ and reducing stability penalties. I will also convert the sparse soft_landing_bonus into a continuous shaping term to provide more consistent guidance.", "expected_effect": "Progress reward will dominate over stability penalty, giving positive total reward. Continuous soft landing shaping will provide gradient toward good landing states instead of rare sparse bonus.", "risk_awareness": "Over-reducing stability penalty may cause agent to move too fast or become unstable. Continuous shaping may reward intermediate states that don't lead to actual landing. Monitor for high_reward_without_success."}
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
    # Significantly increased coefficient to overcome penalty dominance
    progress_reward = 50.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint, reduced) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Reduced all penalties significantly to avoid dominance
    angle_penalty = 0.05 * abs(body_angle)
    angular_penalty = 0.02 * abs(angular_vel)
    speed_penalty = 0.1 * speed
    
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