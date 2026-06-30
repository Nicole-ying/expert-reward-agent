```json
{
  "action": "tune",
  "target": "stability_penalty, soft_landing_bonus, progress_reward",
  "reasoning": "Score -111.24 is declining from best -110.40. stability_penalty dominates with mean -0.218 (always active) vs progress_reward mean +0.323. soft_landing_bonus trigger rate only 0.9% (sparse). original_env_reward mean -1.53 indicates environment punishes movement. Need to: (1) further reduce stability_penalty coefficients and make it conditional near target only, (2) replace sparse binary bonus with continuous shaping for landing quality, (3) increase progress_reward coefficient to strengthen positive signal. This is a tune action because skeleton is not stagnant (only 1 iteration) and best_reward.py shows similar structure worked better with lower coefficients.",
  "expected_effect": "Reduce penalty dominance, increase positive progress signal, make landing bonus more frequent to guide learning. Target: score improvement from -111 to -90 or better.",
  "risk_awareness": "Over-reducing stability penalty may allow unstable behavior. Progress coefficient increase may amplify noise. Need to monitor for new failure modes."
}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Reward moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increased coefficient from 20.0 to 25.0 to strengthen positive signal
    progress_reward = 25.0 * progress_delta
    
    # --- Component 2: Conditional Stability Penalty (reduced dominance) ---
    # Penalize high velocity, large angle, and high angular velocity
    # Only apply when near target (dist < 1.0), otherwise minimal penalty
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.2 * abs(next_body_angle)  # reduced from 0.3
    angular_vel_penalty = 0.05 * abs(next_angular_vel)  # reduced from 0.1
    speed_penalty = 0.1 * speed  # reduced from 0.2
    
    # Apply distance-based gating: only penalize when near target
    near_target = next_dist < 1.0
    if near_target:
        stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    else:
        stability_penalty = -0.05 * speed  # minimal penalty far from target
    
    # --- Component 3: Continuous Landing Quality Shaping (replaces sparse bonus) ---
    # Continuous bonus based on proximity, stability, and contact
    # Smooth shaping that provides signal even when not perfectly landed
    near_target_factor = max(0.0, 1.0 - next_dist / 0.8)  # 0 to 1 as dist goes 0.8 to 0
    stability_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.4) * max(0.0, 1.0 - speed / 1.5)
    contact_factor = min(1.0, (next_left_contact + next_right_contact) / 2.0)  # 0 to 1
    landing_quality = 2.0 * near_target_factor * stability_factor * contact_factor
    
    # --- Component 4: Small Distance Anchor (auxiliary) ---
    # Small negative reward proportional to distance to keep agent aware of goal
    distance_anchor = -0.1 * next_dist
    
    # --- Combine components ---
    total_reward = progress_reward + stability_penalty + landing_quality + distance_anchor
    
    # --- Build components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```