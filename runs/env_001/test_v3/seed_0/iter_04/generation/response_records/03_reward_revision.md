# Response Record

```json
{
  "action": "tune",
  "target": "progress_reward coefficient and landing_shaping coefficient",
  "reasoning": "Training evidence shows landing_shaping dominates (mean 1.625) while progress_reward is weak (mean 0.204). This matches Expert Card 'goal_near_oscillation' risk: agent hovers near goal without completing. Score 158.82 is close to 200 but limited by weak progress signal. Recommended Action suggests increasing progress_reward coefficient and decreasing landing_shaping to rebalance.",
  "expected_effect": "Progress_reward becomes the dominant learning signal (mean ~1.0-2.0), landing_shaping becomes a gentle nudge (mean ~0.5-1.0), agent learns to move decisively toward target rather than hovering.",
  "risk_awareness": "If progress_reward is too high, agent may overshoot or become unstable. If landing_shaping is too low, agent may not learn to stabilize near target. Monitor episode length and landing_shaping trigger rate."
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
    
    # ========== Component 1: Progress Delta Reward (main learning signal, increased) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Increased coefficient to make progress the dominant signal
    progress_reward = 100.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint, unchanged) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.05 * abs(body_angle)
    angular_penalty = 0.02 * abs(angular_vel)
    speed_penalty = 0.1 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Shaping (reduced, gentle nudge) ==========
    # Continuous shaping that rewards being near target, low speed, stable angle, and both contacts
    near_target_score = max(0.0, 1.0 - dist_next / 0.5)  # 1.0 when dist=0, 0.0 when dist>=0.5
    low_speed_score = max(0.0, 1.0 - speed / 0.3)  # 1.0 when speed=0, 0.0 when speed>=0.3
    stable_angle_score = max(0.0, 1.0 - abs(body_angle) / 0.2)  # 1.0 when angle=0, 0.0 when angle>=0.2
    both_contact_score = 1.0 if (left_contact > 0.5) and (right_contact > 0.5) else 0.0
    
    # Reduced coefficient from 5.0 to 2.0 to make it a gentle nudge
    landing_shaping = 2.0 * near_target_score * low_speed_score * stable_angle_score * (0.5 + 0.5 * both_contact_score)
    
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
