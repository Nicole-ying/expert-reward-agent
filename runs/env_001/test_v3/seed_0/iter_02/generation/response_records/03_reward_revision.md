# Response Record

```json
{
  "action": "tune",
  "target": "stability_penalty and progress_reward coefficients",
  "reasoning": "Training evidence shows stability_penalty dominates (mean -0.340, nonzero 1.0) while progress_reward is too weak (mean 0.160). This causes negative total reward and early failure. Expert card 'stability_penalty_dominance' recommends reducing angle/angular-velocity penalty. Soft_landing_bonus has only 0.5% trigger rate, too sparse to help. Need to reduce stability penalty coefficients and increase progress reward coefficient to balance.",
  "expected_effect": "Progress reward will better guide agent toward target, stability penalty will be lighter to avoid hindering movement, total reward should become positive on average, leading to higher external score.",
  "risk_awareness": "If stability penalty is reduced too much, agent may become unstable and crash. Need moderate reduction and keep speed penalty as primary stability constraint."
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
    # Increased coefficient from 10.0 to 15.0 to strengthen main signal
    progress_reward = 15.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Reduced angle penalty from 0.5 to 0.2, angular penalty from 0.2 to 0.1
    # Keep speed penalty at 0.3 as it's most relevant for stability
    angle_penalty = 0.2 * abs(body_angle)
    angular_penalty = 0.1 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Proxy (small bonus) ==========
    # Conditions: near target, low speed, stable angle, both supports contacting
    near_target = dist_next < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
