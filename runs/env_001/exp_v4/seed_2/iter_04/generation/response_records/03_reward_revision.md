# Response Record

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
    
    # --- Component 1: Progress Delta Reward (main learning signal, increased) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increase coefficient from 10.0 to 50.0 to make progress dominate
    progress_reward = 50.0 * progress_delta
    
    # --- Component 2: Stability Penalty (reduced weights to avoid dominating) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    # Reduce all penalties significantly so agent is not afraid to move
    angle_penalty = 0.2 * abs(next_body_angle)  # was 0.4
    angular_vel_penalty = 0.1 * abs(next_angular_vel)  # was 0.2
    speed_penalty = 0.15 * speed  # was 0.3
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Landing Shaping (relaxed conditions for higher activation) ---
    # Relax near_target threshold from 0.5 to 1.0 to activate earlier
    near_target = max(0.0, 1.0 - next_dist / 1.0)  # was 0.5
    # Relax low_speed threshold from 0.5 to 1.0
    low_speed = max(0.0, 1.0 - speed / 1.0)  # was 0.5
    # Relax stable_angle threshold from 0.3 to 0.5
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # was 0.3
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    # Increase shaping coefficient to make it more impactful
    landing_shaping = 3.0 * near_target * low_speed * stable_angle + 2.0 * both_contact * near_target * low_speed
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
