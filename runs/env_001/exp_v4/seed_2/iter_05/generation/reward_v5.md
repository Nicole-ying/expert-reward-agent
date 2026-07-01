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
    
    # --- Component 1: Progress Delta Reward (main learning signal, significantly increased) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increase coefficient from 50.0 to 80.0 to make progress dominate more
    progress_reward = 80.0 * progress_delta
    
    # --- Component 2: Stability Penalty (keep light, but slightly increase to prevent wild movement) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.3 * abs(next_body_angle)  # was 0.2
    angular_vel_penalty = 0.15 * abs(next_angular_vel)  # was 0.1
    speed_penalty = 0.2 * speed  # was 0.15
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Landing Shaping (tighten conditions to prevent hacking, reduce coefficient) ---
    # Tighten near_target threshold from 1.0 to 0.6 to require being closer
    near_target = max(0.0, 1.0 - next_dist / 0.6)  # was 1.0
    # Tighten low_speed threshold from 1.0 to 0.5
    low_speed = max(0.0, 1.0 - speed / 0.5)  # was 1.0
    # Tighten stable_angle threshold from 0.5 to 0.3
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # was 0.5
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    # Reduce shaping coefficient from 3.0 to 1.5 to prevent reward hacking
    landing_shaping = 1.5 * near_target * low_speed * stable_angle + 1.0 * both_contact * near_target * low_speed
    
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