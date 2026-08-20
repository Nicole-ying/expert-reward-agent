# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.05 * abs(vx) - 0.02 * abs(vy) - 0.05 * abs(angle) - 0.01 * abs(angular_vel)

    # Landing quality reward: continuous bounded proxy replacing sparse binary bonus
    # Active only in landing zone (d_next < 0.3), providing gradient toward touchdown
    landing_zone = 0.3
    if d_next < landing_zone:
        # Proximity: 1 at d=0, 0 at d=landing_zone
        prox = 1.0 - d_next / landing_zone

        # Speed quality: 1 when still, 0 when speed >= 0.5
        speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
        vel_quality = max(0.0, 1.0 - speed / 0.5)

        # Angle quality: 1 when upright, 0 when |angle| >= 0.3
        angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)

        # Contact quality: 0 with no contact, 1 with both legs
        contact_quality = 0.5 * (next_obs[6] + next_obs[7])

        # prox gates all qualities; sum allows partial credit during learning
        landing_quality_reward = 2.0 * prox * (vel_quality + angle_quality + 0.5 * contact_quality)
    else:
        landing_quality_reward = 0.0

    total_reward = progress_reward + stability_penalty + landing_quality_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_quality_reward': landing_quality_reward
    }
    return float(total_reward), components
```
