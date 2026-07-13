# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]

    # Distance to target pad (Euclidean)
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5

    # --- Component A: approach progress (improvement_delta, positive only) ---
    # Rewards any reduction in distance, encourages reaching the pad.
    progress = max(0.0, dist_obs - dist_next)
    w_progress = 2.0
    approach_reward = w_progress * progress

    # --- Component B: soft landing speed penalty (quadratic with gate) ---
    # Penalises high velocity only when already close to the target,
    # promoting deceleration for a stable touchdown.
    landing_dist_threshold = 1.0
    gate_landing = max(0.0, 1.0 - dist_obs / landing_dist_threshold)  # active when dist_obs < threshold
    w_speed = 1.0
    speed_sq = vx_next**2 + vy_next**2
    soft_landing_penalty = -w_speed * gate_landing * speed_sq

    # --- Component C: stable upright penalty (quadratic) ---
    # Discourages large body angles that could cause flip or single‑foot contact.
    w_angle = 0.1
    upright_penalty = -w_angle * (angle_next**2)

    # Total reward
    total_reward = approach_reward + soft_landing_penalty + upright_penalty

    reward_components = {
        'approach_reward': approach_reward,
        'soft_landing_penalty': soft_landing_penalty,
        'upright_penalty': upright_penalty
    }

    return float(total_reward), reward_components
```
