# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for lunar‑lander‑like task.
    Components:
      - proximity_progress   : improvement in distance to target
      - attitude_penalty     : penalty for tilt and angular velocity
      - soft_landing_bonus   : continuous bonus for low velocity, upright angle and leg contacts
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- soft landing bonus ----------
    left_contact  = next_obs[6]
    right_contact = next_obs[7]
    # Sum instead of product → much less sparse, still rewards both legs
    contact_score = left_contact + right_contact

    vx = abs(next_obs[2])
    vy = abs(next_obs[3])
    velocity_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

    angle_factor = 1.0 / (1.0 + 5.0 * abs(angle))

    soft_landing_bonus = 4.0 * contact_score * velocity_factor * angle_factor

    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```
