```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1 for the lunar‑lander‑like 2D vehicle landing task.
    - Physical components are built only from declared observation signals.
    - No terminal flags or hidden info are used.
    """
    # ---------- goal_proximity: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)   # positive when approaching

    # ---------- attitude_stabilization: penalty for tilt and angular velocity ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_contact: soft proxy for safe two-leg touch-down ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    # Both legs must be in contact to be counted
    contact_score = left_contact * right_contact

    # Low‑velocity factor (both horizontal and vertical speed)
    vx = abs(next_obs[2])
    vy = abs(next_obs[3])
    velocity_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

    # Upright orientation factor (angle near zero)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(angle))

    soft_landing_bonus = 5.0 * contact_score * velocity_factor * angle_factor

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    reward_terms = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), reward_terms
```