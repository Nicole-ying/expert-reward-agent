# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # 1. Strong shaping for approaching the target
    approach_reward = 10.0 * (dist_curr - dist_next)

    # 2. Soft landing success reward – gives credit for partial success
    contact_ratio = (left_contact + right_contact) / 2.0
    landing_quality = 1.0 if (speed_next < 0.2 and abs(angle_next) < 0.1) else 0.0
    landing_success_reward = 200.0 * contact_ratio * landing_quality

    # 3. Penalties for speed, angle, angular velocity
    speed_penalty = -0.05 * speed_next
    angle_penalty = -0.1 * (angle_next ** 2)
    angvel_penalty = -0.05 * (angvel_next ** 2)

    # 4. Small per-step survival bonus
    survival_bonus = 0.05

    # 5. Fuel penalties
    main_penalty = -0.5 if action == 2 else 0.0
    side_penalty = -0.1 if action in (1, 3) else 0.0
    engine_penalty = main_penalty + side_penalty

    total_reward = (approach_reward +
                    landing_success_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    survival_bonus +
                    engine_penalty)

    components = {
        "approach_reward": approach_reward,
        "landing_success_reward": landing_success_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "survival_bonus": survival_bonus,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components
```
