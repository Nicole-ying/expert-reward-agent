```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights (tuned to provide clearer descent and soft landing signals)
    w_dist = 0.5          # penalty for being far from target
    w_speed = 2.0         # stronger penalty for speed to encourage gentle touchdown
    w_angle = 1.0         # penalty for body angle
    w_angvel = 0.5        # penalty for angular velocity
    survival_penalty = -0.1  # small per‑step penalty to avoid endless episodes
    main_engine_penalty = -0.3
    side_engine_penalty = -0.03

    # Contact bonuses
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    base_contact_bonus = 10.0 * both_legs_on_platform

    # Soft landing bonus (much larger when velocity and angle are small)
    alpha = 5.0
    beta = 10.0
    smooth_vy_gate = 2.718281828 ** (-alpha * abs(vy_next))
    smooth_angle_gate = 2.718281828 ** (-beta * (angle_next ** 2))
    soft_contact_bonus = 50.0 * both_legs_on_platform * smooth_vy_gate * smooth_angle_gate

    # Core state penalties (always active, guiding the agent toward target and stable state)
    dist_penalty = -w_dist * dist_next
    speed_penalty = -w_speed * speed_next
    angle_penalty = -w_angle * (angle_next ** 2)
    angvel_penalty = -w_angvel * (angvel_next ** 2)

    # Engine (fuel) penalty
    engine_penalty = 0.0
    if action == 2:
        engine_penalty += main_engine_penalty
    elif action == 1 or action == 3:
        engine_penalty += side_engine_penalty
    # action 0: no penalty

    # Assemble total reward
    total_reward = (dist_penalty +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    survival_penalty +
                    engine_penalty +
                    base_contact_bonus +
                    soft_contact_bonus)

    components = {
        "dist_penalty": dist_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "survival_penalty": survival_penalty,
        "engine_penalty": engine_penalty,
        "base_contact_bonus": base_contact_bonus,
        "soft_contact_bonus": soft_contact_bonus
    }
    return float(total_reward), components
```