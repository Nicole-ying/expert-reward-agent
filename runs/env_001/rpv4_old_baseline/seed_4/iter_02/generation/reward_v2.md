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
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights
    w_progress = 2.0
    w_speed   = 0.2
    w_angle   = 0.5
    w_angvel  = 0.5
    w_contact = 100.0          # strong incentive for safe contact
    survival_penalty = -0.5    # per step penalty to promote fast landing
    main_engine_penalty = -0.3 # penalty for using fuel-expensive main engine
    side_engine_penalty = -0.03 # small penalty for orientation engines
    alpha     = 5.0            # vertical speed gate sharpness
    beta      = 10.0           # angle gate sharpness

    # 1. Progress: distance improvement minus speed penalty
    progress_reward = w_progress * (dist_curr - dist_next) - w_speed * speed_next

    # 2. Orientation stabilization
    orientation_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # 3. Engine (fuel) penalty
    engine_penalty = 0.0
    if action == 2:
        engine_penalty += main_engine_penalty
    elif action == 1 or action == 3:
        engine_penalty += side_engine_penalty
    # action 0: no penalty

    # 4. Soft‑landing contact reward
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    # Gate: prefer low vertical speed and near‑upright attitude
    smooth_vy_gate = 2.718281828 ** (-alpha * abs(vy_next))
    smooth_angle_gate = 2.718281828 ** (-beta * (angle_next ** 2))
    soft_contact_reward = w_contact * both_legs_on_platform * smooth_vy_gate * smooth_angle_gate

    # Assembling total reward
    total_reward = (progress_reward +
                    orientation_penalty +
                    engine_penalty +
                    survival_penalty +
                    soft_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "engine_penalty": engine_penalty,
        "survival_penalty": survival_penalty,
        "soft_contact_reward": soft_contact_reward
    }
    return float(total_reward), components
```