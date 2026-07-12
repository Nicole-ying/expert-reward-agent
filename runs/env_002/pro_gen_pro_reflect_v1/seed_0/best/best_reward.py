def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ----- 1. Forward progress reward -----
    # Encourage positive horizontal velocity only (discourage backward motion implicitly via other penalties)
    forward_speed_reward = 2.0 * max(0.0, horizontal_velocity)

    # ----- 2. Upright stability penalty -----
    # Penalize deviation from upright orientation and rapid angular motion
    angle_penalty = 1.0 * (hull_angle ** 2)
    angvel_penalty = 0.1 * (hull_angular_velocity ** 2)
    upright_penalty = -(angle_penalty + angvel_penalty)

    # ----- 3. Energy efficiency cost -----
    # Light penalty on joint torque magnitudes to discourage wasteful actions
    torque_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_cost = -0.01 * torque_sq_sum

    total_reward = forward_speed_reward + upright_penalty + action_cost
    components = {
        "forward_speed_reward": forward_speed_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components