def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    # left_c = obs[6], right_c = obs[7] not used for current state

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        # Position factor: prefer x close to 0
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))  # sigma = 0.05
        # Speed factor: penalise high total speed
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))   # sigma = 0.2
        # Attitude factor: prefer upright
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))     # sigma = 0.1
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target (auxiliary)
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    # Safety gate for progress: when near target, suppress reward if speed/angle are high
    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        # Use current vertical speed and body angle to form a soft gate
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty (very small)
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Boundary penalty: discourage moving outside viewport horizontally
    boundary_penalty = 0.0
    if abs(x_pos) > 1.0:
        boundary_penalty = -5.0 * (abs(x_pos) - 1.0)

    # 5. Light angle/angular-velocity penalty (global, to stabilise attitude)
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + boundary_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "boundary_penalty": boundary_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components