def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next state variables
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to landing pad center (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # 1. Main learning signal: progress toward the pad (scaled up)
    w_progress = 5.0
    progress_reward = w_progress * (dist - next_dist)

    # 2. Stability penalty: penalize high velocities
    w_vel = 0.1
    velocity_penalty = -w_vel * (vx**2 + vy**2)

    # 3. Attitude penalty: penalize non-zero body angle (want upright)
    w_angle = 0.01
    attitude_penalty = -w_angle * abs(angle)

    # 4. Soft landing proxy: encourage gentle touchdown with both legs
    w_landing = 2.0
    alpha = 20.0   # sharpness for position proximity
    beta = 5.0     # sharpness for low speed
    contact_factor = left_contact * right_contact  # 1 if both legs touch, else 0
    position_proximity = 2.718281828 ** (-alpha * (next_x**2 + next_y**2))
    speed_term = 2.718281828 ** (-beta * (vx**2 + vy**2))
    landing_proxy = w_landing * contact_factor * position_proximity * speed_term

    # 5. Engine usage penalty: discourage unnecessary engine fire
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    total_reward = progress_reward + velocity_penalty + attitude_penalty + landing_proxy + engine_penalty

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "landing_proxy": landing_proxy,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components