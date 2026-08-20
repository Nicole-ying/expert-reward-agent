def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.
    
    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: soft bonus for simultaneous low speed, upright posture, double contact and proximity to target
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)          # near target → gate ≈ 1; far away → gate ≈ 0
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (joint condition)
    w_landing = 2.0
    # proximity factor: exp(-dist^2 / 0.25)
    prox_factor = 2.718281828 ** (-dist**2 / 0.25)
    # velocity factor: exp(-(|vx|+|vy|)^2 / 0.1)
    vel_abs_sum = abs(vx) + abs(vy)
    vel_factor = 2.718281828 ** (-vel_abs_sum**2 / 0.1)
    # angle factor: exp(-angle^2 / 0.01)
    angle_factor = 2.718281828 ** (-angle**2 / 0.01)
    # contact gate (0/1 product)
    double_contact = left_contact * right_contact
    landing_proxy = w_landing * prox_factor * vel_factor * angle_factor * double_contact

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components