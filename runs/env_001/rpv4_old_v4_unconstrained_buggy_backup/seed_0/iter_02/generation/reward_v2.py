def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v2: proximity-based soft landing proxy replaces dead contact_reward;
        widened vel_damping gate so it actually activates.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft landing proxy (proximity-based, continuous — replaces dead contact_reward)
    # Rewards being in the "landing zone": close to pad + low speed + upright.
    # Creates a smooth gradient from approach → hover → touch down.
    sigma_dist = 1.5                                    # landing zone radius
    sigma_vel = 0.5                                     # acceptable landing speed
    sigma_angle = 0.3                                   # acceptable tilt

    dist = dist_curr
    speed = (vx ** 2 + vy ** 2) ** 0.5

    proximity_factor = 2.718281828 ** (- (dist ** 2) / (2.0 * sigma_dist ** 2))
    vel_factor = 2.718281828 ** (- (speed ** 2) / (2.0 * sigma_vel ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (2.0 * sigma_angle ** 2))

    w_landing = 5.0
    soft_landing_proxy = w_landing * proximity_factor * vel_factor * angle_factor

    # 4. Near-pad velocity damping (wider gate so it activates)
    dist_thresh = 3.0                                   # was 1.0 — far too narrow
    gate = max(0.0, 1.0 - dist / dist_thresh)           # 0 far away, 1 at origin
    w_vel_damp = 0.3
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + soft_landing_proxy + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "vel_damping": vel_damping
    }
    return float(total_reward), components