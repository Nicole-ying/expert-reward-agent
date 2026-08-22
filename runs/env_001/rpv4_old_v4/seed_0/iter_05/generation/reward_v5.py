def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v5: Added fuel_penalty to break the hovering exploit — engine usage is no longer free.
        Reduced w_landing from 10.0→2.0 to prevent landing_proximity_reward domination.
        The fuel penalty makes sustained hovering costly, incentivizing the agent to
        complete the landing and terminate the episode.
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

    # 3. Landing proximity reward — tight state-Gaussian
    #    sigma_dist=0.7 ensures reward is near-zero beyond ~1.5 units.
    #    w_landing reduced from 10.0→2.0 to reduce domination and make room
    #    for fuel_penalty to matter.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 0.7
    sigma_speed = 0.8
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 2.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    # 5. Fuel efficiency penalty — NEW
    #    Penalize engine usage to break the hovering exploit.
    #    action 0 (no engine): zero penalty — gliding is free.
    #    action 2 (main engine): heaviest penalty — maintaining altitude requires this.
    #    action 1/3 (orientation): light penalty — needed for attitude control.
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.2
    else:  # action 1 or 3
        fuel_penalty = -0.05

    total_reward = (progress_reward + orientation_penalty +
                    landing_proximity_reward + vel_damping + fuel_penalty)

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping,
        "fuel_penalty": fuel_penalty
    }
    return float(total_reward), components