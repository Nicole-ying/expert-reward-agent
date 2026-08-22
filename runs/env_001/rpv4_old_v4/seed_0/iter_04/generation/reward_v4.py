def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v4: landing_proximity_reward changed back to state-Gaussian but with tight sigma
        (sigma_dist=0.7 vs iter_2's 2.0) to eliminate the hovering-at-moderate-distance
        exploit while avoiding the iter_3 oscillation exploit from progress-weighting.
        The tight Gaussian is near-zero beyond dist~1.5, so the agent must genuinely
        approach the pad to receive meaningful reward.
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

    # 3. Landing proximity reward — tight state-Gaussian (NOT progress-weighted)
    #    sigma_dist=0.7 ensures reward is near-zero beyond ~1.5 units, eliminating
    #    the iter_2 exploit where the agent hovered at moderate distance.
    #    sigma_speed=0.8 and sigma_angle=0.3 gate the reward behind soft-landing
    #    conditions (low speed, upright), which naturally occur during final approach.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 0.7
    sigma_speed = 0.8
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 10.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + landing_proximity_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components