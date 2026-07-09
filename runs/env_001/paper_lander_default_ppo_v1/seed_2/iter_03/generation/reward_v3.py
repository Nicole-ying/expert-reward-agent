def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: light penalty on high speeds and large angles
    #    Coefficient reduced 12.5x from -0.1 to -0.008 to fix |penalty/progress| > 0.5 dominance
    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        curr_dist < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components