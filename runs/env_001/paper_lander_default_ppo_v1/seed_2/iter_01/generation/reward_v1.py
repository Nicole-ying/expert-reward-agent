def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next observation variables
    x = next_obs[0]      # x relative to pad center
    y = next_obs[1]      # y relative to pad height
    vx = next_obs[2]     # horizontal velocity
    vy = next_obs[3]     # vertical velocity
    angle = next_obs[4]  # body angle
    omega = next_obs[5]  # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Main drive: distance to target (dense negative signal)
    dist_to_target = (x**2 + y**2) ** 0.5
    reward_dist = -2.0 * dist_to_target

    # 2. Stability constraint: penalise high speeds and large angles
    reward_stability = (
        -0.1 * abs(vx) -
        0.1 * abs(vy) -
        0.1 * abs(angle) -
        0.1 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        dist_to_target < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_reward": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components