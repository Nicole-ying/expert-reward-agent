def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # ------------------------------------------------------------
    # 1. Terrain awareness gate (NEW - replaces raw forward_progress)
    #    Use lidar readings (obs[14] to obs[23]) to measure terrain roughness.
    #    Explicit single-index access to avoid any slice syntax.
    # ------------------------------------------------------------
    lidar_0 = obs[14]
    lidar_1 = obs[15]
    lidar_2 = obs[16]
    lidar_3 = obs[17]
    lidar_4 = obs[18]
    lidar_5 = obs[19]
    lidar_6 = obs[20]
    lidar_7 = obs[21]
    lidar_8 = obs[22]
    lidar_9 = obs[23]

    n_lidar = 10.0
    mean_lidar = (
        lidar_0 + lidar_1 + lidar_2 + lidar_3 + lidar_4 +
        lidar_5 + lidar_6 + lidar_7 + lidar_8 + lidar_9
    ) / n_lidar

    variance = (
        (lidar_0 - mean_lidar) ** 2 +
        (lidar_1 - mean_lidar) ** 2 +
        (lidar_2 - mean_lidar) ** 2 +
        (lidar_3 - mean_lidar) ** 2 +
        (lidar_4 - mean_lidar) ** 2 +
        (lidar_5 - mean_lidar) ** 2 +
        (lidar_6 - mean_lidar) ** 2 +
        (lidar_7 - mean_lidar) ** 2 +
        (lidar_8 - mean_lidar) ** 2 +
        (lidar_9 - mean_lidar) ** 2
    ) / n_lidar

    roughness = variance ** 0.5

    # Map roughness to a gate: 1.0 (smooth) -> 0.3 (rough)
    roughness_threshold = 0.3
    roughness_clipped = roughness
    if roughness_clipped > roughness_threshold:
        roughness_clipped = roughness_threshold
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # in [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty (unchanged)
    # ------------------------------------------------------------
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = abs(hull_angle) - angle_threshold
    if angle_excess < 0.0:
        angle_excess = 0.0
    angvel_excess = abs(hull_angvel) - angvel_threshold
    if angvel_excess < 0.0:
        angvel_excess = 0.0
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # ------------------------------------------------------------
    # 3. Air-stability penalty (unchanged)
    # ------------------------------------------------------------
    sum_contact = leg1_contact + leg2_contact
    both_feet_off = 1.0 - sum_contact
    if both_feet_off < 0.0:
        both_feet_off = 0.0
    air_penalty = -0.3 * both_feet_off

    neg_vertical_speed = -vertical_speed
    if neg_vertical_speed < 0.0:
        neg_vertical_speed = 0.0
    vertical_fall_penalty = -1.0 * both_feet_off * neg_vertical_speed

    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_reward + balance_penalty + air_stability_penalty

    components = {
        'forward_reward': forward_reward,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components