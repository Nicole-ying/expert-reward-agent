def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    # vertical_speed no longer needed for air stability penalty

    # ------------------------------------------------------------
    # 1. Terrain awareness gate
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

    roughness_threshold = 0.3
    roughness_clipped = roughness
    if roughness_clipped > roughness_threshold:
        roughness_clipped = roughness_threshold
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # in [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty
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

    total_reward = forward_reward + balance_penalty

    components = {
        'forward_reward': forward_reward,
        'balance_penalty': balance_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components