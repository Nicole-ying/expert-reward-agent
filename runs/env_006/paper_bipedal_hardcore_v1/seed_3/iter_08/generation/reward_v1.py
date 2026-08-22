def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: progress guided by preview of upcoming terrain roughness,
    with a hinge penalty on excessive torso tilt.
    """
    # --- Extract observations from NEXT_OBS ---------------------------------
    hull_angle = next_obs[0]                # torso tilt (rad)
    horizontal_speed = next_obs[2]          # forward speed (m/s)
    lidar = next_obs[14:24]                 # 10 terrain height readings

    # --- Component A: forward progress ---------------------------------------
    # Encourage forward motion; ignored when moving backwards.
    progress_raw = max(0.0, horizontal_speed)
    w_progress = 1.0

    # --- Component B: preview factor from lidar roughness --------------------
    # Higher terrain roughness -> reduce effective progress, encouraging
    # the agent to slow down / adjust gait before obstacles.
    n_lid = len(lidar)
    if n_lid > 0:
        mean_l = sum(lidar) / n_lid
        # variance of lidar readings as roughness measure
        var_l = sum((l - mean_l) ** 2 for l in lidar) / n_lid
        roughness = var_l ** 0.5
    else:
        roughness = 0.0

    # preview_factor in [preview_min, 1.0]; 1.0 on flat ground, decays with roughness
    k_preview = 2.0          # sensitivity to roughness
    preview_factor = 1.0 / (1.0 + k_preview * roughness)

    progress_reward = w_progress * progress_raw * preview_factor

    # --- Component C: posture hinge penalty ----------------------------------
    # Penalize dangerous torso tilt beyond a safe threshold.
    torso_threshold = 0.5          # ~28.6 degrees
    w_posture = 1.0
    excess_tilt = max(0.0, abs(hull_angle) - torso_threshold)
    posture_penalty = -w_posture * excess_tilt

    # --- Combine -------------------------------------------------------------
    total_reward = progress_reward + posture_penalty
    components = {
        "progress_reward": progress_reward,
        "posture_penalty": posture_penalty
    }

    return float(total_reward), components