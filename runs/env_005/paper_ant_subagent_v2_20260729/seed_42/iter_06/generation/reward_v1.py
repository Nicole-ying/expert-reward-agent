def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Survival bonus: encourages the agent to stay alive
    # ------------------------------------------------------------
    survival_bonus = 0.5

    # ------------------------------------------------------------
    # Body height factor (based on next_obs[0] = body_z)
    # healthy range [0.3, 0.9] with linear ramps to 0.2 and 1.0
    # ------------------------------------------------------------
    z = next_obs[0]
    if z < 0.3:
        height_factor = max(0.0, (z - 0.2) / 0.1)      # 0.2 -> 0, 0.3 -> 1
    elif z <= 0.9:
        height_factor = 1.0
    else:
        height_factor = max(0.0, (1.0 - z) / 0.1)      # 1.0 -> 0, 0.9 -> 1

    # ------------------------------------------------------------
    # Upright factor (based on next_obs[2], next_obs[3] = quat_x, quat_y)
    # body_up_z = 1 - 2*(x^2 + y^2); clip negative to 0
    # ------------------------------------------------------------
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    upright_factor = max(0.0, body_up_z)

    # ------------------------------------------------------------
    # Forward velocity (next_obs[13] = body_x_velocity)
    # ------------------------------------------------------------
    forward_vel = next_obs[13]

    # ------------------------------------------------------------
    # Health gate: product of height and upright factors
    # ------------------------------------------------------------
    health_gate = height_factor * upright_factor

    # ------------------------------------------------------------
    # Forward progress: only rewarded when the agent is healthy
    # ------------------------------------------------------------
    forward_progress = forward_vel * health_gate

    # ------------------------------------------------------------
    # Total reward and components
    # ------------------------------------------------------------
    total_reward = survival_bonus + forward_progress
    components = {
        'survival_bonus': survival_bonus,
        'forward_progress': forward_progress,
    }
    return float(total_reward), components