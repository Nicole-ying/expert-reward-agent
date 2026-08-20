def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    quat_x = next_obs[2]
    quat_y = next_obs[3]                     # quaternion components for upright projection
    body_x_vel = next_obs[13]                # forward velocity

    # ---------- 1. upright incentive ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)   # 1 when fully upright, 0 when horizontal
    upright_reward = 0.5 * body_up_z

    # ---------- 2. forward velocity (main progress) ----------
    forward_reward = 2.0 * body_x_vel

    # ---------- 3. height safety hinge ----------
    lower_safe = 0.25
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    height_hinge_penalty = -10.0 * height_penalty

    # ---------- combine ----------
    total_reward = forward_reward + upright_reward + height_hinge_penalty

    components = {
        "forward_reward": forward_reward,
        "upright_reward": upright_reward,
        "height_hinge_penalty": height_hinge_penalty
    }

    return float(total_reward), components