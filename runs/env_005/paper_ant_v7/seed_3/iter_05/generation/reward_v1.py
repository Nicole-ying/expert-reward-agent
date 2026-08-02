def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- body height safety hinge ----------
    # punish when height is approaching the low termination boundary (0.2)
    safe_low = 0.3
    height_err = safe_low - body_z_next
    w_height = 100.0
    height_penalty = -w_height * max(0.0, height_err) ** 2

    # ---------- upright orientation safety hinge ----------
    # body_up = 1 - 2*(qx^2 + qy^2), range [-1, 1]
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    unsafe_up = 0.5
    up_err = unsafe_up - body_up
    w_up = 5.0
    upright_penalty = -w_up * max(0.0, up_err) ** 2

    # ---------- lateral drift penalty ----------
    w_lat = 0.5
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- total reward ----------
    total_reward = forward_reward + height_penalty + upright_penalty + lateral_penalty

    components = {
        "forward_reward": forward_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty
    }
    return float(total_reward), components