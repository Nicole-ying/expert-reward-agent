def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    # body_y_velocity = obs[14]  # reserved for later
    # body_z_velocity = obs[15]  # reserved for later

    # ---------- Compute derived signals ----------
    # Uprightness proxy: component of body z-axis in world z (1.0 = perfectly upright)
    # From quaternion: body_up_z = 1 - 2*(quat_x**2 + quat_y**2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Reward components ----------
    # 1) Forward progress (main learning signal)
    # Only reward positive forward velocity; zero or backward yields 0
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress

    # 2) Body height penalty (hinge at boundaries)
    # Healthy range is (0.2, 1.0). We add penalty when approaching the edges:
    # lower danger: z < 0.25, upper danger: z > 0.95
    height_low = max(0.0, 0.25 - body_z)
    height_high = max(0.0, body_z - 0.95)
    reward_height_penalty = -5.0 * (height_low + height_high)

    # 3) Upright stability penalty
    # Penalise deviations from perfect uprightness (body_up_z = 1.0)
    upright_error = 1.0 - body_up_z  # non-negative when body_up_z <= 1.0
    reward_upright = -2.0 * upright_error

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_height_penalty + reward_upright

    components = {
        "forward": reward_forward,
        "height_penalty": reward_height_penalty,
        "upright_penalty": reward_upright
    }

    return float(total_reward), components