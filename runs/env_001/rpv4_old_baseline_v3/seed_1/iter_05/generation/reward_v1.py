def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]
    """
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]

    # 主学习信号：potential-based shaping
    # 潜力 = -distance - w_speed * speed_sq，speed_sq = vx^2 + vy^2
    # 接近且减速时潜力上升 → 正奖励
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5
    speed_sq = vx**2 + vy**2
    nspeed_sq = nvx**2 + nvy**2

    pot = -dist - 0.2 * speed_sq
    next_pot = -ndist - 0.2 * nspeed_sq

    progress_reward = 15.0 * (next_pot - pot)  # 放大系数使奖励数值明显

    # 姿态稳定约束（对下一步姿态的二次惩罚）
    # 权重较小，防止抑制必要的横向调节
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.1 * (n_av ** 2)

    total_reward = progress_reward - attitude_penalty

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
    }

    return float(total_reward), components