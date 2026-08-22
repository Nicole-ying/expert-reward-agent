def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主信号：前向加速度，鼓励每一步增加前向速度
    acceleration = next_obs[5] - obs[5]
    accel_reward = 2.0 * acceleration

    # 稳定性约束：惩罚躯干倾斜
    posture_penalty = -0.1 * (next_obs[1] ** 2)

    # 稳定性约束：惩罚躯干角速度（抑制翻滚）
    ang_vel_penalty = -0.001 * (next_obs[7] ** 2)

    # 平滑约束：轻微惩罚垂直速度幅度，避免过激的上下抖动
    vertical_penalty = -0.01 * (next_obs[6] ** 2)

    total_reward = accel_reward + posture_penalty + ang_vel_penalty + vertical_penalty

    components = {
        "acceleration_reward": accel_reward,
        "posture_penalty": posture_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components