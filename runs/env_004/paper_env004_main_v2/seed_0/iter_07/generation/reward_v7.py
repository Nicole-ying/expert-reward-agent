def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 主信号：独立奖励前进速度，不再被姿态因子乘性压制
    forward_reward = forward_vel

    # 姿态约束：仅在倾斜超出阈值后才惩罚，允许跳跃必需的动态倾斜
    tilt_threshold = 0.3
    excess_tilt = max(0.0, abs(torso_angle) - tilt_threshold)
    posture_penalty = 1.0 * (excess_tilt ** 2)

    # 角速度约束：抑制快速翻滚，与历史最佳相同的线性形式
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = forward_reward - posture_penalty - ang_vel_penalty

    components = {
        'forward_reward': forward_reward,
        'posture_penalty': -posture_penalty,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components