def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态因子：躯干越直立（角度接近0），因子越接近1
    k_angle = 5.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   - 前进速度为正则奖励，为负则惩罚，引导 agent 持续向前
    #   - 姿态因子在躯干倾斜时自动衰减，抑制为追求速度而过度倾斜
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components