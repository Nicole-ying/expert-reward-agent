def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态惩罚：躯干倾角平方，引导直立但允许跳跃所需的小幅倾斜
    # w_posture=5.0: 0.1rad→罚0.05, 0.2rad→罚0.20, 0.3rad→罚0.45
    w_posture = 5.0
    posture_penalty = w_posture * (torso_angle ** 2)

    # 主学习信号：前进速度与姿态惩罚加法解耦
    #   - forward_vel 始终以原始尺度被奖励/惩罚，不再被姿态因子缩放
    #   - 姿态偏差独立惩罚，agent可接受小幅倾斜以换取更高前进速度
    speed_posture_reward = forward_vel - posture_penalty

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components