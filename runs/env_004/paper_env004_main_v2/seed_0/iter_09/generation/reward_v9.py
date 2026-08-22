def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 前进速度奖励：无界，直接鼓励更快的前向移动
    forward_reward = forward_vel

    # 姿态惩罚：二次型，躯干越倾斜惩罚越重
    # k=10，在0.2rad时惩罚0.4（约为典型forward_vel~2.0的20%）
    k_angle = 10.0
    posture_penalty = k_angle * (torso_angle ** 2)

    # 轻量稳定约束：抑制躯干角速度，防止快速晃动
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = forward_reward - posture_penalty - ang_vel_penalty

    components = {
        'forward_reward': forward_reward,
        'posture_penalty': -posture_penalty,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components