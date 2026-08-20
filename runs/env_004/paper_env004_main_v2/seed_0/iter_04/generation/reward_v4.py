def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键运动信号
    torso_angle = next_obs[1]           # 躯干倾角
    forward_vel = next_obs[5]           # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]         # 躯干角速度

    # 姿态因子：历史最佳iter1的指数衰减结构，躯干越直立因子越接近1
    k_angle = 10.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   前进时奖励，后退时惩罚；姿态倾斜时自动衰减，抑制为速度而过度倾身
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    # 动作效率惩罚：对三关节扭矩平方和施加轻量约束
    #   抑制高频抖动和浪费性大力矩，引导协调高效的跳跃模式
    #   系数0.05：动作平方和在[0,3]范围，单步罚约0~0.15，约为speed_posture的3%~10%
    w_energy = 0.05
    energy_penalty = w_energy * (action[0]**2 + action[1]**2 + action[2]**2)

    total_reward = speed_posture_reward - ang_vel_penalty - energy_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components