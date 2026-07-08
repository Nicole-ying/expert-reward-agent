def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：负距离，引导飞行器持续靠近原点 ===
    # 这是唯一被验证有效的信号骨架（iter 1: score=-111.87）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    distance_reward = -dist_to_target

    # === 稳定性惩罚：弱背景信号，抑制高速/大角度/高角速度 ===
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av = 0.02

    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # === 软着陆连续代理奖励 ===
    # 改进：从二值 if 条件改为连续乘积因子（bounded_continuous_proxy 技法）
    # 每个因子 ∈ [0,1]，乘积提供稠密梯度，引导 agent 同时满足所有着陆条件
    near_factor = max(0.0, 1.0 - dist_to_target / 0.5)       # dist<0.5 时 >0，越近越大
    slow_factor = max(0.0, 1.0 - speed_norm / 0.3)           # speed<0.3 时 >0，越慢越大
    upright_factor = max(0.0, 1.0 - angle_abs / 0.1)         # angle<0.1 时 >0，越正越大
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0       # 平均接触 [0,1]

    soft_landing = near_factor * slow_factor * upright_factor * contact_factor * 5.0

    # === 总奖励 ===
    total_reward = distance_reward + stability_penalty + soft_landing

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing': soft_landing,
    }

    return float(total_reward), components