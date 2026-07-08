def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为原点 (0,0)
    # 计算到目标的欧氏距离（使用 next_obs）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5

    # 主学习信号：负距离，引导飞行器持续靠近目标（保持不变）
    distance_reward = -dist_to_target

    # 稳定性惩罚：轻量抑制高速、大角度和角速度（保持不变）
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av = 0.02

    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # === 核心改动：软着陆从稀疏二值 → 连续 bounded 乘积 ===
    # 原理：每个因子是 bounded 连续函数，提供稠密梯度
    #   proximity = 1/(1+5*dist)：远处≈0，原点=1，处处有梯度
    #   speed_factor = max(0, 1-speed/0.5)：速度<0.5时有梯度，>=0.5时为0
    #   angle_factor = max(0, 1-angle/0.2)：角度<0.2时有梯度，>=0.2时为0
    #   contact_bonus = (L+R)/2：连续接触信号
    # 乘积后乘 5.0，使理想着陆状态(d≈0,v≈0,a≈0,双触)奖励≈5，与 distance_reward 量级可比
    proximity = 1.0 / (1.0 + 5.0 * dist_to_target)
    speed_factor = max(0.0, 1.0 - speed_norm / 0.5)
    angle_factor = max(0.0, 1.0 - angle_abs / 0.2)
    contact_bonus = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_continuous = proximity * speed_factor * angle_factor * contact_bonus * 5.0

    # 总奖励
    total_reward = distance_reward + stability_penalty + soft_landing_continuous

    # 记录各组件
    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_continuous': soft_landing_continuous,
        'total_reward': total_reward
    }

    return float(total_reward), components