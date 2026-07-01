def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.80，严重主导信号
    # 修复：整体系数降低 10 倍，从 0.5/0.2/0.1 降到 0.05/0.02/0.01
    # 目标：让 penalty 均值降到 progress 均值的 10% 以下
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数降低 10 倍）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])      # 原 0.5 → 0.05
    angular_vel_penalty = 0.02 * abs(next_obs[5]) # 原 0.2 → 0.02
    speed_penalty = 0.01 * speed                  # 原 0.1 → 0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，几乎从不触发
    # 修复：二值 if 条件 → 连续乘积形式
    # 每个因子用 max(0, 1 - x/threshold)，值域 [0,1]
    # 这样 agent 每靠近一步都能感受到梯度
    # ============================================================
    
    # 连续因子：距离因子（距离 < 0.5 时为正，越近越大）
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子（速度 < 0.5 时为正，越慢越大）
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子（角度 < 0.3 时为正，越小越大）
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    # 接触因子：两个接触传感器都 > 0.5 时为 1，否则为 0（这个保持二值，因为接触是离散事件）
    contact_factor = 1.0 if (next_obs[6] > 0.5) and (next_obs[7] > 0.5) else 0.0
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 系数从 2.0 提高到 5.0，因为连续乘积天然小于 1，需要补偿
    landing_bonus = 5.0 * dist_factor * speed_factor * angle_factor * contact_factor

    # 动作代价：energy_penalty（保持不变，ratio 很小没问题）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components