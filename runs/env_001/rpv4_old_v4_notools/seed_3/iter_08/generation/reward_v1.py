def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励设计：基于 delta 距离的进展奖励 + 软着陆连续奖励 + 姿态速度约束 + 燃油惩罚
    """
    # 权重系数
    w_approach  = 1.0
    w_soft_land = 1.0
    w_stability = 0.05
    w_energy    = 0.05
    w_contact   = 0.1   # 双腿接触对齐小鼓励

    # 辅助常量
    speed_gate_alpha = 2.0
    land_coeff_v     = 2.0
    land_coeff_d     = 2.0

    # --- 当前和下一步状态提取 ---
    # 相对目标平台中心的位置
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 速度
    vx, vy = next_obs[2], next_obs[3]

    # 姿态
    angle      = next_obs[4]
    ang_vel    = next_obs[5]

    # 接触
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # 动作：主引擎点燃是 2
    main_engine_used = 1.0 if action == 2 else 0.0

    # --- 计算欧氏距离 ---
    current_dist = (x**2 + y**2)**0.5
    next_dist    = (nx**2 + ny**2)**0.5

    # --- 1. 进展奖励（delta 距离 + 速度门控）---
    delta_dist = current_dist - next_dist   # 正值表示靠近目标
    abs_speed  = abs(vx) + abs(vy)
    speed_gate = 2.718281828 ** (-speed_gate_alpha * abs_speed)   # 用 e 的幂次，无需额外库
    approach_delta_gated = w_approach * delta_dist * speed_gate

    # --- 2. 软着陆连续奖励 ---
    # 双腿接触 + 低速 + 靠近中心 → 高奖励
    both_legs = left_contact * right_contact     # 0 或 1
    speed_factor = 2.718281828 ** (-land_coeff_v * abs_speed)
    dist_factor  = 2.718281828 ** (-land_coeff_d * next_dist)
    soft_landing_bonus = w_soft_land * both_legs * speed_factor * dist_factor

    # --- 3. 姿态/角速度稳定性惩罚 ---
    stability_penalty = -w_stability * (abs(angle) + abs(ang_vel))

    # --- 4. 主引擎燃油消耗惩罚 ---
    energy_penalty = -w_energy * main_engine_used

    # --- 5. 双腿接触对齐的小鼓励 ---
    contact_alignment = w_contact * both_legs   # 只有同时接触时才会得到小正向奖励

    # 总奖励
    total_reward = (
        approach_delta_gated
        + soft_landing_bonus
        + stability_penalty
        + energy_penalty
        + contact_alignment
    )

    components = {
        "approach_delta_gated": approach_delta_gated,
        "soft_landing_bonus":   soft_landing_bonus,
        "stability_penalty":    stability_penalty,
        "energy_penalty":       energy_penalty,
        "contact_alignment":    contact_alignment,
    }

    return float(total_reward), components