def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 --------------------
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 稳定性门控（替换独立角度惩罚） --------------------
    # 指数衰减：倾角越大，门控值越接近 0，按比例削弱前进奖励
    # α=3.0 使得：0.3 rad → gate≈0.76, 0.5 rad → gate≈0.47, 0.8 rad → gate≈0.15
    alpha = 3.0
    stability_gate = 2.718281828 ** (-alpha * hull_angle ** 2)
    gated_forward_reward = forward_reward * stability_gate

    # -------------------- 角速度二次惩罚（保持不变） --------------------
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（保留，后续可能调整） --------------------
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = gated_forward_reward + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,          # 原始前进奖励（仅观测）
        "stability_gate": stability_gate,          # 门控值（用于诊断）
        "gated_forward_reward": gated_forward_reward,  # 实际计入总奖励的前进部分
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components