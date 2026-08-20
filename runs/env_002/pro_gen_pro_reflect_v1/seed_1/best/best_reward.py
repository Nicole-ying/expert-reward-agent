def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 姿态门控 ----------
    # 使用平方反比形式，hull_angle=0.5 时 gate≈0.5，0.25 时≈0.8，
    # 确保正常摆动不会过分衰减速度奖励。
    hull_angle_next = next_obs[0]
    gate = 1.0 / (1.0 + (abs(hull_angle_next) / 0.5) ** 2)

    # ---------- 主学习信号：线性速度 × 门控 ----------
    horizontal_velocity_next = next_obs[2]
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 轻量动作效率信号 ----------
    # 每步轻微惩罚力矩平方和，促使策略寻找节能步态。
    action_penalty = -0.02 * (action[0] ** 2 + action[1] ** 2 + 
                              action[2] ** 2 + action[3] ** 2)

    # ---------- 极轻微垂直速度惩罚 ----------
    # 抑制上下颠簸，帮助步态平滑。
    vertical_velocity_next = next_obs[3]
    vertical_penalty = -0.05 * vertical_velocity_next ** 2

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + action_penalty + vertical_penalty
    components = {
        "velocity_gated": velocity_gated,
        "action_penalty": action_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components