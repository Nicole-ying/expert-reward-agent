def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle_next = next_obs[0]
    horizontal_velocity_next = next_obs[2]

    # ---------- 非线性健康门控 ----------
    # 倾角在 0.25 rad 内几乎不衰减，超过后迅速塌缩
    safe_angle = 0.25
    excess = max(0.0, abs(hull_angle_next) - safe_angle)
    gate = 1.0 / (1.0 + 80.0 * excess ** 2)

    # ---------- 主学习信号：速度平方 × 门控 ----------
    # 速度平方提供指数级梯度，推动突破线速度平台
    velocity_squared_gated = gate * horizontal_velocity_next ** 2

    # ---------- 极微能量效率信号 ----------
    # 系数极小，仅暗示节能，绝不压抑探索
    action_penalty = -0.005 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    # ---------- 总奖励 ----------
    total_reward = velocity_squared_gated + action_penalty
    components = {
        "velocity_squared_gated": velocity_squared_gated,
        "action_penalty": action_penalty
    }
    return float(total_reward), components