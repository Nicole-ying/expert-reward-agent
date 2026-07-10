def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 主学习信号：前进速度 ----------
    # horizontal_velocity 在 obs[2]，正值表示向前，负值表示向后
    # 使用 bounded 形式抑制极端速度，同时保持符号信息
    forward_velocity = obs[2]
    bounded_forward = forward_velocity / (1.0 + abs(forward_velocity))
    forward_reward = 1.0 * bounded_forward   # 权重 1.0

    # ---------- 稳定/安全约束：躯干直立惩罚 ----------
    # hull_angle 在 obs[0]，理想值为 0（竖直）
    hull_angle = obs[0]
    upright_penalty = -0.5 * abs(hull_angle)   # 权重 0.5

    # ---------- 健康约束：抑制双脚同时离地 ----------
    #  leg1_contact: obs[8],  leg2_contact: obs[13]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # 当双脚都未触地时，(1-leg1_contact)*(1-leg2_contact) = 1
    double_flight_penalty = -0.2 * ((1.0 - leg1_contact) * (1.0 - leg2_contact))   # 权重 0.2

    # ---------- 合计 ----------
    total_reward = forward_reward + upright_penalty + double_flight_penalty

    components = {
        "forward_velocity_reward": float(forward_reward),
        "upright_penalty": float(upright_penalty),
        "double_flight_penalty": float(double_flight_penalty)
    }
    return float(total_reward), components