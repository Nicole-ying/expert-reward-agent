def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 姿态门控（与迭代 8 相同的安全前兆信号） ----------
    hull_angle_next = next_obs[0]
    gate = 1.0 / (1.0 + (abs(hull_angle_next) / 0.5) ** 2)

    # ---------- 主学习信号：线性前进速度 × 门控 ----------
    horizontal_velocity = next_obs[2]
    velocity_gated = gate * horizontal_velocity

    # ---------- 轻量步态交替奖励 ----------
    # 当双腿接触状态不同（一腿支撑、一腿摆动）时给予正奖励；
    # 双支撑或双离地时奖励为 0。不惩罚，不干扰探索。
    contact_diff = next_obs[8] - next_obs[13]   # leg1_contact - leg2_contact
    gait_bonus = 0.03 * contact_diff ** 2

    # ---------- 动作效率信号 ----------
    # 每步轻微惩罚力矩平方和，激励节能步态。
    action_penalty = -0.02 * (action[0] ** 2 + action[1] ** 2 +
                              action[2] ** 2 + action[3] ** 2)

    # ---------- 极轻微垂直速度惩罚 ----------
    vertical_velocity = next_obs[3]
    vertical_penalty = -0.03 * vertical_velocity ** 2

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + gait_bonus + action_penalty + vertical_penalty
    components = {
        "velocity_gated": velocity_gated,
        "gait_bonus": gait_bonus,
        "action_penalty": action_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components