def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：前进速度 × 直立门控 ----
    forward_velocity = next_obs[13]

    # 身体直立程度（1=完全竖直，<=0=翻倒）
    quat_x, quat_y = next_obs[2], next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_gate = max(0.0, body_up_z)  # 倒立时门控为 0，抹掉前进奖励

    forward_reward = 2.0 * forward_velocity * upright_gate

    # ---- 稳定/健康约束：身体高度安全区间 ----
    body_height = next_obs[0]
    lower_safe = 0.3
    upper_safe = 0.9
    height_penalty = (
        -5.0 * max(0.0, lower_safe - body_height) +
        -5.0 * max(0.0, body_height - upper_safe)
    )

    # ---- 辅助约束：侧向漂移抑制 ----
    lateral_velocity = next_obs[14]
    lateral_penalty = -0.5 * lateral_velocity**2

    # ---- 效率约束：动作能量代价 ----
    action_energy = sum(a**2 for a in action)
    energy_penalty = -0.01 * action_energy

    total_reward = forward_reward + height_penalty + lateral_penalty + energy_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "height_health_penalty": height_penalty,
        "lateral_drift_penalty": lateral_penalty,
        "action_energy_penalty": energy_penalty
    }

    return float(total_reward), components