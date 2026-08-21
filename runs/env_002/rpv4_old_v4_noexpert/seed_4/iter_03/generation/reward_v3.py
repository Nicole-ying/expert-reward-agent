def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，使用光滑有界形式。
    # 增益从 2.0 提升到 2.5，强化主目标，鼓励更快步态。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.5 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：惩罚躯干倾角与躯干角速度（保持不变，稳定性不是当前瓶颈）
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    # 能量效率（次目标）：从 0.03 降到 0.01。
    # 之前 0.03 的惩罚把策略拖慢（每步速度 0.376 -> 0.338），次目标压过了主目标。
    # 保留轻微惩罚以维持训练平滑，但不让它取代前进速度。
    action_sq = sum(a * a for a in action)
    energy_efficiency_penalty = -0.01 * action_sq

    total_reward = forward_progress + upright_stability + energy_efficiency_penalty

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
        "energy_efficiency_penalty": energy_efficiency_penalty,
    }

    return float(total_reward), components