def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，保持 iter_6 验证过的纯加性饱和形态。
    # 证据：iter_1（2v/(1+|v|)）=278.99；iter_6 把增益 2.0→2.1 后速度
    # 0.376→0.409、score 278.99→286.16，是本迭代唯一被实证的正向改进。
    # 本轮继续同一方向做 L1 微调：增益 2.1→2.2，给前向速度更强的边际激励，
    # 同时仍远低于 iter_3 触发前倾 exploit 的 2.5，避免稳定性崩溃。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.2 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：保持 iter_1/iter_6 已验证的形态不变（躯干倾角 + 躯干角速度）。
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components