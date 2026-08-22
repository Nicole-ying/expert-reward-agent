def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，恢复 iter_1 验证过的纯加性饱和形态。
    # 证据：iter_1（2v/(1+|v|) 无门控）=278.99 为历史最佳；
    # iter_5 加了乘性直立门控后 upright 虽稳定（-6.85），但速度被推到 0.43、
    # len 从 1275 掉到 1109，forward_progress ep_sum 从 697 降到 660，score 271 < iter_1。
    # 说明饱和形态本身不需要门控；门控扭曲了速度-稳定性权衡。
    # 这里移除门控，并把增益从 2.0 微调到 2.1，给温和且仍有界的向前推动
    # （远低于 iter_3 触发前倾 exploit 的 2.5）。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.1 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：保持 iter_1 已验证的形态不变（躯干倾角 + 躯干角速度）。
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