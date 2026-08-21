def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：有界线性前向速度奖励。
    # iter_1 的饱和形式 2v/(1+|v|) 在高速区边际收益快速衰减，策略停在 0.376 m/s。
    # iter_3 把增益提到 2.5 后，前向奖励上限 2.5 超过直立惩罚上限 2.0，
    # 诱导躯干前倾追速度（upright_stability 从 -6.7 恶化到 -73）。
    # 这里改为 1.5 * clip(v, -1, 1)：
    #   - 低速区与 iter_1 接近（v=0.376 时 0.564 vs 0.547），不降低现有速度激励；
    #   - 高速区提供持续线性梯度，推动更快步态；
    #   - 上限 1.5 < 直立惩罚上限 2.0，保留稳定性预算，避免前倾 exploit。
    horizontal_velocity = next_obs[2]
    clipped_velocity = max(-1.0, min(1.0, horizontal_velocity))
    forward_progress = 1.5 * clipped_velocity

    # 稳定性约束：惩罚躯干倾角与躯干角速度（保持 iter_1 已验证的形态不变）。
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    # 删除 energy_efficiency_penalty：
    # iter_2(-0.03*action_sq) 与 iter_3(-0.01*action_sq) 都拖慢步态并恶化 upright，
    # 已实证为负贡献，次目标不应压过前进与稳定主目标。
    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components