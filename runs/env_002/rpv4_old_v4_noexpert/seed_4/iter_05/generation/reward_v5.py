def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度（保留 iter_1 验证的饱和形态），乘以上身直立门控。
    # 历史证据：iter_4 用 1.5*clip(v,-1,1) 提供恒定边际梯度，策略靠躯干前倾
    # （θ≈0.172 rad）把 v 从 0.376 提到 0.451，upright_stability 从 -6.7 恶化到 -45.6，
    # len 从 1275 掉到 1061，score 261.45 < iter_1 278.99；iter_3 提高增益也复现前倾 exploit。
    # 加性「速度奖励 + 直立惩罚」无法阻止这种套利，这里改为乘性门控：
    # 只有躯干保持直立时速度奖励才足额发放，使「前倾换速度」失去净收益。
    horizontal_velocity = next_obs[2]
    raw_forward = 2.0 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    torso_angle = next_obs[0]
    # 直立门控：θ≈0.059（iter_1 正常步态）时约 0.99，几乎不衰减；
    # θ≈0.172（iter_4 前倾 exploit）时约 0.92，显著压低速度奖励。
    upright_gate = 1.0 / (1.0 + 3.0 * torso_angle * torso_angle)

    forward_progress = raw_forward * upright_gate

    # 稳定性约束：保持 iter_1 已验证的形态（躯干倾角 + 躯干角速度）。
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