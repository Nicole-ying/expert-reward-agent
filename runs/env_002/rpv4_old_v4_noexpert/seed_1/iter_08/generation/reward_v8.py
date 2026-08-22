def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干不接近摔倒时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：保留"平台 + 高斯尾"结构，但把中等前倾区间（0.25~0.7 rad）收紧回
    # iter_5 已验证的纯高斯 0.7 轮廓。
    # 证据：iter_5（纯高斯 0.7）=291.96 best；iter_6（纯高斯 0.8）=216.5 塌缩；
    # iter_7（平台 0.25 + tail 0.45）=253.1，中等角度仍比 iter_5 松，cp50 深跌到 -99。
    # 本轮 L1：平台 0.25->0.15，sigma 0.45->0.55。
    # 效果：0.7 rad 处 gate=0.368（与 iter_5 一致）、0.8 rad 处 gate=0.247（≈iter_5 0.271），
    # 但 0.4 rad 0.895->0.813、0.5 rad 0.734->0.667，把中等前倾的过度奖励拉回 iter_5 水平。
    hull_angle = float(next_obs[0])
    lean_margin = max(0.0, abs(hull_angle) - 0.15)
    upright_gate = 2.718281828 ** (-(lean_margin / 0.55) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：保持 iter_5 的力度（角度 -4.0，角速度 -0.4），
    # 作为平台门控放宽自然前倾后的加法式兜底护栏，防止向前摔倒。
    angle_margin = max(0.0, abs(hull_angle) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -4.0 * angle_margin ** 2
        - 0.4 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：保持不动。
    vertical_margin = max(0.0, abs(float(next_obs[3])) - 0.6)
    vertical_oscillation_penalty = -0.3 * vertical_margin ** 2

    total_reward = (
        forward_velocity_reward
        + upright_stability_penalty
        + vertical_oscillation_penalty
    )

    components = {
        "forward_velocity_reward": float(forward_velocity_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "vertical_oscillation_penalty": float(vertical_oscillation_penalty),
    }

    return float(total_reward), components