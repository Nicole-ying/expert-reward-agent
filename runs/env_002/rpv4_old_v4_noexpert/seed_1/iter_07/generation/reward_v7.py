def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干不接近摔倒时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：从纯高斯改为"平台 + 高斯尾部"。
    # 证据：门控宽 0.4->0.6->0.7 持续改善（255.9->287.5->292.0），
    # 但 0.8 触发训练中期塌缩（cp50~cp80 全负）并跌至 216.5。
    # 根因：纯高斯对任何前倾（包括 0.1~0.25 rad 的自然前倾）都乘性打折，
    # 逼迫策略在速度与姿态间不稳定权衡；单纯放宽宽度又削弱了摔倒抑制。
    # 本改法：自然前倾区间（<=0.25 rad）全额奖励，大角度保持 iter_5 已实证的抑制强度
    # （0.7 rad 处 gate=0.368），从而把"前进速度"与"躯干倾斜"两个职责解耦。
    hull_angle = float(next_obs[0])
    lean_margin = max(0.0, abs(hull_angle) - 0.25)
    upright_gate = 2.718281828 ** (-(lean_margin / 0.45) ** 2)

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