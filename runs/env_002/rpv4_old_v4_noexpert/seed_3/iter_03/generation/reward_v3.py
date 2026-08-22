def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v3 reward: 延续 v1->v2 已验证的语义方向——加强前向速度驱动。
    # 证据：v2 将 forward_reward 系数从 1.0 提到 1.5 后，score 从 278.88 升到 285.19，
    # forward_reward ep_sum 从 488.8 升到 756.2，而 balance_penalty magnitude_share 仅 4.1%。
    # 说明稳定性已不是瓶颈，剩余缺口是前进速度略低。当前距目标 300 差约 4.9%，做 L1 微调：
    # 仅把 forward_reward 系数 1.5 -> 2.0，继续把 speed/stability 权衡点推向更快。
    # info 为空，不使用；无显式 success/failure 标志，不包含 terminal reward。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 2.0 * next_obs[2]

    # 稳定性/防摔倒约束：惩罚躯干倾斜、剧烈角速度和过度垂直运动。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components