def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v2 reward: 强化前向速度驱动，保持躯干/身体稳定性正则。
    # 诊断：v1 已稳定行走（balance_penalty 仅 2.5% magnitude_share），
    # 剩余缺口是前进速度不够快。因此 L1 提高 forward_reward 系数，
    # 让 speed/stability 权衡点向"更快"移动。
    # info 为空，不使用；无显式 success/failure 标志，不包含 terminal reward。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.5 * next_obs[2]

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