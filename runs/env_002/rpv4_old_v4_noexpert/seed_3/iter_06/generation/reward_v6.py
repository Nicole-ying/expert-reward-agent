def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v6 reward: L1 微调 forward 系数 1.5 -> 1.75，删除 iter5 的四次尾项。
    #
    # 证据：iter2（forward=1.5 + 1.0*tilt² + 0.2*angvel² + 0.1*vertvel²）是历史最佳 285.19。
    # iter3（forward=2.0）checkpoint cp70~80 达 299.7/302.6，但 final 掉到 281.28，
    # 且 score_range 出现摔倒回合 -> 2.0 太快、最终策略脆弱。
    # iter5 在 iter2 基座上加了 4.0*tilt**4，balance_penalty 在 cp50/cp60 飙到 -189/-206，
    # forward ep_sum 从 756 掉到 606，agent 减速避险，score 掉到 180。
    #
    # 本轮 L1（单组件）：仅把 forward 系数取 1.5 与 2.0 之间的新值 1.75，
    # 同时恢复 iter2 验证过的平滑二次平衡项（不再叠四次尾项），
    # 在不引入新平衡形态的前提下把速度/稳定性权衡点推向略快。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.75 * next_obs[2]

    # 平滑连续平衡正则（iter2 精确形态）：二次基座，全程有梯度，无死区、无悬崖。
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