def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v5 reward: 删除 iter4 的死区铰链悬崖，恢复平滑连续平衡正则，
    # 并给 tilt 惩罚增加四次尾项，强化「临近摔倒」区域的连续梯度。
    #
    # 证据：iter4 用 40*max(0,|tilt|-0.4)^2 死区铰链后 score 285.19 -> 29.83。
    # checkpoint cp10 fall_risk=-451 重罚早期探索，cp20~40 agent 站立不动逃避惩罚，
    # 最终 fall_risk_penalty magnitude_share=35.3% 支配奖励，策略在评估种子脆弱。
    # 根因：死区在 0.4 以下无梯度，且 40 系数形成悬崖。
    #
    # 本轮 L2（单组件）：保留 iter2 验证过的 forward=1.5 与二次基座，
    # 把 tilt 惩罚改为「二次基座 + 四次尾项」：正常步态(~0.17 rad)几乎不变，
    # 危险区(|tilt|>0.5)平滑增强、无死区，教会 agent 保持躯干安全余量。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 平滑连续平衡正则：二次基座 + 四次尾项。
    tilt = next_obs[0]
    tilt_base = 1.0 * tilt ** 2          # iter2 验证过的全程连续梯度
    tilt_tail = 4.0 * tilt ** 4          # 危险区快速增强，无死区、无悬崖

    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_base
        + tilt_tail
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components