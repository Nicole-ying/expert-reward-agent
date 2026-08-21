def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v8 reward (L1): 在 iter7 验证过的 tanh 饱和主信号上，把前向驱动乘子 2.0 -> 2.2。
    #
    # 证据：iter7(tanh, 2.0) 把 iter6 的无界线性灾难(120, 双峰 -95..299)救回 274，
    # 巡航速度 ~0.442 m/s，略低于历史最佳 iter2(线性1.5) 的 ~0.454 m/s。
    # 官方得分近似正比于「速度 x 存活步数」，当前缺口主要在前进驱动略弱。
    # 本轮只做 L1 微调：+10% forward 力度，把速度/稳定性权衡点略推向更快，
    # 同时保留 tanh 饱和（上限 2.2，仍有界），避免重演 iter6 高速追逐摔倒。
    #
    # balance 保持 iter2/iter7 验证过的平滑二次形态，不重蹈 iter4 死区铰链 / iter5 四次尾项覆辙。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：有界前向速度奖励。
    # 健康巡航速度约 0.68 m/s（iter2 最佳 285 的平均速度）。
    # 用 tanh(v/0.7) 使 0~0.7 区间梯度强、超速后边际收益衰减，避免极端速度支配奖励。
    v = next_obs[2]
    v_norm = v / 0.7
    # clamp 归一化输入，避免极端速度导致 exp 溢出
    v_norm = max(-3.0, min(3.0, v_norm))
    e2 = 2.718281828 ** (2.0 * v_norm)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    forward_reward = 2.2 * tanh_v

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