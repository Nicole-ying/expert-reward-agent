def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v7 reward (L2): 主信号从「无界线性速度」改为「有界饱和速度」，根除高速追逐。
    #
    # 证据：iter6(forward=1.75 线性) magnitude_share=97.2% 支配奖励，checkpoint
    # 在快而摔/慢而稳两盆地间震荡（-30→-38→115→183→-80→-89→-35→-85→174→91），
    # final=120、score_range=[-95,299] 双峰，多数种子高速摔倒。
    # iter3(2.0 线性) cp70~80 达 299.7/302.6 但 final 脆；iter2(1.5 线性) 最稳 285。
    # 线性系数轴 1.0/1.5/1.75/2.0 已穷尽；本轮改主信号形态为 tanh 饱和，
    # 把「越快越好」改为「达到巡航速度后边际收益衰减」，上限约 2.0/step。
    #
    # balance 保持 iter2 验证过的平滑二次形态，不重蹈 iter4 死区铰链 / iter5 四次尾项覆辙。
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
    forward_reward = 2.0 * tanh_v

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