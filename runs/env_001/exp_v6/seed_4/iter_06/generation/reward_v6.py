def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 诊断与修改理由 ===
    # 问题：上轮 shaping_reward 均值仅 0.014/step，信号强度不到环境惩罚 (-0.91/step) 的 2%。
    #       agent 主要感受环境惩罚，无法学到有效策略 → 80% crash。
    # 修改（层次2）：采用推荐 Φ=-(distance + speed + |angle|) 组合势能，
    #       单次势能差同时反映"靠近 + 减速 + 扶正"三项进度，信号更丰富。
    #       scale=20 将信号放大到与 env reward 可比量级。
    #       移除独立 stability_penalty——稳定目标已融入势能差，减少组件数量。

    # 计算当前与下一时刻的势能要素
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    speed_current = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5

    angle_current = abs(obs[4])
    angle_next = abs(next_obs[4])

    # 组合势能：Φ = -(distance + speed + |angle|)
    phi_current = -(dist_current + speed_current + angle_current)
    phi_next = -(dist_next + speed_next + angle_next)

    # γ=1.0 势能塑形：F = Φ(next) - Φ(current) = 各项减少量之和
    F = phi_next - phi_current
    scale = 20.0
    shaping_reward = scale * F

    total_reward = shaping_reward

    components = {
        "shaping_reward": shaping_reward,
    }
    return float(total_reward), components