def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v4 reward: 主信号仍为前向速度，但把"稳定性"从全局弱平方惩罚
    # 改为"摔倒风险"死区铰链惩罚。
    #
    # 证据：iter3 最终评估 score_range=[-51.748, 300.544] 且
    # early_terminal(<150 且 score<-50)=0/20，说明存在"走得很好但后期摔倒"
    # 的回合（官方摔倒给 -100 terminal penalty，把整轮拉到负），
    # 非摔倒回合均值已约 298.8。缺口在摔倒，不在速度。
    # 此前奖励只有 1.0*hull_angle^2 的弱平滑惩罚，且 info 为空，
    # agent 训练时感知不到官方 -100，缺少临近摔倒的强信号。
    #
    # 本轮 L2：把 tilt 惩罚从"全量程平方"改为"死区铰链平方"，
    # 仅在 |hull_angle| > 0.4 rad 后急剧增强，用观测重构摔倒惩罚梯度。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 摔倒风险：死区铰链，只惩罚超过 0.4 rad 的躯干倾斜。
    # 正常步态倾角约 0.1~0.15 rad，不会被误伤；临近摔倒时惩罚急剧上升。
    tilt = abs(next_obs[0])
    fall_risk_penalty = 40.0 * max(0.0, tilt - 0.4) ** 2

    # 其余稳定正则：角速度与垂直速度（弱）。
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty

    total_reward = forward_reward - fall_risk_penalty - stability_penalty

    components = {
        "forward_reward": forward_reward,
        "fall_risk_penalty": -fall_risk_penalty,
        "stability_penalty": -stability_penalty,
    }

    return float(total_reward), components