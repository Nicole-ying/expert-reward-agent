def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0  # 权重 2.0，鼓励有效靠近

    # ---- 稳定/安全约束：抑制高速、大角度和高角速度 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    # 轻量级线性惩罚，防止剧烈翻滚和高速撞击
    stability_penalty_value = (
        0.1 * abs(vx)
        + 0.1 * abs(vy)
        + 0.1 * abs(angle)
        + 0.05 * abs(angular_vel)
    )
    # 组件记录为负贡献
    stability_penalty = -stability_penalty_value

    # ---- 任务完成近似信号：多条件组合的软着陆 bonus ----
    # 条件：双脚接触 + 位置接近垫中心 + 低速 + 姿态平稳
    soft_landing_bonus = 0.0
    if (next_obs[6] == 1.0 and next_obs[7] == 1.0
            and abs(next_obs[0]) < 0.3
            and abs(next_obs[1]) < 0.1
            and abs(vx) < 0.3
            and abs(vy) < 0.3
            and abs(angle) < 0.2):
        soft_landing_bonus = 1.0  # 为完成提供明确的溢价信号

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,  # 负值
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components