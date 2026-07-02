def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 诊断与修改理由 ===
    # 问题：stability_penalty (-0.068) 是 shaping_reward (0.026) 的 2.6 倍，
    #       agent 在任何状态下都被净惩罚，导致 episode 短 (71步)、全部 crash。
    # 修改1：stability 系数降低 10x，使其成为弱背景信号（~27% of shaping）。
    # 修改2：gamma 1.0 消除静止时的 (1-γ)*dist 虚假奖励。
    # 修改3：components key 与公式变量名一致，移除 total_reward。

    # 1. 势能塑形：Φ = -distance, gamma = 1.0
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # gamma=1.0 → shaping = dist_current - dist_next，纯进度信号
    shaping_reward = dist_current - dist_next

    # 2. 稳定性惩罚：轻量背景约束
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # 系数降低 10x：原 0.05/0.02/0.05 → 现 0.005/0.002/0.005
    w_vel = 0.005
    w_ang = 0.002
    w_angle = 0.005

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    # components 只放总公式中直接出现的变量，key 与变量名一致
    components = {
        "shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
    }
    return float(total_reward), components