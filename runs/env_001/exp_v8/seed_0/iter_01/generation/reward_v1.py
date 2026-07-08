def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为 (0,0)，平台中心
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 当前位置与目标距离
    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    # 下一时刻与目标距离
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标
    progress_reward = d_curr - d_next

    # 2. 轻量稳定约束，基于下一状态
    vx, vy = next_obs[2], next_obs[3]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_v = abs(next_obs[5])

    stability_penalty = (
        -0.05 * speed
        - 0.1 * angle
        - 0.05 * angular_v
    )

    # 3. 软着陆完成近似信号
    soft_landing_proxy = 0.0
    if (
        d_next < 0.15
        and speed < 0.2
        and angle < 0.1
        and next_obs[6] == 1.0
        and next_obs[7] == 1.0
    ):
        soft_landing_proxy = 1.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components