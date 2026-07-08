def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：scale 从 8.0 → 15.0，强化进度引导
    #    骨架推荐 5~20，当前 8.0 下 progress 仅占总 reward ~5%，偏弱
    progress_reward = (d_curr - d_next) * 15.0

    # 2. 软着陆近似信号：连续乘积（保持 best 路线不变）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # 3. 中心邻近奖励：保持上次改动（radius=0.5, max=0.2），不改
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components