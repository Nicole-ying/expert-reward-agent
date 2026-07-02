def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励（不动）──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（不动）──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 连续软着陆引导信号 ──
    # 上一轮：coefficient=0.5, prox_threshold=0.3 → mean=0.237, ratio=83x progress
    # 本轮：coefficient→0.08, prox_threshold→0.2，目标 ratio≈5-10x
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    prox_factor = max(0.0, 1.0 - dist_new / 0.2)       # dist=0→1, dist≥0.2→0（收紧）
    speed_factor = max(0.0, 1.0 - speed / 0.5)          # speed=0→1, speed≥0.5→0
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)    # angle=0→1, |angle|≥0.3→0
    leg_factor = 0.5 * (left_contact + right_contact)   # 两腿→1, 单腿→0.5, 无→0

    # 系数从 0.5 降至 0.08，大幅削弱 landing 信号的主导地位
    soft_landing_continuous = 0.08 * prox_factor * speed_factor * angle_factor * leg_factor

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_continuous

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_continuous": soft_landing_continuous,
        "total_reward": total_reward
    }

    return float(total_reward), components