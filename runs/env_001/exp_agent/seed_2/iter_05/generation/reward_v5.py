def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束 ──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y) + abs(angle) + abs(angular_vel))

    # ── 4. 二值着陆引导信号（回到 iter 2 的 binary 结构）──
    # 原因：连续乘积形式（iter 3-4）奖励"部分满足"，得分持续下降
    # binary 信号只在全部条件同时满足时触发，提供清晰的"成功悬崖"
    # 用 training_progress 做小幅退火：早期 bonus=0.5 鼓励发现着陆区，后期降到 0.3
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    near_target = dist_new < 0.15
    low_speed = speed < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    # training_progress: 0.0 → bonus=0.5, 1.0 → bonus=0.3
    bonus_magnitude = 0.5 - 0.2 * training_progress
    soft_landing_proxy = bonus_magnitude if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 5. 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components