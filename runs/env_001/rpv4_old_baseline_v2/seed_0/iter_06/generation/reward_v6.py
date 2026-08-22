def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # 1. 主正向：距离势场塑形，向目标 (0,0) 移动
    d_prev = (x * x + y * y) ** 0.5
    d_curr = (nx * nx + ny * ny) ** 0.5
    dist_reward = 8.0 * (d_prev - d_curr)

    # 2. 软着陆速度惩罚（仅在地面附近生效）
    ground_factor = 1.0 if ny < 1.0 else 0.0
    vel_penalty = -0.4 * (abs(nvx) + abs(nvy)) * ground_factor

    # 3. 角度偏离惩罚
    angle_penalty = -0.3 * abs(nangle)

    # 4. 接触奖励（降低稀疏性）
    if nlc > 0 and nrc > 0 and ny < 0.3 and abs(nangle) < 0.2 and abs(nvx) < 0.2 and abs(nvy) < 0.2:
        contact_bonus = 10.0
    elif nlc > 0 or nrc > 0:
        contact_bonus = 0.5
    else:
        contact_bonus = 0.0

    # 5. 燃油消耗惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. 极轻生存步数惩罚
    step_penalty = -0.02

    total = (
        dist_reward
        + vel_penalty
        + angle_penalty
        + contact_bonus
        + fuel_penalty
        + step_penalty
    )

    components = {
        "dist_reward": dist_reward,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "step_penalty": step_penalty,
    }
    return float(total), components