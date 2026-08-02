def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 1. 主进展信号：向中心靠近
    progress_delta = 8.0 * (dist - next_dist)

    # 2. 接触奖励：双足接触平台即给正向
    contact_reward = 0.3 * (nleft + nright)

    # 3. 速度约束(higne)：仅在速度过大时惩罚
    speed_threshold = 0.6
    vx_violation = max(0.0, abs(nvx) - speed_threshold)
    vy_violation = max(0.0, abs(nvy) - speed_threshold)
    speed_penalty = -0.1 * (vx_violation + vy_violation)

    # 4. 角度约束(higne)
    angle_threshold = 0.15
    angle_violation = max(0.0, abs(nangle) - angle_threshold)
    angle_penalty = -0.2 * angle_violation

    # 5. 角速度约束(higne)
    angvel_threshold = 0.3
    angvel_violation = max(0.0, abs(nangvel) - angvel_threshold)
    angvel_penalty = -0.1 * angvel_violation

    # 6. 着陆奖励：接近中心且有接触
    landing_bonus = 0.0
    if next_dist < 0.3 and (nleft + nright) >= 1.0:
        landing_bonus = 0.5

    # 7. 边界预警：出界风险软约束
    boundary_warning = -0.5 * max(0.0, next_dist - 1.0)

    total_reward = (
        progress_delta +
        contact_reward +
        speed_penalty +
        angle_penalty +
        angvel_penalty +
        landing_bonus +
        boundary_warning
    )

    components = {
        'progress_delta': progress_delta,
        'contact_reward': contact_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'landing_bonus': landing_bonus,
        'boundary_warning': boundary_warning
    }

    return float(total_reward), components