def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前/下一步状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 上一状态
    x_prev, y_prev = obs[0], obs[1]

    # 距离与速度
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5

    # 1. 进展奖励：向目标点靠近
    progress = dist_prev - dist_next
    w_progress = 10.0

    # 2. 轻度高度惩罚（防止长期在高空盘旋）
    height_cost = abs(y) * 0.1

    # 3. 普遍速度惩罚（鼓励全程减速）
    speed_cost = speed_norm * 0.1

    # 4. 姿态正则化（保持机体竖直、稳定）
    orientation_cost = (angle ** 2 + 0.1 * ang_vel ** 2) * 0.01

    # 5. 接触奖励（随接近目标而增强，引导低空接地）
    single_contact = left_contact + right_contact
    # 当距离 < 0.5 时，乘法因子从 1.0 线性增到 4.0
    proximity_mult = 1.0 + 3.0 * max(0.0, 1.0 - dist_next / 0.5)
    contact_reward = (5.0 * single_contact + 10.0 * both_legs) * proximity_mult

    # 6. 垂直速度控制：贴近地面时避免重着陆
    vy_cost = 0.0
    if y < 0.3:
        vy_cost = 10.0 * (yv ** 2)

    # 7. 软着陆大奖（满足条件时大额一次性奖励）
    landing_bonus = 0.0
    if both_legs > 0.5 and dist_next < 0.3 and speed_norm < 0.5:
        landing_bonus = 500.0

    # 8. 微小引擎惩罚（鼓励节能）
    engine_penalty = 0.001 if action != 0 else 0.0

    # 汇总
    total_reward = (
        w_progress * progress
        - height_cost
        - speed_cost
        - orientation_cost
        + contact_reward
        - vy_cost
        + landing_bonus
        - engine_penalty
    )

    components = {
        'progress': w_progress * progress,
        'height_cost': -height_cost,
        'speed_cost': -speed_cost,
        'orientation_cost': -orientation_cost,
        'contact_reward': contact_reward,
        'vy_cost': -vy_cost,
        'landing_bonus': landing_bonus,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components