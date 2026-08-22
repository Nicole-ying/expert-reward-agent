def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 距离与速度
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5

    # 1. 进度奖励：大幅提升权重，鼓励向目标移动
    progress = dist_prev - dist_next
    w_progress = 12.0

    # 2. 全局速度惩罚（线性，轻量，避免过早扼杀探索）
    speed_penalty_global = speed_norm
    w_speed_global = 0.03

    # 3. 近端速度引导：用平滑函数鼓励靠近目标时减速
    near_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed_penalty_near = near_factor * speed_norm
    w_speed_near = 0.1

    # 4. 垂直速度惩罚，温和防止重着陆
    ground_prox = 1.0 / (1.0 + abs(y) + 1e-5)
    vert_speed_penalty = ground_prox * abs(yv)
    w_vert_speed = 0.08

    # 5. 接近奖励：平滑的高斯型奖励，即使没有着陆也能获得正反馈
    proximity_reward = 3.0 / (1.0 + 20.0 * dist_next ** 2)

    # 6. 软着陆奖励：同时要求接近目标且速度极低
    soft_landing = 8.0 / (1.0 + 50.0 * dist_next ** 2) * (1.0 / (1.0 + 50.0 * speed_norm ** 2))

    # 7. 接触着陆奖励：双足接触且靠近目标时给予强奖励
    contact_bonus = 15.0 * both_legs / (1.0 + 50.0 * dist_next ** 2)

    # 8. 姿态正则化（保持平稳）
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.001

    total_reward = (
        w_progress * progress
        - w_speed_global * speed_penalty_global
        - w_speed_near * speed_penalty_near
        - w_vert_speed * vert_speed_penalty
        + proximity_reward
        + soft_landing
        + contact_bonus
        - w_ori * angle_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty_global': -w_speed_global * speed_penalty_global,
        'speed_penalty_near': -w_speed_near * speed_penalty_near,
        'vert_speed_penalty': -w_vert_speed * vert_speed_penalty,
        'proximity': proximity_reward,
        'soft_landing': soft_landing,
        'contact_bonus': contact_bonus,
        'orientation': -w_ori * angle_penalty,
    }

    return float(total_reward), components