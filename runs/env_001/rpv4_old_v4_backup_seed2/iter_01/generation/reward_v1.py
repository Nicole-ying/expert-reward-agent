def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：双脚接触 + 位置对中 + 低速 + 小角度
    x_thresh = 0.3
    y_thresh = 0.3
    vel_thresh = 0.5
    angle_thresh = 0.2  # rad ≈ 11°

    contact_factor = next_left_contact * next_right_contact  # 0 或 1
    pos_x_factor = max(0.0, 1.0 - abs(next_x) / x_thresh)
    pos_y_factor = max(0.0, 1.0 - abs(next_y) / y_thresh)
    vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / vel_thresh)
    angle_factor = max(0.0, 1.0 - abs(next_angle) / angle_thresh)

    landing_multiplier = (contact_factor * pos_x_factor * pos_y_factor
                          * vel_factor * angle_factor)
    w_land = 2.0
    landing_reward = w_land * landing_multiplier

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components