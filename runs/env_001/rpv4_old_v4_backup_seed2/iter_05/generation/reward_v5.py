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

    # 4. 着陆奖励：proximity门控 + contact放大
    #    设计要点：
    #    - proximity 门控保证可发现性（进入目标附近即有信号）
    #    - contact 放大提供着陆激励（双脚接触时 5x 奖励）
    #    - 低权重防止悬停 exploit（无接触时 max=0.5/步，远低于 iter2 的 3.0/步）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.0)  # Manhattan < 1.0

    if proximity > 0.0:
        # 着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.3)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 0.5)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.3)

        # 接触信号：0 / 0.5 / 1.0
        contact = (next_left_contact + next_right_contact) / 2.0

        # contact 放大倍数：无接触=1x，单脚=3x，双脚=5x
        contact_mult = 1.0 + 4.0 * contact

        # 最终着陆分数：proximity × contact_mult × 平均质量
        landing_score = proximity * contact_mult * (
            pos_quality + vel_quality + angle_quality
        ) / 3.0
    else:
        landing_score = 0.0

    w_land = 0.5
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components