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

    # 4. 速度方向引导（新组件，替代 landing）
    #    奖励「速度方向指向目标」，提供即时正向导航梯度
    #    距离衰减：远处需要导航引导，近处让减速信号接管
    if old_dist > 0.05:
        # 单位方向向量：从当前位置指向目标 (0,0)
        dir_x = -x / old_dist
        dir_y = -y / old_dist

        # 速度方向与目标方向的余弦相似度
        spd = (next_vx ** 2 + next_vy ** 2) ** 0.5 + 0.01
        alignment = (next_vx * dir_x + next_vy * dir_y) / spd  # [-1, 1]

        # 距离衰减权重：远处=1.0，距离0→0（线性衰减，5单位外满权重）
        distance_weight = min(1.0, old_dist / 5.0)

        w_guidance = 0.5
        guidance_reward = w_guidance * distance_weight * alignment
    else:
        guidance_reward = 0.0

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + guidance_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "velocity_guidance": guidance_reward
    }
    return float(total_reward), components