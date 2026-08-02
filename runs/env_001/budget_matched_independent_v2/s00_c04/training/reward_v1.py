def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆台高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步信号（用于动作惩罚）
    current_x = obs[0]
    current_y = obs[1]
    current_vx = obs[2]
    current_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.5 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近越强调减速
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作惩罚：鼓励节能（减少不必要的引擎使用）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15

    # 7. 进度奖励：如果距离显著减小，给予正向激励
    prev_distance = (current_x ** 2 + current_y ** 2) ** 0.5
    distance_delta = prev_distance - distance
    progress_reward = 0.3 * max(0.0, distance_delta)

    # 8. 稳定着陆奖励：当接近目标且速度很小时给予额外奖励
    stable_landing_bonus = 0.0
    if distance < 0.5 and speed < 0.3 and both_contact:
        stable_landing_bonus = 3.0

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_reward + stable_landing_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_reward': progress_reward,
        'stable_landing_bonus': stable_landing_bonus,
    }

    return float(total_reward), components