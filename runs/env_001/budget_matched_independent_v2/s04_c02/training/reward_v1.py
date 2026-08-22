def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]
    angle_prev = obs[4]
    ang_vel_prev = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚，每步减少距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.02 + 0.08 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.05 * abs(angle)

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.02 * abs(ang_vel)

    # 5. 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    engine_used = 1.0 if action != 0 else 0.0
    fuel_penalty = -0.02 * engine_used

    # 7. 进度奖励：如果距离显著减小，给予额外奖励
    distance_prev = (x_prev**2 + y_prev**2) ** 0.5
    distance_delta = distance_prev - distance
    progress_reward = 0.2 * max(0, distance_delta)

    # 8. 速度变化奖励：如果速度在合理范围内减小，给予奖励
    speed_prev = (vx_prev**2 + vy_prev**2) ** 0.5
    speed_delta = speed_prev - speed
    # 只在接近目标时奖励减速
    if distance < 2.0:
        deceleration_reward = 0.1 * max(0, speed_delta)
    else:
        deceleration_reward = 0.0

    # 汇总
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + fuel_penalty + 
                    progress_reward + deceleration_reward)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty,
        'progress_reward': progress_reward,
        'deceleration_reward': deceleration_reward,
    }

    return float(total_reward), components