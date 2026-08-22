def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志
    
    # 从 obs 提取上一时刻的位置（用于计算速度变化惩罚）
    prev_x = obs[0]
    prev_y = obs[1]
    
    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -distance * 0.5  # 线性惩罚，每步减少距离
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_penalty_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 2.0))
    speed_penalty = -speed * speed_penalty_weight * 0.8
    
    # 3. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -abs(angle) * 0.3
    
    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -abs(ang_vel) * 0.2
    
    # 5. 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = both_contact * 1.0
    
    # 6. 动作惩罚：鼓励少用引擎（动作1、2、3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15
    
    # 7. 速度变化惩罚：鼓励平滑运动（避免剧烈抖动）
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_change = abs(speed - prev_speed)
    smoothness_penalty = -speed_change * 0.1
    
    # 8. 接近目标时的速度奖励：当非常接近目标时，鼓励完全静止
    if distance < 0.3:
        stillness_bonus = -speed * 2.0  # 强烈惩罚速度
    else:
        stillness_bonus = 0.0
    
    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    smoothness_penalty + stillness_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'smoothness_penalty': smoothness_penalty,
        'stillness_bonus': stillness_bonus,
    }
    
    return float(total_reward), components