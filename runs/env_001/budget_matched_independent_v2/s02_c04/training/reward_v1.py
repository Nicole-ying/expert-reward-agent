def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（当前步执行动作后的状态）
    x = next_obs[0]          # 水平位置（相对于目标）
    y = next_obs[1]          # 垂直位置（相对于着陆点高度）
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机体角度
    ang_vel = next_obs[5]    # 角速度
    left_contact = next_obs[6]   # 左侧支撑接触标志
    right_contact = next_obs[7]  # 右侧支撑接触标志
    
    # 从 obs 提取上一步的状态（用于计算速度变化）
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # 1. 距离奖励：鼓励接近目标
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx**2 + vy**2) ** 0.5
    speed_penalty = -0.05 * speed
    
    # 3. 着陆速度惩罚：当接近目标时，对垂直速度施加额外惩罚
    vertical_speed_penalty = -0.1 * max(0, vy) * (1.0 / (1.0 + distance))
    
    # 4. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.2 * abs(angle)
    
    # 5. 角速度惩罚：鼓励稳定姿态
    angular_velocity_penalty = -0.05 * abs(ang_vel)
    
    # 6. 接触奖励：鼓励双脚同时接触（稳定着陆）
    contact_reward = 0.5 * (left_contact + right_contact)
    
    # 7. 动作惩罚：鼓励少用引擎
    action_penalty = -0.02 * (action != 0)  # 非零动作表示使用了引擎
    
    # 8. 进度奖励：当接近目标且速度低时给予额外奖励
    progress_bonus = 0.0
    if distance < 0.5 and speed < 0.5:
        progress_bonus = 1.0
    
    # 9. 加速度惩罚：鼓励平滑运动（通过速度变化量近似）
    acc_x = abs(vx - prev_vx)
    acc_y = abs(vy - prev_vy)
    acceleration_penalty = -0.01 * (acc_x + acc_y)
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + vertical_speed_penalty + 
                    angle_penalty + angular_velocity_penalty + contact_reward + 
                    action_penalty + progress_bonus + acceleration_penalty)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'vertical_speed_penalty': vertical_speed_penalty,
        'angle_penalty': angle_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_bonus': progress_bonus,
        'acceleration_penalty': acceleration_penalty
    }
    
    return float(total_reward), components