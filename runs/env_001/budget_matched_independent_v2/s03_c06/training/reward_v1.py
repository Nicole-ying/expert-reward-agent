def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # 水平位置（相对于目标）
    y = next_obs[1]          # 垂直位置（相对于着陆台高度）
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机体角度
    ang_vel = next_obs[5]    # 角速度
    left_contact = next_obs[6]   # 左侧接触标志
    right_contact = next_obs[7]  # 右侧接触标志
    
    # 从 obs 提取上一时刻的状态用于速度变化计算
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # 1. 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    speed_penalty = -0.05 * speed
    
    # 3. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.2 * (angle ** 2)
    
    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)
    
    # 5. 着陆奖励：当两个支撑点都接触时给予奖励
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact
    
    # 6. 引擎使用惩罚：根据动作类型惩罚引擎使用
    # 动作: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    engine_penalty = 0.0
    if action == 2:  # 主引擎
        engine_penalty = -0.3
    elif action in [1, 3]:  # 姿态引擎
        engine_penalty = -0.1
    
    # 7. 速度变化奖励：鼓励有效减速（速度减小）
    prev_speed = (prev_vx ** 2 + prev_vy ** 2) ** 0.5
    speed_change = prev_speed - speed
    deceleration_reward = 0.2 * max(0, speed_change)
    
    # 8. 接近目标时的速度调节：距离越近，越应该减速
    distance_factor = 2.718281828 ** (-0.5 * distance)
    proximity_speed_penalty = -0.1 * speed * distance_factor
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                   ang_vel_penalty + contact_reward + engine_penalty + 
                   deceleration_reward + proximity_speed_penalty)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'engine_penalty': engine_penalty,
        'deceleration_reward': deceleration_reward,
        'proximity_speed_penalty': proximity_speed_penalty
    }
    
    return float(total_reward), components