def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # 水平位置（相对于目标）
    y = next_obs[1]          # 垂直位置（相对于着陆点高度）
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机体角度
    ang_vel = next_obs[5]    # 角速度
    left_contact = next_obs[6]   # 左侧接触标志
    right_contact = next_obs[7]  # 右侧接触标志
    
    # 从 obs 提取上一时刻的位置（用于计算速度变化）
    prev_x = obs[0]
    prev_y = obs[1]
    
    # 1. 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed
    
    # 3. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)
    
    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.02 * (ang_vel ** 2)
    
    # 5. 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎推力最大
        action_penalty = -0.1
    elif action in [1, 3]:  # 姿态引擎
        action_penalty = -0.05
    
    # 7. 进度奖励：如果正在接近目标且减速，给予额外奖励
    # 计算位置变化方向
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_delta = distance - prev_distance
    progress_reward = 0.0
    if distance_delta < -0.01:  # 正在接近目标
        progress_reward = 0.2 * (-distance_delta)
    
    # 8. 着陆奖励：当双脚接触且速度很小时给予一次性奖励
    landing_bonus = 0.0
    if both_contact > 0.5 and speed < 0.5 and distance < 0.3:
        landing_bonus = 2.0
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                   ang_vel_penalty + contact_reward + action_penalty + 
                   progress_reward + landing_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_reward': progress_reward,
        'landing_bonus': landing_bonus
    }
    
    return float(total_reward), components