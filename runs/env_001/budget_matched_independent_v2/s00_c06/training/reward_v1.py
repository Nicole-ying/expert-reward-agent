def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]       # 水平位置（相对于目标）
    y = next_obs[1]       # 垂直位置（相对于着陆台高度）
    vx = next_obs[2]      # 水平速度
    vy = next_obs[3]      # 垂直速度
    angle = next_obs[4]   # 机体角度
    ang_vel = next_obs[5] # 角速度
    left_contact = next_obs[6]   # 左侧接触标志
    right_contact = next_obs[7]  # 右侧接触标志
    
    # 从 obs 提取上一时刻的状态用于速度变化计算
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # 计算距离和速度
    distance = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5
    
    # 判断是否着陆（两个支撑点都接触）
    landed = left_contact > 0.5 and right_contact > 0.5
    
    # 1. 距离奖励：鼓励接近目标
    # 使用平滑的负距离，距离越近奖励越大
    distance_reward = -0.1 * distance
    
    # 2. 速度奖励：鼓励在接近目标时减速
    # 当距离较远时允许较高速度，距离近时惩罚速度
    speed_penalty = -0.05 * speed * (1.0 / (1.0 + distance * 0.5))
    
    # 3. 着陆奖励：成功着陆给予正向奖励
    landing_reward = 0.0
    if landed:
        # 着陆时速度越小奖励越大
        landing_reward = 2.0 * (1.0 - min(speed / 2.0, 1.0))
    
    # 4. 姿态奖励：鼓励保持直立（角度接近0）
    # 角度以弧度表示，惩罚偏离直立状态
    angle_penalty = -0.02 * (angle ** 2)
    
    # 5. 角速度奖励：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 6. 燃料效率奖励：惩罚不必要的引擎使用
    # 动作1、2、3都会消耗燃料，动作0不消耗
    fuel_penalty = 0.0
    if action == 1 or action == 3:  # 姿态调整引擎
        fuel_penalty = -0.02
    elif action == 2:  # 主引擎
        fuel_penalty = -0.05
    
    # 7. 进度奖励：鼓励持续向目标移动
    # 计算速度在朝向目标方向的分量
    if distance > 0.01:
        # 朝向目标的单位向量
        target_dir_x = -x / distance
        target_dir_y = -y / distance
        # 速度在目标方向上的投影
        approach_speed = vx * target_dir_x + vy * target_dir_y
        # 正向接近给予奖励，远离给予惩罚
        progress_reward = 0.02 * max(approach_speed, 0.0)
    else:
        progress_reward = 0.0
    
    # 8. 接触奖励：鼓励安全接触
    contact_reward = 0.0
    if left_contact > 0.5 or right_contact > 0.5:
        # 有接触时，速度越小奖励越大
        contact_reward = 0.1 * (1.0 - min(speed / 3.0, 1.0))
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + landing_reward + 
                   angle_penalty + ang_vel_penalty + fuel_penalty + 
                   progress_reward + contact_reward)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'landing_reward': landing_reward,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'fuel_penalty': fuel_penalty,
        'progress_reward': progress_reward,
        'contact_reward': contact_reward
    }
    
    return float(total_reward), components