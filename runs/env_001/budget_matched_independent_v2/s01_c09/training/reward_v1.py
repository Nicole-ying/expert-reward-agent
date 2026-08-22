def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position: 水平坐标相对于目标
    y = next_obs[1]          # y_position: 垂直坐标相对于着陆点高度
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志
    
    # 从 obs 提取当前状态用于动作惩罚
    current_x = obs[0]
    current_y = obs[1]
    
    # 计算距离和速度
    distance = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    
    # 1. 接近目标奖励：鼓励向目标移动
    # 使用负指数形式，距离越近奖励越大
    approach_reward = 2.0 * (2.718281828 ** (-0.5 * distance))
    
    # 2. 速度控制奖励：鼓励在接近目标时减速
    # 当距离较远时允许较高速度，距离近时惩罚高速
    speed_penalty = -0.5 * speed * (1.0 - 2.718281828 ** (-0.3 * distance))
    
    # 3. 姿态稳定奖励：鼓励保持直立姿态
    # 角度偏离垂直方向越少越好，角速度越小越好
    angle_penalty = -0.3 * (angle ** 2 + 0.1 * ang_vel ** 2)
    
    # 4. 着陆接触奖励：鼓励双脚同时接触
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 1.0 * both_contact
    
    # 5. 动作效率惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.2
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.1
    
    # 6. 进度奖励：根据训练进度调整探索-利用平衡
    # 早期更注重探索（接近目标），后期更注重精确控制
    exploration_bonus = 0.5 * (1.0 - training_progress) * (2.718281828 ** (-0.2 * distance))
    precision_bonus = 0.5 * training_progress * (1.0 - speed / (1.0 + speed))
    
    # 计算总奖励
    total_reward = (approach_reward + speed_penalty + angle_penalty + 
                    contact_reward + action_penalty + 
                    exploration_bonus + precision_bonus)
    
    components = {
        'approach_reward': approach_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'exploration_bonus': exploration_bonus,
        'precision_bonus': precision_bonus
    }
    
    return float(total_reward), components