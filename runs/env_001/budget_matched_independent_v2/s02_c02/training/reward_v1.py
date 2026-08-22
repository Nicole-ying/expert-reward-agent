def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆台高度）
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
    distance_reward = -0.1 * distance  # 线性惩罚，每步减少距离

    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近越强调减速
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    # 角度归一化到 [-pi, pi] 范围，这里假设角度以弧度表示
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚，鼓励小角度

    # 4. 角速度惩罚：鼓励稳定（减少旋转）
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（节省燃料）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.1
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.05
    # action == 0 无惩罚

    # 7. 速度变化惩罚：鼓励平滑运动（避免剧烈加速/减速）
    # 计算速度变化量（近似）
    speed_change = abs(speed - (prev_vx ** 2 + prev_vy ** 2) ** 0.5) if 'prev_vx' in dir() else 0.0
    # 简化：使用当前速度作为代理（因为无法直接获取上一时刻速度）
    # 改用动作变化惩罚：如果连续使用引擎则惩罚
    # 这里用动作本身作为代理

    # 8. 进度奖励：当接近目标且速度很小时给予额外奖励
    settled_bonus = 0.0
    if distance < 0.5 and speed < 0.2 and abs(angle) < 0.1 and abs(ang_vel) < 0.1:
        settled_bonus = 1.0  # 成功稳定在目标区域

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + settled_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'settled_bonus': settled_bonus,
    }

    return float(total_reward), components