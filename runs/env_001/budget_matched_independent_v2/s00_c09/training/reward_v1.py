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

    # 从 obs 提取上一时刻位置（用于计算速度变化，辅助奖励设计）
    prev_x = obs[0]
    prev_y = obs[1]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离，每步最大约 -0.1*max_distance

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx**2 + vy**2) ** 0.5
    # 当距离较远时允许较高速度，接近时要求低速
    speed_penalty = -0.05 * speed * (1.0 / (1.0 + distance * 0.5))

    # 3. 姿态奖励：鼓励直立（角度接近0）
    # 角度归一化到 [-pi, pi]，这里假设角度范围合理
    angle_penalty = -0.02 * (angle**2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel**2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:
        action_penalty = -0.02  # 姿态引擎小惩罚
    elif action == 2:
        action_penalty = -0.05  # 主引擎较大惩罚

    # 7. 进度奖励：如果向目标移动给予正向激励
    # 计算位置变化方向
    dx = prev_x - x  # 向目标移动时 dx 为正（如果目标在原点）
    dy = prev_y - y
    # 简化：如果距离减小则给予小奖励
    prev_distance = (prev_x**2 + prev_y**2) ** 0.5
    progress_reward = 0.0
    if distance < prev_distance:
        progress_reward = 0.02 * (prev_distance - distance)

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_reward)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
    }

    return float(total_reward), components