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

    # 从 obs 提取当前步的信号（用于动作惩罚）
    x_curr = obs[0]
    y_curr = obs[1]
    vx_curr = obs[2]
    vy_curr = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_reward = -0.5 * dist  # 线性惩罚距离，每步梯度稳定

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，接近时惩罚速度
    speed_penalty = -0.3 * speed * (1.0 / (1.0 + dist * 0.5))

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.2 * abs(angle)  # 角度偏离惩罚

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * abs(ang_vel)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15

    # 7. 进度奖励：当接近目标且速度低时给予额外奖励
    # 距离<0.5且速度<0.3视为"接近稳定"
    near_target = dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(angle) < 0.2
    if near_target and low_speed and stable_angle:
        progress_bonus = 1.0
    else:
        progress_bonus = 0.0

    # 汇总奖励
    total_reward = (dist_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_bonus)

    components = {
        "dist_reward": dist_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_bonus": progress_bonus,
    }

    return float(total_reward), components