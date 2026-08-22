def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]       # x_position relative to target
    y = next_obs[1]       # y_position relative to pad height
    vx = next_obs[2]      # x_velocity
    vy = next_obs[3]      # y_velocity
    angle = next_obs[4]   # body_angle
    ang_vel = next_obs[5] # angular_velocity
    left_contact = next_obs[6]   # left support contact flag
    right_contact = next_obs[7]  # right support contact flag

    # 从 obs 提取上一时刻信号用于速度变化
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，接近时惩罚速度
    speed_penalty = -0.05 * speed * (1.0 / (1.0 + distance * 0.5))

    # 3. 角度奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时着地
    both_contact = left_contact * right_contact  # 1.0 if both, 0.0 otherwise
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=no_engine, 1=left, 2=main, 3=right
    engine_used = 1.0 if action != 0 else 0.0
    action_penalty = -0.02 * engine_used

    # 7. 速度变化奖励：鼓励平稳减速（负加速度）
    accel = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
    smoothness_penalty = -0.01 * accel

    # 8. 进度奖励：当接近目标且速度低时给予额外奖励
    settled_bonus = 0.0
    if distance < 0.5 and speed < 0.5 and both_contact > 0.5:
        settled_bonus = 1.0

    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    smoothness_penalty + settled_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "settled_bonus": settled_bonus,
    }

    return float(total_reward), components