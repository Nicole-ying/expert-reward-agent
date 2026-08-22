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

    # 从 obs 提取当前状态用于速度变化计算
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，距离近时严格惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 2.0)))

    # 3. 姿态奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 燃料效率惩罚：惩罚使用引擎（动作1=左转, 2=主引擎, 3=右转）
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.02 * engine_used

    # 7. 速度变化奖励：鼓励在接近目标时减速
    speed_prev = (vx_prev ** 2 + vy_prev ** 2) ** 0.5
    speed_change = speed_prev - speed  # 正数表示减速
    deceleration_reward = 0.03 * speed_change * (1.0 / (1.0 + distance * 0.5))

    # 汇总奖励
    total_reward = distance_reward + speed_penalty + angle_penalty + ang_vel_penalty + contact_reward + fuel_penalty + deceleration_reward

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "deceleration_reward": deceleration_reward,
    }

    return float(total_reward), components