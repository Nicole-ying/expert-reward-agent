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

    # 从 obs 提取当前步的信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励直立姿态（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定，减少旋转
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1=左转, 2=主引擎, 3=右转）
    # 动作0是无引擎，不惩罚
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎（推力大，惩罚更大）
        action_penalty = -0.05

    # 7. 进度奖励：如果向目标移动则给予正向奖励
    # 计算距离变化（负值表示接近）
    prev_distance = (x_prev**2 + y_prev**2) ** 0.5
    distance_delta = prev_distance - distance  # 正数表示接近
    progress_reward = 0.2 * max(0, distance_delta)

    # 8. 速度变化奖励：鼓励减速（速度减小）
    prev_speed = (vx_prev**2 + vy_prev**2) ** 0.5
    speed_delta = prev_speed - speed  # 正数表示减速
    deceleration_reward = 0.1 * max(0, speed_delta)

    # 汇总
    total_reward = (
        distance_reward
        + speed_penalty
        + angle_penalty
        + ang_vel_penalty
        + contact_reward
        + action_penalty
        + progress_reward
        + deceleration_reward
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
        "deceleration_reward": deceleration_reward,
    }

    return float(total_reward), components