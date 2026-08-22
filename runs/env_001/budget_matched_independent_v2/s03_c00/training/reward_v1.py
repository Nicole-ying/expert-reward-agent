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

    # 从 obs 提取上一时刻的位置（用于计算速度变化，辅助判断是否减速）
    prev_x = obs[0]
    prev_y = obs[1]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 2.0)))

    # 3. 角度奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.02 * (angle ** 2)

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触目标平台
    contact_bonus = 0.0
    if left_contact > 0.5 and right_contact > 0.5:
        contact_bonus = 0.5  # 双脚同时接触给予奖励
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_bonus = 0.1  # 单脚接触给予少量奖励

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=no_engine, 1=left, 2=main, 3=right
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎
        action_penalty = -0.05

    # 7. 速度变化奖励：鼓励减速（接近目标时）
    # 计算速度变化方向（从obs到next_obs）
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_change = prev_speed - speed  # 正数表示减速
    deceleration_reward = 0.0
    if speed_change > 0 and distance < 2.0:  # 仅在接近目标时奖励减速
        deceleration_reward = 0.1 * speed_change

    # 8. 存活奖励：鼓励持续尝试
    alive_bonus = 0.01

    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_bonus + action_penalty + 
                    deceleration_reward + alive_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "deceleration_reward": deceleration_reward,
        "alive_bonus": alive_bonus,
    }

    return float(total_reward), components