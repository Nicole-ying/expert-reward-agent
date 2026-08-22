def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]      # x_position relative to target pad
    y = next_obs[1]      # y_position relative to pad height
    vx = next_obs[2]     # x_velocity
    vy = next_obs[3]     # y_velocity
    angle = next_obs[4]  # body_angle
    ang_vel = next_obs[5] # angular_velocity
    left_contact = next_obs[6]   # left support contact flag
    right_contact = next_obs[7]  # right support contact flag

    # 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离，每步梯度稳定

    # 速度惩罚：鼓励减速靠近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))

    # 姿态奖励：鼓励直立（角度为0），减少角速度
    angle_penalty = -0.02 * (angle ** 2 + ang_vel ** 2)

    # 接触奖励：鼓励双脚同时接触目标平台
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    action_penalty = -0.01 if action != 0 else 0.0

    # 存活奖励：鼓励持续探索直到成功
    alive_bonus = 0.01

    # 总奖励
    total_reward = distance_reward + speed_penalty + angle_penalty + contact_reward + action_penalty + alive_bonus

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "alive_bonus": alive_bonus,
    }

    return float(total_reward), components