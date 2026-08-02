def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]       # x_position: 水平坐标（相对于目标）
    y = next_obs[1]       # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]      # x_velocity: 水平速度
    vy = next_obs[3]      # y_velocity: 垂直速度
    angle = next_obs[4]   # body_angle: 机体角度
    ang_vel = next_obs[5] # angular_velocity: 角速度
    left_contact = next_obs[6]  # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7] # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步的动作（用于燃料惩罚）
    # action 是离散值 0-3，对应 no_engine, left, main, right

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚，每步减少

    # 2. 速度惩罚：鼓励减速（着陆时需要低速）
    speed = (vx**2 + vy**2) ** 0.5
    speed_penalty = -0.05 * speed

    # 3. 姿态奖励：鼓励保持直立（角度接近 0）
    angle_penalty = -0.02 * abs(angle)  # 角度偏离惩罚

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * abs(ang_vel)

    # 5. 燃料惩罚：鼓励少用引擎
    # 动作 1=左引擎, 2=主引擎, 3=右引擎 都消耗燃料
    fuel_penalty = -0.02 if action in [1, 2, 3] else 0.0

    # 6. 着陆奖励：当两个支撑点都接触时给予正向奖励
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    contact_bonus = 0.5 if both_contact else 0.0

    # 7. 速度-距离耦合奖励：当接近目标时更强调减速
    # 使用 sigmoid 形式的权重，距离越近权重越大
    proximity_weight = 1.0 / (1.0 + 2.718281828 ** (distance - 2.0))
    coupled_penalty = -0.1 * proximity_weight * speed

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty +
                    ang_vel_penalty + fuel_penalty + contact_bonus +
                    coupled_penalty)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'fuel_penalty': fuel_penalty,
        'contact_bonus': contact_bonus,
        'coupled_penalty': coupled_penalty,
    }

    return float(total_reward), components