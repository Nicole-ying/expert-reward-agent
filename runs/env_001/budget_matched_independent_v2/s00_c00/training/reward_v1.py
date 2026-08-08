def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]       # x_position: 水平坐标（相对于目标）
    y = next_obs[1]       # y_position: 垂直坐标（相对于着陆台高度）
    vx = next_obs[2]      # x_velocity: 水平速度
    vy = next_obs[3]      # y_velocity: 垂直速度
    angle = next_obs[4]   # body_angle: 机体角度
    ang_vel = next_obs[5] # angular_velocity: 角速度
    left_contact = next_obs[6]  # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7] # right_support_contact: 右支撑接触标志

    # 从 obs 提取上一时刻的接触状态（用于检测着陆事件）
    prev_left = obs[6]
    prev_right = obs[7]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚，每步鼓励靠近

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0），着陆时更重要
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定，减少旋转
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 着陆奖励：检测首次接触目标表面
    # 着陆定义为：两个支撑点都接触，且之前至少有一个未接触
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    prev_both = (prev_left > 0.5) and (prev_right > 0.5)
    landing_bonus = 2.0 if (both_contact and not prev_both) else 0.0

    # 6. 稳定着陆维持奖励：保持双接触状态
    stable_landing_bonus = 0.5 if both_contact else 0.0

    # 7. 燃料效率惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    fuel_penalty = 0.0
    if action == 2:  # 主引擎推力最大
        fuel_penalty = -0.3
    elif action in [1, 3]:  # 姿态引擎
        fuel_penalty = -0.1
    # action 0 无惩罚

    # 8. 进度自适应：随着训练推进，逐渐强调着陆稳定性
    progress_factor = min(training_progress, 1.0)
    adaptive_landing_weight = 1.0 + progress_factor * 2.0

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        landing_bonus * adaptive_landing_weight +
        stable_landing_bonus * adaptive_landing_weight * 0.5 +
        fuel_penalty
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus * adaptive_landing_weight,
        "stable_landing_bonus": stable_landing_bonus * adaptive_landing_weight * 0.5,
        "fuel_penalty": fuel_penalty,
    }

    return float(total_reward), components