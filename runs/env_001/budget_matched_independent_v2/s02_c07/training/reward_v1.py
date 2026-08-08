def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号 (索引基于 observation_space.fields)
    x = next_obs[0]          # x_position: 水平坐标相对于目标
    y = next_obs[1]          # y_position: 垂直坐标相对于着陆点高度
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取上一时刻信号用于速度变化计算
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]

    # 1. 距离奖励: 鼓励接近目标 (目标在原点)
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚: 鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    speed_penalty = -0.05 * speed

    # 3. 姿态奖励: 鼓励保持直立 (角度为0)
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚: 鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 着陆奖励: 当两个支撑都接触时给予正向奖励
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作效率惩罚: 鼓励少用引擎 (动作1,2,3消耗燃料)
    action_penalty = 0.0
    if action == 1 or action == 2 or action == 3:
        action_penalty = -0.02

    # 7. 速度变化惩罚: 鼓励平滑运动 (避免剧烈加速)
    acc_x = vx - prev_vx
    acc_y = vy - prev_vy
    jerk_penalty = -0.01 * ((acc_x ** 2) + (acc_y ** 2))

    # 8. 接近目标时速度衰减奖励: 鼓励在接近时减速
    # 当距离小且速度大时给予惩罚
    approach_quality = -0.1 * (distance * speed)  # 距离*速度越小越好

    # 9. 角度稳定性: 鼓励在接近目标时保持稳定角度
    angle_stability = -0.3 * (distance * (angle ** 2))  # 距离近时角度更重要

    # 10. 存活奖励: 鼓励持续探索
    survival_bonus = 0.01

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_reward +
        action_penalty +
        jerk_penalty +
        approach_quality +
        angle_stability +
        survival_bonus
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "jerk_penalty": jerk_penalty,
        "approach_quality": approach_quality,
        "angle_stability": angle_stability,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components