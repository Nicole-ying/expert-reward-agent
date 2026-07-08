def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs 解包
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # x_vel = next_obs[2]   # 未直接使用，但保留以备扩展
    # y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 距离奖励：基于到目标平台中心的欧氏距离，密集引导接近目标
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -distance

    # 2. 姿态惩罚：避免倾斜过大，鼓励竖直姿态
    angle_penalty = -abs(angle)

    # 3. 角速度惩罚：抑制快速旋转，提高稳定性
    angular_vel_penalty = -abs(ang_vel)

    # 4. 能量惩罚：减少引擎使用，满足节省燃料要求
    energy_penalty = 0.0
    if action == 2:               # 主引擎
        energy_penalty = -0.1
    elif action in (1, 3):        # 转向引擎
        energy_penalty = -0.05
    # action 0 不额外惩罚

    # 5. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.01

    # 总奖励为各项直接相加
    total_reward = (
        distance_reward
        + angle_penalty
        + angular_vel_penalty
        + energy_penalty
        + time_penalty
    )

    components = {
        "distance_reward": distance_reward,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
        "time_penalty": time_penalty,
        "total_reward": total_reward,
    }

    return float(total_reward), components