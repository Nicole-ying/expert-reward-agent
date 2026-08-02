def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance

    # 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚强度：越近越强调减速
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))

    # 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)

    # 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    action_penalty = -0.02 if action != 0 else 0.0

    # 着陆奖励：当接近目标、速度低、姿态正、双脚接触时给予额外奖励
    landing_bonus = 0.0
    if distance < 0.5 and speed < 0.5 and abs(angle) < 0.2 and both_contact:
        landing_bonus = 2.0

    total_reward = distance_reward + speed_penalty + angle_penalty + ang_vel_penalty + contact_reward + action_penalty + landing_bonus

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components