def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]          # 相对目标平台水平坐标
    y = next_obs[1]          # 相对平台高度
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机身角度
    omega = next_obs[5]      # 角速度
    left_contact = next_obs[6]   # 左支撑接触标志
    right_contact = next_obs[7]  # 右支撑接触标志

    w_x = 0.05
    w_y = 0.05
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05
    w_landing = 1.0

    target_proximity = -w_x * abs(x) - w_y * abs(y)
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)
    landing_reward = w_landing * (left_contact * right_contact)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty + landing_reward

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components