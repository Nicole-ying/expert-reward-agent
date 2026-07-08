def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: 将 landing_bonus 从二值条件改为连续乘积形式，提供着陆梯度。

    Components:
    1. progress_reward:   -distance_to_landing_pad (密集引导)
    2. orientation_penalty: 小惩罚非零姿态角和角速度
    3. landing_bonus:       连续乘积 proxy —— 距离近 × 速度低 × 腿触地
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy for successful landing ----
    # 三个连续因子相乘，每个提供梯度，共同约束"接近 + 低速 + 触地"
    dist_thresh = 0.5      # 距离阈值（放宽，提前给梯度）
    speed_thresh = 1.0     # 速度阈值（放宽，提前给梯度）
    bonus_scale = 10.0     # 最大可能奖励（完美着陆时）

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)
    contact_factor = min(1.0, n_left_contact + n_right_contact)  # 0, 0→0; 1, 0→1; 1, 1→1

    landing_bonus = bonus_scale * dist_factor * speed_factor * contact_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components