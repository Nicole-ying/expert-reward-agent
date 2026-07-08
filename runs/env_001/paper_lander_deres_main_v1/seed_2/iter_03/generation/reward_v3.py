def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: 移除 landing_bonus 中的 contact_factor，让接近+低速的梯度在触地前就生效。

    诊断：v2 的 landing_bonus 有 contact_factor（二值），导致 nonzero_rate 仅 4.38%。
    95%+ 的 step 没有正向信号，agent 缺乏"减速接近"的引导，crash 率 70%。

    修改：landing_bonus = bonus_scale * dist_factor * speed_factor（去掉 contact_factor）。
    靠近目标垫且速度低即有奖励，不等腿触地。

    Components:
    1. progress_reward:   -distance_to_landing_pad（密集引导，不变）
    2. orientation_penalty: 小惩罚非零姿态角和角速度（不变）
    3. landing_bonus:       连续乘积 —— 距离近 × 速度低（不再要求触地）
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

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy — proximity × low speed (no contact gate) ----
    # 去掉了 contact_factor，让 agent 在接近+减速时就收到正向梯度
    dist_thresh = 0.5
    speed_thresh = 1.0
    bonus_scale = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)

    landing_bonus = bonus_scale * dist_factor * speed_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components