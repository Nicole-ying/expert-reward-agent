def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚（从 -0.002 加强至 -0.005）
    living_penalty = -0.005

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components