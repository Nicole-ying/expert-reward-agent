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

    # 2. 软着陆引导：连续正奖励，随距离、速度、姿态和双腿接触程度衰减
    w_soft = 1.0
    prox_thresh = 0.5
    vel_thresh = 0.5
    ang_thresh = 0.5

    proximity_gate = max(0.0, 1.0 - next_dist / prox_thresh)
    vel_mag = abs(next_vx) + abs(next_vy)
    vel_gate = max(0.0, 1.0 - vel_mag / vel_thresh)
    ang_gate = max(0.0, 1.0 - abs(body_angle) / ang_thresh)
    contact_ratio = (left_contact + right_contact) / 2.0

    soft_landing_bonus = w_soft * proximity_gate * vel_gate * ang_gate * contact_ratio

    # 3. 碰撞预防（保留 iter 2 结构，几乎无触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚（使用 iter 2 的轻量值）
    living_penalty = -0.002

    total_reward = (reward_prox + soft_landing_bonus +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_bonus": soft_landing_bonus,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components