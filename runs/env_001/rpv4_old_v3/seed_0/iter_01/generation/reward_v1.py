def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1: 主进展信号 + 软着陆约束 + 碰撞预防 + 时间惩罚
    """
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]

    # 距离计算（欧几里得距离）
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（基于距离改善的差分奖励）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)  # 距离减小得正奖励

    # 2. 软着陆约束：接近垫子时惩罚高速和倾斜姿态
    w_soft = 0.5
    proximity_threshold = 0.3  # 当距离 < 0.3 时激活
    gate = max(0.0, 1.0 - next_dist / proximity_threshold)  # 线性衰减门控
    # 速度用绝对值，姿态用平方（抑制大幅倾斜）
    soft_penalty_intensity = abs(next_vx) + abs(next_vy) + body_angle ** 2
    penalty_soft = -w_soft * gate * soft_penalty_intensity

    # 3. 碰撞预防：水平越界惩罚
    x_bound = 0.9  # 视野假设边界
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    # 4. 碰撞预防：高速垂直坠落惩罚（低于垫子且下落过快）
    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:  # 低于垫子且速度向下超过阈值
        # next_vy 为负值，差值平方惩罚强度随速度增大而增加
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 5. 时间惩罚：推动快速完成任务
    living_penalty = -0.002

    total_reward = (reward_prox + penalty_soft +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_penalty": penalty_soft,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components