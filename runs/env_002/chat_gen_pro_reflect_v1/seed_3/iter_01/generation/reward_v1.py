def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为前进激励
    #    采用凸化形式 (signal**2) 打破低速平台，鼓励更快行走
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress_reward = 1.0 * (horizontal_vel ** 2)

    # ============================================================
    # 2. balance_maintenance: 稳定/安全约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 quadratic_penalty 形式，对倾斜角度和角速度施加轻量惩罚
    #    权重较小，避免压制探索
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 3. gait_cycle_encouragement: 步态周期鼓励
    #    使用 leg1_contact (obs[8]) 和 leg2_contact (obs[13])
    #    鼓励双腿交替接触地面，惩罚双腿同时离地
    #    采用 binary_state_signal 组合形式
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # 鼓励至少一条腿接触地面（避免跳跃/蹦跳）
    contact_sum = leg1_contact + leg2_contact
    # 当双腿同时离地时给予惩罚
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0
    # 鼓励交替接触（当一条腿接触时给予小奖励）
    alternating_bonus = 0.1 * (leg1_contact * (1.0 - leg2_contact) + leg2_contact * (1.0 - leg1_contact))

    # ============================================================
    # 组合总奖励
    # ============================================================
    total_reward = forward_progress_reward + balance_penalty + both_off_ground_penalty + alternating_bonus

    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        "forward_progress_reward": forward_progress_reward,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "alternating_bonus": alternating_bonus
    }

    return float(total_reward), components