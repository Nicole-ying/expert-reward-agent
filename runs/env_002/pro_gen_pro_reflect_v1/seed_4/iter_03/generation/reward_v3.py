def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 观测提取 --------------------
    hull_angle     = float(obs[0])
    hull_ang_vel   = float(obs[1])
    leg1_contact   = float(obs[8])
    leg2_contact   = float(obs[13])

    # -------------------- 步态循环奖励（主信号） --------------------
    # 使用闭包内状态记录上一次接触状况，确保每条腿的未接触→接触转换被奖励
    # 由于 compute_reward 在每步调用一次，我们通过函数属性存储状态
    if not hasattr(compute_reward, "prev_contacts"):
        compute_reward.prev_contacts = [False, False]  # [leg1, leg2]
    
    prev_leg1, prev_leg2 = compute_reward.prev_contacts
    curr_leg1 = leg1_contact > 0.5
    curr_leg2 = leg2_contact > 0.5
    
    # 每条腿的未接触→接触事件 (rising edge)
    event1 = 1.0 if (curr_leg1 and not prev_leg1) else 0.0
    event2 = 1.0 if (curr_leg2 and not prev_leg2) else 0.0
    
    gait_cycle_reward = 0.3 * (event1 + event2)

    # 更新状态为下一次调用准备
    compute_reward.prev_contacts = [curr_leg1, curr_leg2]

    # -------------------- 姿态安全边界 (hinge) --------------------
    tilt_penalty = -0.2 * max(0.0, abs(hull_angle) - 0.35)

    # -------------------- 动作能量惩罚 --------------------
    torque_penalty = -0.0005 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # -------------------- 角速度轻量化约束 --------------------
    angvel_penalty = -0.0002 * (hull_ang_vel ** 2)

    # -------------------- 总奖励 --------------------
    total_reward = gait_cycle_reward + tilt_penalty + torque_penalty + angvel_penalty

    components = {
        "gait_cycle_reward": gait_cycle_reward,
        "tilt_penalty": tilt_penalty,
        "torque_penalty": torque_penalty,
        "angvel_penalty": angvel_penalty
    }

    return float(total_reward), components