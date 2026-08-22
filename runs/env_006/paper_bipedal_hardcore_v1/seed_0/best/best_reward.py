def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]
    next_leg1_contact = next_obs[12]
    next_leg2_contact = next_obs[13]

    # Component A: 前进速度奖励乘以健康姿态门控
    forward_speed = max(0.0, horizontal_speed)  # 只奖励向前移动
    angle_factor = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    vert_factor = 1.0 / (1.0 + 2.0 * abs(vertical_speed))
    health_gate = angle_factor * vert_factor
    forward_reward = 2.0 * forward_speed * health_gate

    # Component B: 动作代价（抑制过度用力）
    action_cost = 0.01 * sum(a ** 2 for a in action)

    # Component C: 接触过渡奖励（鼓励正常的交替步态，惩罚双脚离地）
    contact_reward = 0.0
    if leg1_contact != next_leg1_contact or leg2_contact != next_leg2_contact:
        # 发生了接触状态变化
        if next_leg1_contact == 0 and next_leg2_contact == 0:
            contact_reward = -0.2   # 双脚离地（跳跃/摔倒前兆）
        elif next_leg1_contact == 1 and next_leg2_contact == 1:
            contact_reward = 0.0    # 双脚同时触地，不奖不罚
        else:
            contact_reward = 0.1    # 单脚支撑的正常步态切换

    total_reward = forward_reward - action_cost + contact_reward
    components = {
        "forward_reward_gated": forward_reward,
        "action_cost": -action_cost,
        "contact_transition_reward": contact_reward
    }
    return float(total_reward), components