def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    horizontal_vel = obs[2]
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    leg1_contact = obs[8]
    leg2_contact = obs[13]

    # --- 1. 主奖励：前进速度，线性+凸化 ---
    forward = max(0.0, horizontal_vel)
    forward_reward = 5.0 * forward + 2.0 * (forward ** 2)

    # --- 2. 步态效率：与前进速度绑定的交替奖励 ---
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_efficiency = 0.5 * gait_quality * forward

    # --- 3. 姿态健康门控 + 角速度惩罚（替代原 balance_penalty）---
    # 基础门控：abs(hull_angle) <= 0.3 为 1.0，>= 0.7 为 0.0
    angle_excess = max(0.0, abs(hull_angle) - 0.3)
    base_gate = max(0.0, min(1.0, 1.0 - angle_excess / 0.4))
    # 加下限 0.3，保证即使严重倾斜仍有微弱正向梯度
    balance_gate = 0.3 + 0.7 * base_gate
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    # 总效果：当 hull 端正时相当于只受角速度惩罚；倾斜时压低了前进相关奖励
    balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty

    # --- 4. 双脚离地惩罚 ---
    contact_sum = leg1_contact + leg2_contact
    if contact_sum < 0.5:
        both_off_penalty = -0.1 * (1.0 - contact_sum)
    else:
        both_off_penalty = 0.0

    # --- 5. 动作能耗惩罚 ---
    action_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = forward_reward + gait_efficiency + balance_modulation + both_off_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "gait_efficiency": gait_efficiency,
        "balance_modulation": balance_modulation,
        "both_off_penalty": both_off_penalty,
        "action_penalty": action_penalty,
        "gait_quality": gait_quality,
        "balance_gate": balance_gate
    }
    return float(total_reward), components