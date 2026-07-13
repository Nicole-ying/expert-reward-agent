```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一状态量
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 到目标点的欧氏距离（目标位于原点）
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5

    # 超参数（首次尝试的基础权重，可根据后续训练反馈调整）
    w_progress = 1.0
    w_horizontal = 0.5
    w_vel = 0.5
    w_att = 0.3
    w_contact = 2.0

    # 安全约束的阈值
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2

    # 接触奖励中的理想着陆状态阈值
    ideal_angle_tol = 0.1
    ideal_vy_tol = 0.2

    # 1. 向目标移动的稠密进展奖励（距离缩短）
    progress = w_progress * (dist_old - dist_new)

    # 2. 水平对准惩罚（全局生效，惩罚水平偏离中心）
    horizontal_penalty = -w_horizontal * (x_new ** 2)

    # 3. 安全速度约束（无门控，全局生效）
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx ** 2 + excess_vy ** 2)

    # 4. 安全姿态约束（无门控，全局生效）
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle ** 2 + excess_angvel ** 2)

    # 5. 软着陆接触奖励（需要双脚同时接触 + 姿态竖直 + 低垂直速度）
    both_contact = min(left_contact, right_contact)  # 两腿均接触时才为1
    angle_quality = max(0.0, 1.0 - abs(angle_new) / ideal_angle_tol)
    vy_quality = max(0.0, 1.0 - abs(vy_new) / ideal_vy_tol)
    contact_bonus = w_contact * both_contact * angle_quality * vy_quality

    # 总奖励
    total_reward = progress + horizontal_penalty + vel_penalty + att_penalty + contact_bonus

    # 组件字典（所有出现在总公式中的奖励项）
    components = {
        'progress': progress,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components
```