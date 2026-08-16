# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v5 — 着陆引导连续化。
    保持目标接近势能差、距离门控速度惩罚，
    将 landing_contact 从稀疏事件改为位置/速度/接触的连续乘积奖励。
    """
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== A. 目标接近 shaping（势能差） ==========
    current_potential = -(abs(x_pos) + abs(y_pos))
    next_potential = -(abs(next_x_pos) + abs(next_y_pos))
    goal_proximity = 10.0 * (next_potential - current_potential)

    # ========== B. 速度惩罚（距离门控版） ==========
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    avg_speed = (current_speed + next_speed) / 2.0
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    gating = 1.0 / (1.0 + 5.0 * dist_to_target)
    speed_penalty = -0.2 * avg_speed * gating

    # ========== C. 姿态约束 ==========
    angle_penalty = -0.5 * abs(next_body_angle)
    ang_vel_penalty = -0.2 * abs(next_ang_vel)
    attitude_constraint = angle_penalty + ang_vel_penalty

    # ========== D. 安全着陆接触奖励（连续版） ==========
    # 到中心的欧氏距离
    dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # 位置接近度：高斯核，标准差0.15
    nearness = 2.718281828 ** (-dist ** 2 / (2 * 0.15 ** 2))
    # 速度低度：速度模（L1）的指数衰减，时间常数0.3
    speed_norm = abs(next_x_vel) + abs(next_y_vel)
    slowness = 2.718281828 ** (-speed_norm / 0.3)
    # 接触分数：双腿接触为佳
    contact_score = (next_left_contact + next_right_contact) / 2.0  # ∈ [0,1]
    # 连续着陆奖励，最大约 3.0
    landing_contact = 3.0 * nearness * slowness * contact_score

    # ========== E. 燃料效率 ==========
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        speed_penalty +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    reward_components = {
        'goal_proximity': goal_proximity,
        'speed_penalty': speed_penalty,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), reward_components
```
