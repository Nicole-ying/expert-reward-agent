```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 权重参数 ==========================
    w_progress = 3.0            # 距离缩减奖励
    w_time = -0.01              # 每步时间惩罚，鼓励快速完成任务
    # 稳定性约束（速度和姿态）
    w_vx = 0.01
    w_vy = 0.01
    w_angle = 0.1
    w_ang_vel = 0.01
    # 着陆/接触相关
    w_landing = 50.0            # 每步软着陆奖励（降低以平衡代价）
    w_crash_contact = 10.0      # 非目标区接触惩罚

    # ========================== 观察量解析 ==========================
    x_cur  = obs[0]
    y_cur  = obs[1]
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========================== 组件 A：距离缩减奖励 ==========================
    dist_before = (x_cur**2 + y_cur**2) ** 0.5
    dist_after  = (x_next**2 + y_next**2) ** 0.5
    progress_reward = w_progress * (dist_before - dist_after)

    # ========================== 组件 B：时间惩罚 ==========================
    time_penalty = w_time  # 常数，每步相同

    # ========================== 组件 C：稳定性/安全约束 ==========================
    penalty_vx = -w_vx * (vx_next ** 2)
    penalty_vy = -w_vy * (vy_next ** 2)
    penalty_angle = -w_angle * abs(angle_next)
    penalty_ang_vel = -w_ang_vel * abs(ang_vel_next)
    stability_penalty = penalty_vx + penalty_vy + penalty_angle + penalty_ang_vel

    # ========================== 组件 D：引擎使用惩罚（区分类别） ==========================
    if action == 0:
        action_penalty = 0.0
    elif action in (1, 3):          # 姿态引擎，相对较小惩罚
        action_penalty = -0.05
    elif action == 2:               # 主引擎，较大惩罚
        action_penalty = -0.1
    else:
        action_penalty = 0.0

    # ========================== 组件 E：软着陆奖励 ==========================
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    near_target = (abs(x_next) < 0.1 and abs(y_next) < 0.1)
    low_speed   = (abs(vx_next) < 0.2 and abs(vy_next) < 0.5)
    if any_contact and near_target and low_speed:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # ========================== 组件 F：非目标区接触惩罚 ==========================
    if any_contact and not near_target:
        crash_penalty = -w_crash_contact
    else:
        crash_penalty = 0.0

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + time_penalty + stability_penalty + action_penalty + landing_reward + crash_penalty

    components = {
        'progress_reward': progress_reward,
        'time_penalty': time_penalty,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'landing_reward': landing_reward,
        'crash_penalty': crash_penalty
    }

    return float(total_reward), components
```