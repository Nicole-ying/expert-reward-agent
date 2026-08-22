# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 权重参数 ==========================
    w_progress = 5.0             # 距离缩减奖励，加大引导力
    w_time = -0.005              # 极小时间惩罚
    w_action_orient = -0.1       # 姿态引擎惩罚
    w_action_main = -0.2         # 主引擎惩罚

    # 接近目标区稳定性约束权重，仅当 dist < 2.0 时生效
    w_vx_near = 0.01
    w_vy_near = 0.01
    w_angle_near = 0.05
    w_ang_vel_near = 0.01

    w_landing = 200.0            # 成功着陆奖励
    w_crash = -100.0             # 非目标区接触惩罚

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

    # ========================== 组件 C：接近目标区稳定性约束 ==========================
    # 仅当处于目标附近（dist_after < 2.0）时，才对速度、角度施加轻微惩罚，
    # 引导减速摆正；远距离不做限制，允许自由机动。
    near_region = max(0.0, 1.0 - dist_after / 2.0)  # dist<2时线性增大到1
    penalty_vx_near = -w_vx_near * near_region * abs(vx_next)
    penalty_vy_near = -w_vy_near * near_region * abs(vy_next)
    penalty_angle_near = -w_angle_near * near_region * abs(angle_next)
    penalty_ang_vel_near = -w_ang_vel_near * near_region * abs(ang_vel_next)
    stability_penalty = penalty_vx_near + penalty_vy_near + penalty_angle_near + penalty_ang_vel_near

    # ========================== 组件 D：引擎使用惩罚 ==========================
    if action == 0:
        action_penalty = 0.0
    elif action in (1, 3):          # 姿态引擎
        action_penalty = w_action_orient
    elif action == 2:               # 主引擎
        action_penalty = w_action_main
    else:
        action_penalty = 0.0

    # ========================== 组件 E：成功着陆奖励 ==========================
    # 判断两脚是否充分接触（采用逻辑与保证稳定着陆）
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x_next) < 0.2 and abs(y_next) < 0.2)
    low_speed   = (abs(vx_next) < 0.5 and abs(vy_next) < 0.5)
    upright = abs(angle_next) < 0.3
    if full_contact and near_target and low_speed and upright:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # ========================== 组件 F：非目标区接触惩罚 ==========================
    # 任何单脚或双脚接触，但不在目标区域即视为危险接触
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    if any_contact and not near_target:
        crash_penalty = w_crash
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
