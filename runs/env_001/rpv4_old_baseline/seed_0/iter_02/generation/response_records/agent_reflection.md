# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 权重参数 ==========================
    w_progress = 2.0        # 距离缩减的正面奖励
    w_vx = 0.001            # 水平速度惩罚（小权重，避免过度压制接近行为）
    w_vy = 0.001            # 垂直速度惩罚
    w_angle = 0.005         # 姿态角惩罚
    w_ang_vel = 0.001       # 角速度惩罚
    w_action = 0.001        # 引擎使用惩罚（极小，允许必要机动）
    w_landing = 150.0       # 精确软着陆奖励（每步，条件严格）

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

    # ========================== 组件 B：稳定性/安全约束（弱惩罚） ==========================
    penalty_vx = -w_vx * (vx_next ** 2)
    penalty_vy = -w_vy * (vy_next ** 2)
    penalty_angle = -w_angle * (angle_next ** 2)
    penalty_ang_vel = -w_ang_vel * (ang_vel_next ** 2)
    stability_penalty = penalty_vx + penalty_vy + penalty_angle + penalty_ang_vel

    # ========================== 组件 C：引擎使用惩罚 ==========================
    if action != 0:
        action_penalty = -w_action
    else:
        action_penalty = 0.0

    # ========================== 组件 D：软着陆大额奖励 ==========================
    # 条件：至少一个支撑脚接触，且位置、速度均接近零，角度/角速度也可考虑但不强制
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    near_target = (abs(x_next) < 0.1 and abs(y_next) < 0.1)
    low_speed   = (abs(vx_next) < 0.2 and abs(vy_next) < 0.5)
    if any_contact and near_target and low_speed:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + stability_penalty + action_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'landing_reward': landing_reward
    }

    return float(total_reward), components
```
