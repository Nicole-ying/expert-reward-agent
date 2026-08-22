# Response Record

# 设计理由
本轮修改 **Level 2 结构变换**，将永不死触发（active_rate=0）的 `boundary_penalty` 组件替换为一个新的 `safety_penalty` 组件，填补“碰撞前兆安全约束”的信号缺口。  
当前 agent 在 68 步内快速坠落并 crash，主要获得 progress 奖励（距离减少），却没有任何信号迫使它在接近地面时减速或保持姿态。  
新组件利用观测 `y_pos`、`y_vel`、`angle`（均已声明可用）构建连续 hinge 惩罚：当高度低于 0.3 且下降速度超过 0.2 时，按超出速度与角度偏离比例惩罚，且随高度降低平滑增强。  
公式：  
`danger_speed = max(0, -y_vel - 0.2)`  
`proximity = max(0, 1 - y_pos / 0.3)`  
`attitude = 1 + 2 * abs(angle)`  
`safety_penalty = -0.2 * danger_speed * proximity * attitude`  
系数 0.2 确保在典型危险情况（下降速度 -0.3，高度 0.1）时单步惩罚约 -0.02，不超过主信号 per‑step（约 0.02）的 0.3 倍。在成功着陆时（y_vel≈0，y_pos≈0，双腿已触地）惩罚为零，不会干扰最终接触奖励。  
保留其他组件（progress、landing_soft_reward、angle_penalty、action_cost），不做额外修改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Safety penalty (replaces boundary_penalty)
    # Penalise dangerous descent: too fast downward speed when close to ground,
    # amplified by body tilt.
    height_limit = 0.3
    v_limit = 0.2          # safe downward speed threshold (negative means down, so -y_vel positive)
    proximity = max(0.0, 1.0 - y_pos / height_limit)  # [0,1] when y_pos < 0.3
    danger_speed = max(0.0, -y_vel - v_limit)         # >0 when downward speed exceeds limit
    attitude = 1.0 + 2.0 * abs(angle)                 # tilt penalty multiplier
    safety_penalty = -0.2 * danger_speed * proximity * attitude

    # 5. Light angle/angular-velocity penalty
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + safety_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "safety_penalty": safety_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 边界惩罚从未触发，缺少 crash 前兆安全信号；高度与垂速观测未充分用于生存约束。
- **behavior**: agent 以极短 episode（len=68）快速下降获取 progress 奖励后 crash，未学到减速。
- **signal**: 缺少接近地面时对高速下降和姿态偏移的惩罚。
- **level**: Level 2
- **hypothesis**: 添加连续 safety_penalty 迫使 agent 在低空减速并保持竖直，延长生存时间，进而有机会触发 landing_soft_reward 和学习安全着陆。
- **risk**: 惩罚过强可能导致 agent 悬停不敢下降，若 len 提升但 score 仍低需调低系数；另外 y_vel 符号假设（向下为负）若错误可能使惩罚方向颠倒，需在下一轮根据实际情况微调。
