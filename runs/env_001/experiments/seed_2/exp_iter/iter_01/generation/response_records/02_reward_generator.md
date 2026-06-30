# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_ang_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # 4. 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎（action != 0）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.1 * engine_use
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于距离变化量的密集引导奖励，鼓励智能体每一步都更接近目标位置。这是核心学习信号，权重设为10.0。

2. **stability_penalty**（稳定约束）：惩罚高速、大姿态角和大角速度，引导智能体以稳定姿态接近目标。权重适中，避免过度约束导致不敢移动。

3. **landing_bonus**（任务完成proxy）：当智能体同时满足接近目标、低速、姿态稳定且双支撑接触时给予小奖励。这是对成功着陆的软性近似，权重较小（2.0），避免hack。

4. **energy_penalty**（动作代价）：小权重惩罚使用引擎，鼓励燃料效率。权重很小（-0.1），避免智能体因害怕消耗燃料而不敢移动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，且info始终为空字典。无法可靠判断成功或失败终止，因此不使用终端奖励，避免误导智能体。

## 留到后续迭代的组件

- **time_penalty**：当前未加入，因为可能鼓励冒险行为。如果训练后发现智能体在目标附近徘徊不完成，可以小权重加入。
- **gated_reward**：复杂门控机制，v1保持简单。
- **potential_based_shaping**：比progress_delta更抽象，后续如果需要更标准的塑形再考虑。
- **terminal_success_reward / terminal_failure_penalty**：等待wrapper明确暴露success/failure信号后再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。如果观察到，需要加强landing_bonus条件或增加时间惩罚。
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆。可能原因是progress_reward主导，需要检查是否landing_bonus条件太宽松。
3. **fast_crash_near_goal**：高速冲向目标导致坠毁。需要增大stability_penalty权重或增加速度上限惩罚。
4. **agent_afraid_to_move**：智能体因害怕惩罚而不敢移动。可能原因是energy_penalty或stability_penalty权重过大，需要调低。
