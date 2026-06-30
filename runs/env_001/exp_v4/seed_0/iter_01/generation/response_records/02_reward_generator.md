# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测值
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 组件1: 主学习信号 - progress_delta_reward
    # 奖励每一步更接近目标（目标在原点(0,0)）
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 组件2: 稳定约束 - stability_penalty
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.3 * abs(next_angular_vel)
    speed_penalty = 0.2 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 软着陆代理 - soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 组件4: 动作代价 - energy_penalty (小权重)
    # 轻微惩罚使用引擎，鼓励节能
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use
    
    # 计算总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）：基于每一步到目标距离的变化量，奖励智能体向目标靠近。这是核心引导信号，提供密集的、与任务目标直接相关的学习信号。

2. **stability_penalty**（稳定约束，权重0.2-0.5）：惩罚高速、大姿态角和大角速度，鼓励智能体以稳定姿态接近目标。这对着陆任务至关重要，防止智能体以危险方式接近目标。

3. **soft_landing_proxy**（任务完成代理，固定奖励2.0）：当智能体同时满足接近目标（距离<0.3）、低速（速度<0.5）、姿态稳定（角度<0.2）和双支撑接触时，给予小奖励。这是对成功着陆的软性近似，但不伪造success flag。

4. **energy_penalty**（动作代价，权重0.05）：轻微惩罚使用引擎，鼓励节能。权重很小，避免智能体不敢使用引擎。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info始终为空字典。因此无法可靠判断成功或失败终止，使用终端奖励会引入虚假信号或导致学习不稳定。

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：当wrapper明确暴露成功/失败标志后再加入。
- **time_penalty**：如果智能体接近目标但拖太久，后续可加入小权重时间惩罚。
- **gated_reward**：如果出现安全风险（如高速撞击），可加入安全门控。
- **dynamic curriculum**：如果训练初期无法学习，可基于training_progress调整组件权重。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。如果出现，需要增加稳定约束权重或收紧soft_landing_proxy条件。
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆。可能soft_landing_proxy条件太宽，需要收紧或移除。
3. **fast_crash_near_goal**：智能体高速冲向目标但坠毁。需要增加速度惩罚或加入安全门控。
4. **agent_afraid_to_move**：智能体不敢使用引擎。如果出现，降低energy_penalty权重或移除。
