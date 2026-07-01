# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
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
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (origin in relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: encourage low velocity, upright angle, low angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)
    angular_vel_penalty = -0.3 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: small bonus when near target, stable, and both supports contact
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：密集过程引导，每一步奖励智能体更接近目标（原点）。
   - 数学形式：`d(obs, goal) - d(next_obs, goal)`，其中距离为欧几里得距离。
   - 这是核心学习信号，直接驱动智能体向目标移动。

2. **stability_penalty**（稳定约束，权重-0.5/-0.3/-0.2）
   - 角色：轻量约束，鼓励智能体保持低速度、直立姿态、低角速度。
   - 包含三个子项：姿态角惩罚、角速度惩罚、速度惩罚。
   - 权重较小，避免过度约束导致智能体不敢移动。

3. **soft_landing_bonus**（任务完成proxy，权重2.0）
   - 角色：当智能体同时满足"接近目标"、"低速"、"稳定姿态"、"双支撑接触"时给予小奖励。
   - 这是一个软代理信号，不是真正的成功标志，但能引导智能体完成着陆动作。
   - 条件严格（四个条件同时满足），避免reward hacking。

4. **energy_penalty**（动作代价，权重-0.05）
   - 角色：极小权重惩罚使用引擎，鼓励燃料效率。
   - 权重非常小，不会阻碍智能体学习移动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`。
- info字典为空，无法获取显式成功/失败标志。
- 使用这些终端奖励会诱导LLM发明不存在的info字段，违反设计原则。

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：当wrapper明确暴露success/failure标志后再加入。
- **time_penalty**：如果智能体接近目标但拖太久，后续可小权重加入。
- **gated_reward**：如果安全约束被进度奖励抵消，后续可加入安全门控。
- **potential_based_shaping**：如果需要更标准的塑形，后续可替换progress_delta_reward。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。如果观察到，需要收紧soft_landing_bonus条件或增加稳定惩罚。
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆。如果观察到，需要增加着陆条件或调整权重。
3. **fast_crash_near_goal**：智能体高速冲向目标但坠毁。如果观察到，需要增加速度惩罚或调整progress_delta_reward的权重。
4. **agent_afraid_to_move**：智能体因惩罚过大而不敢移动。如果观察到，需要降低stability_penalty和energy_penalty的权重。
