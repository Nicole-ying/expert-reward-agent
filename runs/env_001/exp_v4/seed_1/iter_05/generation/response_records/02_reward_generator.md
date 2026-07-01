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
    # Distance to goal (target is at origin, obs are relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 5.0
    progress_reward = progress_scale * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and angular velocity
    # Use next_obs to penalize the resulting state after action
    vel_magnitude = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_weight = 0.5
    vel_penalty_weight = 0.3
    angular_vel_penalty_weight = 0.2
    
    # Angle penalty: penalize deviation from upright (angle=0)
    angle_penalty = -angle_penalty_weight * abs(next_body_angle)
    
    # Velocity penalty: penalize high speed (encourage slow approach)
    vel_penalty = -vel_penalty_weight * vel_magnitude
    
    # Angular velocity penalty: penalize spinning
    angular_vel_penalty = -angular_vel_penalty_weight * abs(next_angular_vel)
    
    stability_penalty = angle_penalty + vel_penalty + angular_vel_penalty
    
    # 3. Soft landing proxy: small bonus when near target, slow, stable, and both contacts
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    near_target = next_dist < near_target_threshold
    low_speed = vel_magnitude < low_speed_threshold
    stable_angle = abs(next_body_angle) < stable_angle_threshold
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small action penalty to encourage fuel efficiency (very small weight)
    # action 0 = no engine, action 1/2/3 = engine use
    action_penalty_weight = 0.05
    action_penalty = 0.0
    if action != 0:  # Any engine use
        action_penalty = -action_penalty_weight
    
    # Combine all components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward** (主学习信号)
   - 基于距离变化量的密集过程奖励，鼓励智能体每一步都更接近目标
   - 使用 `current_dist - next_dist` 计算，正值表示接近目标
   - 权重 5.0，提供清晰的学习梯度

2. **stability_penalty** (稳定约束)
   - 包含三个子项：姿态角惩罚、速度惩罚、角速度惩罚
   - 鼓励智能体保持直立姿态、低速接近、避免旋转
   - 权重适中（0.5/0.3/0.2），避免过度约束导致不敢移动

3. **soft_landing_bonus** (任务完成 proxy)
   - 当智能体同时满足：靠近目标、低速、姿态稳定、双侧支撑接触时给予小奖励
   - 权重 2.0，作为成功着陆的软信号
   - 条件组合严格，避免 reward hacking

4. **action_penalty** (效率约束)
   - 极小权重（0.05）的动作惩罚，鼓励减少不必要的引擎使用
   - 仅惩罚非零动作（引擎点火），不惩罚无操作

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`，没有明确的成功标志
- `explicit_failure_flag_available=false`，没有明确的失败标志
- `info` 固定为空字典，无法获取终止原因
- 使用这些项会诱导 LLM 发明不存在的信号，违反设计原则

## 留到后续迭代的组件

- **time_penalty**：当前任务没有时间限制，过早加入可能导致冒险行为
- **energy_penalty**：当前动作惩罚权重已很小，后续可调整为基于燃料消耗的惩罚
- **gated_reward**：复杂门控机制，v1 阶段不需要
- **terminal_success_reward**：当 wrapper 明确暴露 success 信号后再加入
- **terminal_failure_penalty**：当能区分失败终止原因后再加入

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡，无法稳定着陆
   - 观察：progress_reward 接近零但 soft_landing_bonus 很少触发
   - 对策：增加 stability_penalty 权重或调整 soft_landing 条件

2. **high_reward_without_success**：智能体获得高奖励但未成功着陆
   - 观察：progress_reward 高但 soft_landing_bonus 不触发
   - 对策：收紧 soft_landing 条件或增加 terminal 信号

3. **fast_crash_near_goal**：智能体快速接近目标但坠毁
   - 观察：progress_reward 高但稳定性惩罚不足
   - 对策：增加速度/角度惩罚权重

4. **agent_afraid_to_move**：智能体因惩罚过重而不敢移动
   - 观察：智能体停留在初始位置，progress_reward 接近零
   - 对策：降低 stability_penalty 权重或增加 progress_reward 权重
