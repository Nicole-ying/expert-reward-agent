# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables from observations
    # Current position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    
    # Next position (relative to target)
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Current velocities
    x_vel = obs[2]
    y_vel = obs[3]
    
    # Next velocities
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    
    # Orientation and angular velocity
    body_angle = obs[4]
    next_body_angle = next_obs[4]
    angular_vel = obs[5]
    next_angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = obs[6]
    right_contact = obs[7]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (Main Learning Signal) ==========
    # Reward the agent for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    
    # Scale progress reward to be meaningful
    progress_scale = 5.0
    progress_reward = progress_delta * progress_scale
    
    # ========== Component 2: Stability Penalty (Light Constraint) ==========
    # Penalize high speed, large angle, and high angular velocity
    # Use next_obs to penalize the resulting state after action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_weight = 0.1
    speed_penalty_weight = 0.05
    angular_vel_penalty_weight = 0.02
    
    # Penalize deviation from upright (angle=0) and high angular velocity
    angle_penalty = -angle_penalty_weight * (next_body_angle ** 2)
    speed_penalty = -speed_penalty_weight * speed
    angular_vel_penalty = -angular_vel_penalty_weight * (next_angular_vel ** 2)
    
    stability_penalty = angle_penalty + speed_penalty + angular_vel_penalty
    
    # ========== Component 3: Soft Landing Proxy (Small Bonus) ==========
    # Small bonus when near target, low speed, stable angle, and both supports contact
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(next_body_angle) < stable_angle_threshold
    is_both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus_weight = 2.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        landing_bonus = landing_bonus_weight
    else:
        landing_bonus = 0.0
    
    # ========== Component 4: Small Action Penalty (Efficiency) ==========
    # Very light penalty for using engines to encourage fuel efficiency
    # action 0 = no engine, actions 1-3 = engine use
    action_penalty_weight = 0.02
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight
    
    # ========== Combine Components ==========
    total_reward = progress_reward + stability_penalty + landing_bonus + action_penalty
    
    # ========== Build Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "angle_penalty": angle_penalty,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重5.0）：基于当前位置到目标距离的变化，奖励每一步更接近目标。这是核心引导信号，让智能体学会向目标平台移动。

2. **stability_penalty**（稳定约束，小权重）：包含三个子项：
   - 角度惩罚（权重0.1）：惩罚姿态角偏离0（直立状态）
   - 速度惩罚（权重0.05）：惩罚高速运动
   - 角速度惩罚（权重0.02）：惩罚旋转运动
   这些约束帮助智能体在接近目标时保持稳定，为着陆做准备。

3. **landing_bonus**（任务完成proxy，权重2.0）：当同时满足四个条件时给予小奖励：
   - 距离目标<0.3
   - 速度<0.5
   - 姿态角<0.2弧度
   - 左右支撑均接触
   这是一个软性着陆信号，不是硬性成功标志。

4. **action_penalty**（效率约束，权重0.02）：轻微惩罚使用引擎的动作，鼓励燃料效率。权重很小，避免智能体不敢移动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info始终为空字典。无法区分成功（稳定着陆）和失败（坠毁/出界）的终止原因，因此不能使用终点奖励或惩罚。

## 留到后续迭代的组件

- **energy_penalty**（大权重动作惩罚）：当前仅使用很小的动作惩罚，后续若智能体学会接近目标但燃料消耗过大，可增加权重。
- **time_penalty**：当前未使用，后续若智能体拖太久不完成，可加入。
- **gated_reward**：当前未使用，后续若安全约束被进度奖励抵消，可加入安全门控。
- **terminal_success_reward**：当wrapper能明确暴露成功信号后再加入。
- **potential_based_shaping**：当前progress_delta已足够，后续若需要更标准的塑形可替换。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡，无法稳定着陆。表现为progress_reward高但landing_bonus始终为0。解决方案：增加stability_penalty权重或收紧landing_bonus条件。

2. **high_reward_without_success**：智能体获得高progress_reward但从未触发landing_bonus。解决方案：检查landing_bonus条件是否太严格，或增加时间惩罚。

3. **fast_crash_near_goal**：智能体快速接近目标但高速撞击。表现为progress_reward高但stability_penalty也高。解决方案：增加速度惩罚权重。

4. **agent_afraid_to_move**：action_penalty导致智能体不敢使用引擎。解决方案：降低action_penalty权重或移除。