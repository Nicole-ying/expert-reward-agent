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
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Reward for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 5.0
    progress_reward = progress_delta * progress_scale
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_angular_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus) ---
    # Small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # --- Component 4: Action Penalty (small efficiency cost) ---
    # Penalize using any engine (actions 1, 2, 3)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
    # --- Components dict ---
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

1. **progress_reward** (主学习信号)：基于每一步到目标距离的变化量，奖励智能体向目标靠近。这是核心引导信号，权重设为5.0以提供足够强的学习梯度。

2. **stability_penalty** (稳定约束)：惩罚高速、大姿态角和快速旋转，鼓励智能体以稳定方式接近目标。权重较轻（0.2-0.5），避免过度限制探索。

3. **soft_landing_bonus** (任务完成proxy)：当智能体同时满足接近目标、低速、姿态稳定且双支撑接触时给予小奖励。这是对成功着陆的软性近似，但不直接作为成功信号。

4. **action_penalty** (效率代价)：对使用任何引擎的动作施加微小惩罚（-0.05），鼓励智能体减少不必要的引擎使用，但权重极低以避免影响主要学习。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info字典为空。因此无法可靠地判断成功或失败终止，使用终端奖励会引入错误信号或诱导LLM虚构info字段。

## 留到后续迭代的组件

- **energy_penalty**：当前动作惩罚极轻，后续可增加引擎使用成本。
- **time_penalty**：如果智能体在目标附近徘徊不完成，可加入时间惩罚。
- **gated_reward**：如果出现高速撞击或危险姿态，可加入安全门控。
- **terminal_success_reward**：当wrapper明确暴露成功标志后再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近来回移动，无法稳定着陆。此时应增加stability_penalty权重或收紧soft_landing条件。
2. **high_reward_without_success**：智能体获得高奖励但未真正着陆。应检查soft_landing_bonus条件是否太宽松。
3. **fast_crash_near_goal**：智能体高速冲向目标并坠毁。应增加速度惩罚或加入安全门控。
4. **agent_afraid_to_move**：智能体因惩罚过重而不敢移动。应降低stability_penalty或action_penalty。