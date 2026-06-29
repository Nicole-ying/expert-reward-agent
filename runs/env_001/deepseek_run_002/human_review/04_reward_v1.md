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
    
    # Goal is at (0, 0) in relative coordinates
    # Compute distances
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 1. Progress delta reward (main learning signal)
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Stability penalty (light constraint)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * (next_body_angle ** 2)
    angular_vel_penalty = 0.1 * (next_angular_vel ** 2)
    speed_penalty = 0.2 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy (small bonus for being near target with low speed and stable)
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small distance anchor to prevent getting stuck far away
    distance_anchor = -0.1 * current_dist
    
    # Combine components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）：每一步更接近目标时给予正奖励，远离时给予负奖励。这是核心引导信号，直接驱动智能体向目标移动。

2. **stability_penalty**（稳定约束，权重组合）：惩罚大的姿态角、角速度和速度，鼓励飞行器以稳定姿态接近目标，避免高速撞击或姿态失控。

3. **soft_landing_bonus**（任务完成proxy，权重2.0）：当飞行器同时满足接近目标、低速、姿态稳定、双支撑接触时给予小奖励。这是对成功着陆的软近似，但不伪造success flag。

4. **distance_anchor**（辅助锚点，权重-0.1）：小权重惩罚当前距离，防止智能体在远处停滞不前，同时避免与主信号重复。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- environment_card明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，info为空字典。
- 使用这些信号需要伪造success/failure flag，违反设计原则。
- 当前通过progress_delta和soft_landing_proxy提供密集引导，不需要终点信号。

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，避免智能体不敢使用引擎。后续当智能体能稳定到达目标后，再加入小权重惩罚引擎使用。
- **time_penalty**：当前未加入，避免鼓励冒险行为。后续如果智能体在目标附近徘徊太久，再加入小权重时间惩罚。
- **terminal_success_reward**：当wrapper明确暴露success flag后再加入。
- **gated_reward**：当前不需要复杂门控，后续如果安全与进度冲突再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近来回震荡，无法稳定着陆。表现为progress_reward在0附近波动，soft_landing_bonus很少触发。
2. **high_reward_without_success**：智能体获得高progress_reward但从未触发soft_landing_bonus，说明它接近目标但不停留。
3. **fast_crash_near_goal**：智能体高速冲向目标，获得高progress_reward但姿态失控。表现为stability_penalty很大负值。
4. **agent_afraid_to_move**：智能体几乎不移动，stability_penalty主导行为。表现为progress_reward很小，总奖励为负。