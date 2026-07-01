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
    # Distance to goal (origin) at current and next step
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(body_angle)
    angular_vel_penalty = -0.3 * abs(angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # --- Component 3: Soft Landing Proxy (task completion hint) ---
    # Small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.5
    low_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5 < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于每一步到目标（原点）距离的减少量，权重10.0。这是核心引导信号，鼓励智能体持续向目标靠近。

2. **stability_penalty**（稳定/安全约束）：包含姿态角惩罚（-0.5×|angle|）、角速度惩罚（-0.3×|angular_velocity|）和速度惩罚（-0.2×|velocity|）。权重较小，防止智能体在接近目标时姿态失控或速度过快。

3. **soft_landing_bonus**（任务完成proxy）：当智能体同时满足接近目标（距离<0.5）、低速（速度<0.3）、稳定姿态（角度<0.2）和双侧接触时，给予2.0的小奖励。这是一个软信号，鼓励最终稳定着陆。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境明确标注 `explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info字典为空。无法可靠区分成功/失败终止，因此不使用终端奖励，避免伪造信号。

## 留到后续迭代的组件

- **energy_penalty**：当前动作代价未加入，因为v1优先让智能体学会接近目标，过早惩罚引擎使用可能导致不敢动。
- **time_penalty**：未加入，避免鼓励冒险行为。
- **gated_reward**：复杂门控逻辑留到后续。
- **terminal_success_reward / terminal_failure_penalty**：等待wrapper暴露明确成功/失败标志后再加入。

## 训练后应观察的failure mode

1. **目标附近震荡**：progress_reward在目标附近变小，可能导致智能体在目标周围徘徊而不完成着陆。应观察是否出现high_reward_without_success。
2. **高速撞击**：如果stability_penalty权重不足，智能体可能高速冲向目标。应观察是否出现fast_crash_near_goal。
3. **接触奖励hacking**：soft_landing_bonus条件较严格，但仍需观察是否出现非正常着陆但获得bonus的情况。