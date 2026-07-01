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
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Distance to goal (origin) at current step
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to goal at next step
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to goal
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus for good landing conditions) ---
    # Conditions: near target, low speed, stable angle, both supports contacting
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.3
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

1. **progress_reward** (主学习信号)：基于距离变化量的密集过程奖励。每一步，如果智能体更接近目标（原点），则获得正奖励；如果远离目标，则获得负奖励。权重设为10.0，使其成为主导信号。

2. **stability_penalty** (稳定约束)：轻量惩罚项，用于抑制高速、大姿态角和快速旋转。包含三个子项：速度惩罚(0.3×speed)、姿态角惩罚(0.5×|angle|)、角速度惩罚(0.2×|angular_velocity|)。权重较小，避免过度约束导致智能体不敢移动。

3. **soft_landing_bonus** (任务完成代理)：当智能体同时满足四个条件（靠近目标<0.5、低速<0.5、姿态稳定<0.3弧度、双侧支撑接触）时，给予2.0的小额奖励。这是一个软性代理信号，不是真正的成功标志。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 为空字典。没有可用的显式成功/失败标志，因此不能使用终端奖励。使用它们会诱导 LLM 虚构 info 字段，违反设计原则。

## 留到后续迭代的组件

- **energy_penalty**：动作代价惩罚，当前未加入以避免智能体不敢使用引擎。后续当智能体能稳定到达目标后，再加入小权重惩罚引擎使用。
- **time_penalty**：时间效率惩罚，当前未加入以避免智能体冒险快速失败。后续如果智能体在目标附近徘徊太久再加入。
- **gated_reward**：安全门控，当前未使用。后续如果出现高速撞击或飞出边界等不安全行为，可考虑加入。
- **terminal_success_reward / terminal_failure_penalty**：等待 wrapper 明确暴露 success/failure 标志后再加入。

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡，无法稳定着陆。表现为 progress_reward 在0附近波动，soft_landing_bonus 偶尔触发但无法持续。
2. **high_reward_without_success**：智能体获得高 progress_reward 但从未触发 soft_landing_bonus，说明它只是不断接近-远离目标循环。
3. **fast_crash_near_goal**：智能体高速冲向目标但无法减速，导致坠毁。表现为 progress_reward 很高但 stability_penalty 也很高。
4. **agent_afraid_to_move**：如果 stability_penalty 权重过大，智能体可能选择不动，表现为所有奖励接近0。
