# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables
    # Position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Velocity
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # Orientation
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (main learning signal) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    progress_reward = 10.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(body_angle)
    angular_penalty = 0.2 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Proxy (small bonus) ==========
    # Conditions: near target, low speed, stable angle, both supports contacting
    near_target = dist_next < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # ========== Components Dictionary ==========
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：提供密集的、连续的过程引导，奖励智能体每一步更接近目标位置。
   - 数学形式：`d(obs, goal) - d(next_obs, goal)`，基于obs[0]、obs[1]计算欧氏距离。
   - 选择理由：这是导航到达任务最直接、最有效的学习信号，能引导智能体从任何位置向目标移动。

2. **stability_penalty**（轻量稳定约束，负值）
   - 角色：鼓励智能体在接近目标时保持低速、小姿态角、小角速度，为稳定着陆做准备。
   - 数学形式：对速度、姿态角、角速度分别施加小权重惩罚。
   - 选择理由：任务要求"稳定在中央目标平台"，因此需要早期引入稳定约束，但权重较小以避免过度保守。

3. **soft_landing_proxy**（任务完成近似信号，小权重2.0）
   - 角色：当智能体同时满足"接近目标、低速、稳定姿态、双支撑接触"时给予小奖励，作为成功着陆的软代理。
   - 数学形式：条件组合的二元奖励。
   - 选择理由：由于没有显式success flag，需要提供一个软信号鼓励完成着陆动作。权重很小，避免hack。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，info始终为空字典。
- 无法可靠区分终止原因是成功着陆、坠毁还是超出边界。
- 使用这些项会诱导LLM发明不存在的info字段，违反设计原则。

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，因为v1应优先让智能体学会移动和接近目标，过早加入燃料惩罚可能导致不敢使用引擎。
- **time_penalty**：未加入，避免鼓励冒险行为。
- **gated_reward**：未加入，门控机制可能阻碍早期探索。
- **terminal_success_reward / terminal_failure_penalty**：等待wrapper明确暴露success/failure信号后再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。表现为progress_reward接近0但soft_landing_bonus始终为0。
2. **high_reward_without_success**：智能体获得高progress_reward但从未触发soft_landing_bonus，说明它学会了接近但不稳定着陆。
3. **fast_crash_near_goal**：智能体高速冲向目标然后坠毁，说明stability_penalty权重不足。
4. **agent_afraid_to_move**：智能体原地不动，说明stability_penalty或progress_reward权重设置不当。
