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
    # Reward moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high velocity, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus) ---
    # Bonus when near target, low speed, stable angle, and both supports contact
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # --- Component 4: Small Distance Anchor (auxiliary) ---
    # Small negative reward proportional to distance to keep agent aware of goal
    distance_anchor = -0.1 * next_dist
    
    # --- Combine components ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # --- Build components dict ---
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

1. **progress_delta_reward** (主学习信号，权重10.0)：奖励每一步更接近目标位置(0,0)。这是核心学习信号，引导智能体向目标移动。

2. **stability_penalty** (稳定约束，权重组合0.5+0.2+0.3)：惩罚高速度、大姿态角和快角速度。确保智能体在接近目标时保持稳定，避免高速撞击或姿态失控。

3. **soft_landing_proxy** (任务完成近似信号，权重2.0)：当智能体同时满足近目标(<0.3)、低速(<0.5)、小角度(<0.2)和双支撑接触时给予小奖励。这是对成功着陆的软代理，但不伪造success flag。

4. **distance_anchor** (辅助锚点，权重0.1)：小权重负距离奖励，作为辅助信号防止智能体完全忽略目标位置。权重很小，不会与progress_delta_reward冲突。

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`：没有显式成功标志，无法可靠判断成功终止
- `explicit_failure_flag_available=false`：没有显式失败标志，无法区分碰撞失败与成功着陆
- 使用terminal奖励会引入错误信号，误导智能体

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，避免智能体不敢使用引擎。后续当智能体能稳定接近目标后再加入燃料优化
- **time_penalty**：当前未加入，避免智能体因时间压力而冒险。后续若发现智能体在目标附近徘徊太久再加入
- **gated_reward**：当前未使用，避免门控过严导致学不到。后续若安全被进度奖励完全抵消再加入
- **terminal_success_reward / terminal_failure_penalty**：等待wrapper明确暴露success/failure信号后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不稳定着陆。表现为progress_reward高但soft_landing_bonus很少触发。对策：增大stability_penalty或收紧soft_landing条件。

2. **high_reward_without_success**：智能体获得高progress_reward但从未触发soft_landing_bonus。对策：检查是否progress_reward权重过大，或增加distance_anchor权重。

3. **fast_crash_near_goal**：智能体快速接近目标但高速撞击。表现为progress_reward高但stability_penalty负值大。对策：增大速度惩罚权重。

4. **agent_afraid_to_move**：智能体因stability_penalty而不敢移动。对策：降低stability_penalty权重或增加progress_reward权重。

5. **contact_reward_hacking**：智能体在远离目标的地方触发接触。对策：收紧soft_landing_proxy的near_target条件。
