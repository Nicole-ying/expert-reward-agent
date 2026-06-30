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
    # Reward for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * (next_body_angle ** 2)
    angular_vel_penalty = 0.1 * (next_angular_vel ** 2)
    speed_penalty = 0.2 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus) ---
    # Bonus when near target, low speed, stable angle, and both supports contact
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：引导智能体每一步都更接近目标位置(0,0)
   - 数学形式：`10.0 * (current_dist - next_dist)`，其中距离为欧几里得距离
   - 这是最直接的任务引导信号，让智能体学会向目标移动

2. **stability_penalty**（稳定约束，负值）
   - 角色：轻量惩罚高速、大姿态角和大角速度，防止震荡和失控
   - 数学形式：`-(0.5*angle² + 0.1*angular_vel² + 0.2*speed)`
   - 权重较小，避免过度约束导致智能体不敢移动

3. **soft_landing_proxy**（任务完成近似信号，小权重2.0）
   - 角色：当智能体同时满足接近目标、低速、稳定姿态和双支撑接触时给予小奖励
   - 条件组合：`near_target (dist<0.5) AND low_speed (speed<0.5) AND stable_angle (|angle|<0.2) AND both_contact`
   - 注意：这不是真正的success flag，而是软性代理信号，权重很小

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- 没有可靠的 success/failure 信号可用，使用这些项会引入错误假设
- 后续迭代中，如果 wrapper 暴露了明确的终止原因，可以再加入

## 留到后续迭代的组件

- **energy_penalty**：当前动作空间是离散的，加入能耗惩罚可能让智能体不敢使用引擎，导致无法学习基本移动
- **time_penalty**：过早加入可能导致智能体冒险快速失败，而不是学习稳定接近
- **gated_reward**：复杂门控机制在v1阶段可能过度约束学习
- **terminal_success_reward / terminal_failure_penalty**：等待明确的 success/failure 信号可用

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体接近目标但来回震荡，不完成着陆
   - 观察：progress_reward 高但 soft_landing_bonus 始终为0
   - 对策：后续增加 time_penalty 或收紧 stability_penalty

2. **high_reward_without_success**：智能体获得高奖励但从未触发 soft_landing
   - 观察：progress_reward 很高但 episode 不结束
   - 对策：收紧 soft_landing 条件或增加 terminal 信号

3. **fast_crash_near_goal**：智能体高速冲向目标然后坠毁
   - 观察：progress_reward 前期很高，但 stability_penalty 突然变大
   - 对策：增加 speed_penalty 权重或加入 velocity clipping

4. **agent_afraid_to_move**：稳定性惩罚过强导致智能体不敢移动
   - 观察：progress_reward 始终接近0
   - 对策：降低 stability_penalty 权重
