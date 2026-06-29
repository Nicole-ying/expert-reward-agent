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
    
    # Goal is at (0, 0) in relative coordinates
    goal_x = 0.0
    goal_y = 0.0
    
    # ---- Component 1: Progress Delta Reward (main learning signal) ----
    # Reward for moving closer to the goal
    current_dist = ((x_pos - goal_x) ** 2 + (y_pos - goal_y) ** 2) ** 0.5
    next_dist = ((next_x_pos - goal_x) ** 2 + (next_y_pos - goal_y) ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # ---- Component 2: Stability Penalty (light constraint) ----
    # Penalize high velocity, large body angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_scale = 0.1
    speed_penalty_scale = 0.05
    angular_vel_penalty_scale = 0.05
    
    angle_penalty = -angle_penalty_scale * abs(next_body_angle)
    speed_penalty = -speed_penalty_scale * speed
    angular_vel_penalty = -angular_vel_penalty_scale * abs(next_angular_vel)
    
    stability_penalty = angle_penalty + speed_penalty + angular_vel_penalty
    
    # ---- Component 3: Soft Landing Proxy (small bonus for task completion proxy) ----
    # Bonus when near target, low speed, stable angle, and both supports contact
    near_target_threshold = 0.3
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2
    
    near_target = next_dist < near_target_threshold
    low_speed = speed < low_speed_threshold
    stable_angle = abs(next_body_angle) < stable_angle_threshold
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 1.0
    
    # ---- Combine components ----
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # ---- Build components dict ----
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

1. **progress_delta_reward** (主学习信号，权重2.0)
   - 角色：提供密集的、基于距离变化的过程引导，鼓励智能体每一步都更接近目标位置。
   - 数学形式：`(当前距离 - 下一时刻距离) * 2.0`
   - 这是核心学习信号，因为任务目标是到达目标位置，而progress_delta直接衡量接近目标的进度。

2. **stability_penalty** (稳定/安全约束，小权重)
   - 角色：轻量约束，鼓励智能体在接近目标时保持低速、小姿态角和低角速度。
   - 包含三个子项：角度惩罚(-0.1*|angle|)、速度惩罚(-0.05*speed)、角速度惩罚(-0.05*|angular_velocity|)
   - 权重较小，避免过度约束导致智能体不敢移动。

3. **soft_landing_proxy** (任务完成近似信号，小权重)
   - 角色：当智能体同时满足"接近目标、低速、稳定姿态、双支撑接触"时给予小奖励(+1.0)。
   - 这是对成功着陆的软代理，不是真正的success flag，但能引导智能体完成最终着陆动作。
   - 条件组合严格，避免contact reward hacking。

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确声明`explicit_success_flag_available=false`和`explicit_failure_flag_available=false`，且info字典为空。
- 无法区分成功着陆(body_not_awake_or_settled)和失败(crash/飞出边界)的终止原因。
- 使用terminal奖励会诱导LLM发明不存在的info字段，违反设计原则。

## 留到后续迭代的组件

- **energy_penalty**：v1未加入，因为太早加入可能导致智能体不敢使用引擎，无法学习基本移动策略。后续当智能体能稳定接近目标后再加入。
- **time_penalty**：v1未加入，因为可能导致智能体冒险快速失败。后续若观察到智能体在目标附近徘徊太久再加入。
- **gated_reward**：v1未使用复杂门控，因为门控过严可能导致学不到。后续若安全被进度奖励抵消再加入。
- **terminal_success_reward / terminal_failure_penalty**：当wrapper明确暴露success/failure标志后再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近来回震荡，无法稳定着陆。表现为progress_reward在0附近波动，但soft_landing_bonus很少触发。
2. **high_reward_without_success**：智能体获得高progress_reward但从未触发soft_landing_bonus，说明它学会了接近目标但不会完成着陆。
3. **fast_crash_near_goal**：智能体快速接近目标但以高速撞击，表现为高progress_reward但高stability_penalty。
4. **agent_afraid_to_move**：stability_penalty权重过大导致智能体不敢移动，表现为progress_reward长期为负或接近0。
