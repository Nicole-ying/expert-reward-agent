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
    # Reward for moving closer to the target (origin)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and angular velocity
    # Use next_obs to penalize the state after action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)
    angular_vel_penalty = -0.2 * abs(next_angular_vel)
    speed_penalty = -0.3 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(next_body_angle) < 0.1
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # Total reward
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # Components dict
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
   - 角色：引导智能体每一步都更接近目标位置（原点）
   - 数学形式：`current_dist - next_dist`，正值表示向目标靠近
   - 这是核心学习信号，提供密集的进度反馈

2. **stability_penalty**（稳定/安全约束，权重-0.5/-0.2/-0.3）
   - 角色：惩罚高速、大姿态角和大角速度，促进稳定飞行
   - 包含三个子项：角度惩罚、角速度惩罚、速度惩罚
   - 权重较小，避免过度约束导致智能体不敢移动

3. **soft_landing_proxy**（任务完成近似信号，权重2.0）
   - 角色：在接近目标、低速、稳定姿态且双支撑接触时给予小奖励
   - 条件组合：`near_target and low_speed and stable_angle and both_contact`
   - 权重较小，仅作为辅助信号，不主导学习

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info 字典为空，无法区分成功终止和失败终止
- 使用终端奖励会引入不确定性，可能导致错误的学习信号

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，因为过早加入可能导致智能体不敢使用引擎，无法学习基本导航
- **time_penalty**：当前未加入，因为可能鼓励冒险行为，导致快速失败
- **gated_reward**：当前未加入，因为门控条件难以定义，且可能阻碍探索
- **terminal_success_reward**：等待 wrapper 明确暴露 success 信号后再加入

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆
   - 观察：progress_reward 高但 soft_landing_bonus 不触发
   - 对策：增大 soft_landing_bonus 权重或收紧条件

2. **high_reward_without_success**：奖励高但未成功着陆
   - 观察：总奖励高但 episode 未以 success 终止
   - 对策：检查 progress_reward 是否主导，考虑加入时间惩罚

3. **fast_crash_near_goal**：接近目标时速度过快导致坠毁
   - 观察：progress_reward 高但突然终止（crash）
   - 对策：增大速度惩罚权重或加入速度门控

4. **agent_afraid_to_move**：稳定性惩罚过强导致智能体不敢移动
   - 观察：智能体停留在起点附近，progress_reward 低
   - 对策：降低稳定性惩罚权重