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
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. Stability penalty: penalize high velocity, large angle, and angular velocity
    # Use next_obs to penalize the state after action
    vel_magnitude = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_angular_vel) * 0.3
    vel_penalty = vel_magnitude * 0.2
    stability_penalty = angle_penalty + angular_vel_penalty + vel_penalty
    
    # 3. Soft landing proxy: small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.5
    low_speed = vel_magnitude < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 1.0
    
    # Combine components
    total_reward = progress_reward - stability_penalty + landing_bonus
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward** (主学习信号，权重 2.0)
   - 角色：提供密集的过程引导，每一步奖励智能体更接近目标位置 (0,0)
   - 使用 `current_dist - next_dist` 计算距离变化，正值表示接近目标
   - 这是核心学习信号，引导智能体学会向目标移动

2. **stability_penalty** (稳定/安全约束)
   - 角色：惩罚高速、大姿态角和角速度，鼓励稳定飞行
   - 包含三个子项：速度惩罚 (0.2)、姿态角惩罚 (0.5)、角速度惩罚 (0.3)
   - 使用 next_obs 的状态，确保惩罚的是动作执行后的结果
   - 权重适中，避免过度惩罚导致智能体不敢移动

3. **landing_bonus** (任务完成 proxy，小权重 1.0)
   - 角色：当智能体同时满足接近目标、低速、稳定姿态和双接触时给予小奖励
   - 这是 soft proxy，不是真正的 success flag
   - 条件组合确保不会把单纯的接触当作成功
   - 权重较小，不会主导学习信号

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- 没有可靠的 success/failure 信号可用
- 使用这些项会诱导 LLM 发明不存在的 info 字段，违反设计原则

## 留到后续迭代的组件

- **energy_penalty**：当前动作空间是离散的，燃料消耗关系不明确，过早加入可能导致智能体不敢使用引擎
- **time_penalty**：可能鼓励冒险行为，等智能体能稳定接近目标后再加入
- **gated_reward**：门控机制过于复杂，v1 应保持简单
- **terminal_success_reward**：需要 wrapper 明确暴露 success 信号后再加入

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡，不完成着陆
   - 观察：progress_reward 接近 0 但 landing_bonus 很少触发
   - 对策：收紧 landing_bonus 条件或增加稳定性惩罚

2. **high_reward_without_success**：获得高奖励但未真正成功着陆
   - 观察：progress_reward 很高但 landing_bonus 不触发
   - 对策：增加 landing_bonus 权重或收紧条件

3. **fast_crash_near_goal**：快速接近目标但坠毁
   - 观察：progress_reward 高但稳定性惩罚也高
   - 对策：增加稳定性惩罚权重或调整速度阈值

4. **agent_afraid_to_move**：智能体不敢移动
   - 观察：所有奖励都很低，智能体原地不动
   - 对策：降低稳定性惩罚权重或增加 progress_reward 权重
