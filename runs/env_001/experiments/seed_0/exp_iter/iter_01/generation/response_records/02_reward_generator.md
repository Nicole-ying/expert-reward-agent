# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
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
    
    # Component 1: Progress delta reward (main learning signal)
    # Reward getting closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = progress_delta * 10.0
    
    # Component 2: Stability penalty (light constraint)
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_angular_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 3: Soft landing proxy (small bonus for being near target with stability)
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.3
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # Component 4: Small action penalty (efficiency, very small weight)
    # Penalize using any engine (actions 1, 2, 3)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
    # Build components dict
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：提供密集的、连续的学习引导，奖励每一步更接近目标位置(0,0)
   - 数学形式：`current_dist - next_dist`，正值表示向目标靠近
   - 这是核心学习信号，让智能体学会向目标移动

2. **stability_penalty**（稳定约束，轻量）
   - 角色：惩罚高速、大姿态角和高角速度，鼓励稳定飞行
   - 包含三个子项：角度惩罚(0.5)、角速度惩罚(0.3)、速度惩罚(0.2)
   - 权重较小，避免过度约束导致智能体不敢移动

3. **soft_landing_proxy**（任务完成近似信号，小权重2.0）
   - 角色：当智能体同时满足接近目标、低速、稳定姿态和双接触时给予小奖励
   - 条件组合：`near_target and low_speed and stable_angle and both_contact`
   - 权重很小，不会主导学习，但提供完成任务的额外激励

4. **action_penalty**（效率约束，极小权重-0.05）
   - 角色：轻微惩罚使用引擎，鼓励节能
   - 权重极小(-0.05)，不会让智能体害怕使用引擎
   - 风险说明：可能让智能体在早期不敢移动，但权重足够小，风险可控

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`，info字典为空，无法获取成功/失败标志
- 使用这些项会诱导伪造info字段或错误判断终止条件
- 留到后续迭代，当wrapper明确暴露success/failure信号后再加入

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：需要显式成功/失败标志
- **time_penalty**：当前任务没有时间限制，后续若发现拖太久再加入
- **energy_penalty**（更大权重）：当前使用极小权重，后续可增大以优化燃料消耗
- **gated_reward**：当前不需要复杂门控，后续若安全被进度奖励抵消再加入
- **potential_based_shaping**：比progress_delta更抽象，后续若需要更标准shaping再替换

## 训练后应观察的failure mode

1. **goal_near_oscillation**：目标附近震荡，不完成着陆
   - 观察：progress_reward接近0但soft_landing_bonus不触发
   - 对策：增大soft_landing_bonus或收紧条件

2. **high_reward_without_success**：获得高奖励但未成功着陆
   - 观察：progress_reward高但soft_landing_bonus不触发
   - 对策：收紧soft_landing条件或加入terminal_success_reward

3. **fast_crash_near_goal**：接近目标时高速撞击
   - 观察：progress_reward高但stability_penalty也高
   - 对策：增大stability_penalty权重

4. **agent_afraid_to_move**：智能体不敢移动（action_penalty过强）
   - 观察：action_penalty主导，progress_reward低
   - 对策：减小action_penalty权重或移除
