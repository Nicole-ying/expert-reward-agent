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
    # Reward for moving closer to the target (target is at (0,0) in relative coordinates)
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and high angular velocity
    # Use next_obs to penalize the state after taking action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy: small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_distance < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small distance anchor to prevent getting stuck far away
    distance_anchor = -0.1 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward** (主学习信号，权重10.0)
   - 角色：提供密集的过程引导，奖励每一步更接近目标位置
   - 数学形式：`current_distance - next_distance`，正值表示向目标靠近
   - 这是核心学习信号，引导智能体学会向目标移动

2. **stability_penalty** (稳定约束，负向惩罚)
   - 角色：轻量约束，惩罚高速、大姿态角和快速旋转
   - 包含三个子项：速度惩罚(0.3)、角度惩罚(0.5)、角速度惩罚(0.2)
   - 权重较小，避免过度约束导致智能体不敢移动

3. **soft_landing_bonus** (任务完成proxy，权重2.0)
   - 角色：当智能体同时满足"近目标、低速度、小角度、双接触"时给予小奖励
   - 这是软性的成功近似信号，不是伪造的success flag
   - 条件严格，避免contact reward hacking

4. **distance_anchor** (辅助锚点，权重-0.1)
   - 角色：小权重的距离惩罚，防止智能体在远离目标时获得过高奖励
   - 权重很小，不会干扰主学习信号

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- 没有可用的成功/失败标志，使用终端奖励会需要伪造info字段或依赖不可靠的终止条件判断
- 这些组件留到后续迭代，当wrapper明确暴露success/failure信号后再加入

## 留到后续迭代的组件

- **energy_penalty**：当前不加入，避免智能体不敢使用引擎。后续当智能体能稳定到达目标后，再加入小权重燃料惩罚优化效率
- **time_penalty**：当前不加入，避免鼓励冒险行为。后续如果发现智能体在目标附近徘徊太久再加入
- **gated_reward**：当前不加入，复杂门控可能阻碍早期学习
- **terminal_success_reward / terminal_failure_penalty**：等待显式信号可用后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆
   - 观察：progress_delta_reward 接近0但 soft_landing_bonus 不触发
   - 对策：收紧 soft_landing_bonus 条件或增加稳定性惩罚

2. **high_reward_without_success**：智能体获得高奖励但未成功着陆
   - 观察：progress_delta_reward 很高但 soft_landing_bonus 不触发
   - 对策：增加 soft_landing_bonus 权重或收紧条件

3. **fast_crash_near_goal**：智能体快速冲向目标但坠毁
   - 观察：progress_delta_reward 很高但稳定性惩罚不足
   - 对策：增加速度/角度惩罚权重

4. **agent_afraid_to_move**：智能体不敢移动（如果后续加入energy_penalty）
   - 观察：总奖励很低，智能体停留在初始位置
   - 对策：降低energy_penalty权重或移除
