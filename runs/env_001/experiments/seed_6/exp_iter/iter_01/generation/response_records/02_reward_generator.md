# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
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
    
    # 1. 主学习信号：progress_delta_reward
    # 计算到目标(0,0)的距离变化
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标
    
    # 2. 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. 动作代价（小权重）
    # 惩罚使用引擎（action != 0）
    action_penalty = -0.1 if action != 0 else 0.0
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
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
   - 角色：引导智能体每一步都更接近目标位置(0,0)
   - 基于当前距离与下一时刻距离的差值，提供密集的进度信号
   - 这是核心学习驱动信号

2. **stability_penalty**（稳定约束，权重0.2-0.5）
   - 角色：惩罚高速、大姿态角和大角速度，促进稳定飞行
   - 包含三个子项：姿态角惩罚(0.5)、角速度惩罚(0.2)、速度惩罚(0.3)
   - 轻量约束，避免智能体以不稳定方式接近目标

3. **soft_landing_proxy**（任务完成近似信号，固定奖励2.0）
   - 角色：当智能体同时满足接近目标、低速、姿态稳定、双接触时给予小奖励
   - 这是对成功着陆的软性近似，不依赖显式success flag
   - 条件严格（四个条件同时满足），避免contact reward hacking

4. **action_penalty**（动作代价，权重0.1）
   - 角色：轻微惩罚使用引擎，鼓励燃料效率
   - 权重很小，避免智能体不敢移动

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info为空字典，没有可用的success/failure标志
- 使用这些信号会违反设计原则，且可能诱导LLM虚构info字段

## 留到后续迭代的组件

- **energy_penalty**：当前使用小权重action_penalty作为替代，后续可增大权重或改为更精细的燃料模型
- **time_penalty**：如果智能体在目标附近徘徊不完成，可加入时间惩罚
- **gated_reward**：如果出现不安全行为（如高速撞击），可加入安全门控
- **terminal_success_reward**：当wrapper明确暴露success标志后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近来回振荡，无法稳定着陆
   - 观察：progress_reward接近0但soft_landing_bonus从未触发
   - 对策：增大stability_penalty权重，或收紧soft_landing_proxy条件

2. **high_reward_without_success**：智能体获得高奖励但从未完成着陆
   - 观察：progress_reward持续为正但soft_landing_bonus从未触发
   - 对策：增大soft_landing_bonus权重，或加入时间惩罚

3. **fast_crash_near_goal**：智能体快速接近目标但以高速撞击
   - 观察：progress_reward很高但stability_penalty也很高
   - 对策：增大速度惩罚权重，或加入安全门控

4. **agent_afraid_to_move**：智能体因惩罚过大而不敢移动
   - 观察：智能体停留在初始位置附近
   - 对策：降低stability_penalty和action_penalty权重
