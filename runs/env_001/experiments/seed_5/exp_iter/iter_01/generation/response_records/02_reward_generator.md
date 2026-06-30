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
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)
    angular_vel_penalty = -0.3 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 软着陆代理：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. 小权重距离锚点：distance_anchor
    # 辅助progress_delta，防止目标附近震荡
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：引导智能体每一步都更接近目标位置(0,0)
   - 计算方式：当前距离减去下一时刻距离，正值表示靠近目标
   - 这是核心学习信号，提供密集的进度反馈

2. **stability_penalty**（稳定约束）
   - 角色：惩罚高速、大姿态角和大角速度，促进稳定飞行
   - 包含三个子项：姿态角惩罚(-0.5)、角速度惩罚(-0.3)、速度惩罚(-0.2)
   - 权重较小，避免过度约束导致不敢移动

3. **soft_landing_bonus**（任务完成代理，权重2.0）
   - 角色：当智能体同时满足接近目标、低速、姿态稳定、双接触时给予小奖励
   - 这是软代理，不是硬性成功信号，仅作为额外引导
   - 条件组合防止contact hacking

4. **distance_anchor**（辅助锚点，权重-0.1）
   - 角色：小权重距离惩罚，防止目标附近震荡
   - 权重远小于progress_delta，不会主导学习

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确声明`explicit_success_flag_available=false`和`explicit_failure_flag_available=false`
- info始终为空字典，无法获取成功或失败信号
- 使用这些组件会诱导LLM发明不存在的info字段，违反设计原则

## 留到后续迭代的组件

- **energy_penalty**：当前动作空间是离散的4个动作，过早加入能耗惩罚可能导致智能体不敢使用引擎
- **time_penalty**：可能导致冒险行为或快速失败，应在智能体能稳定到达目标后再加入
- **gated_reward**：复杂门控机制，v1阶段应保持简单
- **terminal_success_reward**：需要wrapper明确暴露success信号后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：目标附近震荡，不完成着陆
   - 观察指标：距离小但soft_landing_bonus不触发
   - 应对：增大distance_anchor权重或收紧soft_landing条件

2. **high_reward_without_success**：获得高奖励但未真正成功
   - 观察指标：progress_reward高但soft_landing_bonus低
   - 应对：检查是否在目标附近徘徊，考虑加入时间惩罚

3. **fast_crash_near_goal**：接近目标时高速撞击
   - 观察指标：距离小但速度大，soft_landing_bonus不触发
   - 应对：增大stability_penalty中的速度惩罚权重

4. **agent_afraid_to_move**：智能体不敢移动
   - 观察指标：progress_reward持续为负或接近零
   - 应对：减小stability_penalty权重或增大progress_reward权重
