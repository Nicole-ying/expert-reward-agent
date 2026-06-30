# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取位置信息（相对于目标着陆台）
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 提取速度信息
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 提取姿态信息
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 提取接触信息
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算当前位置到目标的距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 稳定约束 - stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(body_angle)
    angular_vel_penalty = 0.2 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # ========== 组件3: 任务完成proxy - soft_landing_bonus ==========
    # 当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    near_target = next_distance < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：提供密集的导航引导，奖励每一步更接近目标位置
   - 数学形式：`current_distance - next_distance`
   - 这是核心学习信号，让智能体学会向目标移动

2. **stability_penalty**（稳定约束，权重组合0.5/0.2/0.3）
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定飞行
   - 包含三个子项：姿态角惩罚、角速度惩罚、速度惩罚
   - 权重较轻，避免过度约束导致智能体不敢移动

3. **soft_landing_bonus**（任务完成proxy，固定值2.0）
   - 角色：当飞行器同时满足接近目标、低速、姿态稳定、双支撑接触时给予小奖励
   - 权重较小，不会主导学习，但提供着陆完成的软信号
   - 条件组合严格，避免contact reward hacking

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确说明`explicit_success_flag_available=false`和`explicit_failure_flag_available=false`
- info始终为空字典，无法区分成功和失败终止
- 使用这些项会需要伪造success/failure信号，违反设计原则

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，避免智能体不敢使用引擎。后续当智能体能稳定到达目标后再加入以优化燃料效率
- **time_penalty**：当前未加入，避免鼓励冒险行为。后续如果发现智能体在目标附近徘徊太久再加入
- **gated_reward**：当前未使用，因为安全条件不明确。后续如果发现进度奖励导致不安全行为再加入
- **terminal_success_reward**：等待wrapper明确暴露success信号后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。表现为progress_delta_reward接近0但soft_landing_bonus不触发
2. **high_reward_without_success**：智能体获得高奖励但从未成功着陆。需要检查soft_landing_bonus条件是否太严格
3. **fast_crash_near_goal**：智能体高速冲向目标但坠毁。需要增加stability_penalty权重
4. **agent_afraid_to_move**：智能体几乎不动。需要降低stability_penalty权重或增加progress_delta_reward权重