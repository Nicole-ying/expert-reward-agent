# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态变量
    # 位置（相对于目标）
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 速度
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 姿态
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 接触标志
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 组件1: 主学习信号 - progress_delta_reward
    # 奖励每一步更接近目标
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 组件2: 轻量稳定约束 - stability_penalty
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(body_angle)
    angular_vel_penalty = 0.2 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 软着陆代理奖励 - soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 组件4: 小权重距离锚点 - distance_anchor
    # 辅助引导，防止在远距离时progress_delta信号太弱
    distance_anchor = -0.1 * next_dist
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # 组件字典
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：核心引导信号，奖励每一步比上一步更接近目标位置
   - 数学形式：`current_dist - next_dist`，正值表示接近目标
   - 这是导航到达任务最直接的过程引导信号

2. **stability_penalty**（稳定约束，权重组合约1.0）
   - 角色：轻量惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
   - 包含三个子项：角度惩罚(0.5)、角速度惩罚(0.2)、速度惩罚(0.3)
   - 权重较小，避免过度约束导致不敢移动

3. **soft_landing_bonus**（任务完成代理，固定值2.0）
   - 角色：当满足"接近目标+低速+姿态稳定+双支撑接触"条件时给予小奖励
   - 这是对成功着陆的软性近似，不是伪造success flag
   - 条件严格，避免contact reward hacking

4. **distance_anchor**（辅助锚点，权重0.1）
   - 角色：小权重距离惩罚，防止在远距离时progress_delta信号太弱
   - 权重很小，不会与主信号冲突

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info固定为空字典，没有可用的success/failure标志
- 使用这些信号会违反接口契约，且无法正确实现

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，因为v1太早加入可能导致agent不敢使用引擎
- **time_penalty**：当前未加入，避免鼓励冒险行为
- **gated_reward**：当前未加入，门控过严可能阻碍学习
- **terminal_success_reward**：等待wrapper明确暴露success信号后再加入
- **terminal_failure_penalty**：等待能区分失败终止后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：目标附近震荡，不完成着陆
   - 表现：progress_delta_reward高但soft_landing_bonus从未触发
   - 对策：后续可增加时间惩罚或收紧soft_landing条件

2. **high_reward_without_success**：奖励高但从未成功着陆
   - 表现：总奖励持续上升但episode不结束
   - 对策：检查soft_landing条件是否太宽松，或需要加入terminal信号

3. **fast_crash_near_goal**：接近目标时高速撞击
   - 表现：progress_delta_reward高但stability_penalty不足以抑制高速
   - 对策：增大速度惩罚权重或加入速度上限约束

4. **agent_afraid_to_move**：因稳定惩罚过大而不敢移动
   - 表现：agent停留在起点附近，progress_delta_reward接近0
   - 对策：降低stability_penalty权重