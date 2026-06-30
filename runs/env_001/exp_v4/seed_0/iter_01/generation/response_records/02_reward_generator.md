# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward ==========
    # 目标位置为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 当前距离平方
    current_dist_sq = obs[0]**2 + obs[1]**2
    # 下一步距离平方
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    
    # 距离减少为正奖励，距离增加为负奖励
    progress_delta = current_dist_sq - next_dist_sq
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_delta * progress_scale
    
    # ========== 2. 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为这是动作执行后的状态
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚（鼓励减速）
    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚（鼓励保持水平）
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚（鼓励稳定）
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 3. 任务完成 proxy：soft_landing_proxy ==========
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target_threshold = 0.3  # 距离阈值
    low_speed_threshold = 0.2    # 速度阈值
    stable_angle_threshold = 0.1  # 角度阈值
    
    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5
    
    soft_landing_bonus_weight = 1.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    else:
        soft_landing_proxy = 0.0
    
    # ========== 4. 动作代价（小权重） ==========
    # 轻微惩罚使用引擎，鼓励燃料效率
    # action 0 是无引擎，其他动作使用引擎
    action_penalty_weight = 0.01
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + action_penalty
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号）：基于距离平方的变化量，奖励每一步更接近目标位置。使用平方距离而非欧氏距离，计算更简单且梯度更平滑。

2. **stability_penalty**（稳定/安全约束）：包含三个子项：
   - speed_penalty：惩罚高速，鼓励减速接近
   - angle_penalty：惩罚大姿态角，鼓励保持水平
   - angular_vel_penalty：惩罚角速度，鼓励稳定

3. **soft_landing_proxy**（任务完成 proxy）：当同时满足接近目标、低速、稳定姿态和双支撑接触时给予小奖励，作为成功着陆的软信号。

4. **action_penalty**（效率/动作代价）：小权重惩罚使用引擎，鼓励燃料效率。

## 每个组件的角色

- **progress_delta_reward**：核心学习信号，引导智能体向目标移动
- **stability_penalty**：防止高速撞击、姿态失稳，确保安全接近
- **soft_landing_proxy**：提供成功着陆的额外激励，帮助智能体学习完整任务
- **action_penalty**：轻微约束引擎使用，避免无意义消耗

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 字典为空，无法获取显式的成功/失败信号。因此不在 v1 中使用终端奖励，避免伪造信号。

## 留到后续迭代的组件

- **energy_penalty**：当前已通过 action_penalty 轻微约束，后续可加大权重或改为基于燃料消耗的惩罚
- **time_penalty**：如果智能体在目标附近徘徊不完成，后续可加入时间惩罚
- **gated_reward**：如果出现安全被进度奖励抵消的情况，后续可加入安全门控
- **terminal_success_reward**：当 wrapper 明确暴露 success 信号后再加入

## 训练后应该观察的 failure mode

1. **goal_near_oscillation**：在目标附近震荡不完成——观察 progress_delta_reward 是否在目标附近频繁正负交替
2. **high_reward_without_success**：获得高奖励但未成功着陆——检查 soft_landing_proxy 是否被 hack
3. **fast_crash_near_goal**：高速撞击目标——观察 stability_penalty 是否足够抑制高速
4. **agent_afraid_to_move**：智能体不敢移动——检查 action_penalty 是否过重
5. **contact_reward_hacking**：智能体通过非正常方式触发接触——观察 soft_landing_proxy 条件是否太宽松
