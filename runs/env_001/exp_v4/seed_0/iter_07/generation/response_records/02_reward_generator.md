# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    # 当前位置
    x_pos = obs[0]
    y_pos = obs[1]
    # 下一步位置
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    # 当前速度
    x_vel = obs[2]
    y_vel = obs[3]
    # 姿态
    body_angle = obs[4]
    angular_vel = obs[5]
    # 接触标志
    left_contact = obs[6]
    right_contact = obs[7]
    
    # 下一步速度（用于稳定性惩罚）
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    
    # 计算到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标
    progress_delta = current_dist - next_dist
    progress_scale = 10.0
    progress_reward = progress_delta * progress_scale
    
    # ========== 组件2: 稳定约束 - stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle)
    angular_vel_penalty = abs(next_angular_vel)
    
    stability_scale = 0.5
    speed_scale = 0.3
    angle_scale = 0.2
    angular_vel_scale = 0.1
    
    stability_penalty = -(
        speed_scale * speed +
        angle_scale * angle_penalty +
        angular_vel_scale * angular_vel_penalty
    ) * stability_scale
    
    # ========== 组件3: 任务完成proxy - soft_landing_proxy ==========
    # 当接近目标、低速、小角度且双接触时给予小奖励
    near_target_threshold = 0.3
    low_speed_threshold = 0.2
    stable_angle_threshold = 0.2
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(next_body_angle) < stable_angle_threshold
    is_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 2.0
    
    # ========== 组件4: 动作代价 - energy_penalty (小权重) ==========
    # 使用引擎（动作1,2,3）时给予小惩罚
    engine_actions = [1, 2, 3]
    energy_penalty_scale = 0.1
    energy_penalty = 0.0
    if action in engine_actions:
        energy_penalty = -energy_penalty_scale
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于每一步到目标距离的减少量（delta），奖励智能体持续接近目标。这是核心引导信号，权重最大（scale=10.0）。

2. **stability_penalty**（稳定约束）：惩罚高速、大姿态角和角速度，鼓励智能体以稳定方式接近目标。权重适中（scale=0.5），避免过强导致不敢移动。

3. **soft_landing_proxy**（任务完成proxy）：当智能体同时满足接近目标、低速、小角度和双接触四个条件时，给予小奖励（2.0）。这是对成功着陆的近似信号，但不直接当作成功标志。

4. **energy_penalty**（动作代价）：使用任何引擎（动作1/2/3）时给予小惩罚（-0.1），鼓励燃料效率。权重很小，避免智能体不敢使用引擎。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info字典为空，无法获取成功/失败标志
- 使用这些组件会诱导伪造info字段或误判终止原因

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：当wrapper明确暴露success/failure标志后再加入
- **time_penalty**：如果智能体接近目标但拖太久，后续可加入小权重时间惩罚
- **gated_reward**：如果出现安全被进度奖励抵消的情况，再加入安全门控
- **potential_based_shaping**：如果需要更标准的塑形，可替换或补充progress_delta

## 训练后应观察的failure mode

1. **goal_near_oscillation**：目标附近震荡，不完成着陆。如果出现，需要收紧soft_landing_proxy条件或增加稳定性惩罚权重。
2. **high_reward_without_success**：奖励很高但从未成功着陆。需要检查soft_landing_proxy是否被hack，或增加着陆条件。
3. **fast_crash_near_goal**：高速冲向目标附近后坠毁。需要增加速度惩罚或加入安全门控。
4. **agent_afraid_to_move**：智能体不敢使用引擎。需要降低energy_penalty权重或增加progress_reward权重。
