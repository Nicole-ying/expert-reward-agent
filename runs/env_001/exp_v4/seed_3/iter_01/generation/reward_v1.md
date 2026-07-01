# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty
    # 惩罚速度、姿态角和角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_ang_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 1.0
    
    # 4. 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎（action != 0）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -engine_use * 0.1
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于距离变化量的密集过程引导。每一步更接近目标(0,0)时给予正奖励，远离时给予负奖励。这是核心学习信号，引导智能体向目标移动。

2. **stability_penalty**（稳定约束）：惩罚高速、大姿态角和快速旋转。确保智能体在接近目标时保持稳定姿态和低速，为着陆做准备。

3. **landing_bonus**（任务完成proxy）：当智能体同时满足接近目标、低速、姿态稳定且双支撑接触时给予小奖励。这是软性的着陆成功近似信号，鼓励智能体完成最终着陆动作。

4. **energy_penalty**（动作代价）：小权重惩罚使用引擎，鼓励燃料效率。权重很小（0.1），避免智能体不敢移动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info始终为空字典，无法获取成功或失败标志
- 使用这些终端奖励会诱导伪造信号或错误假设

## 留到后续迭代的组件

- **terminal_success_reward**：当wrapper明确暴露success标志后再加入
- **terminal_failure_penalty**：当能区分失败终止后再加入
- **time_penalty**：如果智能体拖太久不完成，后续可小权重加入
- **gated_reward**：如果安全约束被进度奖励抵消，后续可加入门控机制
- **potential_based_shaping**：如果需要更标准的塑形，后续可替换progress_delta

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆。如果观察到，需要加强landing_bonus条件或增加稳定约束权重。

2. **high_reward_without_success**：智能体获得高奖励但从未成功着陆。需要收紧landing_bonus条件或增加终端奖励。

3. **fast_crash_near_goal**：智能体高速冲向目标导致坠毁。需要增加速度惩罚权重或加入安全门控。

4. **agent_afraid_to_move**：智能体因惩罚过重而不敢移动。需要降低stability_penalty或energy_penalty权重。