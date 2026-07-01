# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 使用欧几里得距离的负变化量作为势能塑形
    current_pos = obs[0:2]  # [x_position, y_position]
    next_pos = next_obs[0:2]  # [x_position, y_position]
    
    # 计算当前步和下一步到目标(0,0)的距离
    current_dist = (current_pos[0]**2 + current_pos[1]**2) ** 0.5
    next_dist = (next_pos[0]**2 + next_pos[1]**2) ** 0.5
    
    # progress_delta: 距离减少为正奖励，增加为负奖励
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 稳定/安全约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：平方形式，对高速更敏感
    speed = (vel_x**2 + vel_y**2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed**2
    
    # 姿态角惩罚：角度偏离0度
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * body_angle**2
    
    # 角速度惩罚
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel**2
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双支撑接触时给予小奖励
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(body_angle) < stable_angle_threshold
    is_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    landing_bonus_weight = 1.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        landing_bonus = landing_bonus_weight
    else:
        landing_bonus = 0.0
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus
    
    # 组件字典
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

1. **progress_reward**（主学习信号）：基于距离变化量的密集过程奖励。每一步，如果飞行器更接近目标（原点），获得正奖励；远离则获得负奖励。这是核心引导信号，让智能体学会向目标移动。

2. **stability_penalty**（稳定/安全约束）：包含速度惩罚、姿态角惩罚和角速度惩罚。鼓励飞行器以低速、小姿态角、小角速度接近目标，为后续稳定着陆做准备。权重较小，避免过度约束导致不敢移动。

3. **landing_bonus**（任务完成 proxy）：当飞行器同时满足接近目标、低速、小姿态角、双支撑接触四个条件时，给予一个小奖励。这是一个软性的着陆成功代理信号，不依赖显式 success flag。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，没有明确的成功/失败标志可用。使用终端奖励需要区分终止原因，但当前观测无法可靠区分成功着陆、坠毁或出界。强行使用会导致奖励信号不准确，误导学习。

## 留到后续迭代的组件

- **energy_penalty**：引擎使用惩罚。当前 v1 优先让智能体学会到达目标，过早加入能耗惩罚可能导致智能体不敢使用引擎。后续当智能体能稳定到达目标后，再加入小权重能耗惩罚优化燃料效率。
- **time_penalty**：时间步惩罚。当前 v1 通过 progress_reward 自然鼓励快速到达，不需要额外时间惩罚。后续如果发现智能体在目标附近徘徊，可以加入小权重时间惩罚。
- **gated_reward**：安全门控。当前 v1 使用轻量稳定性惩罚，如果发现智能体在高速接近目标时坠毁，后续可以加入安全门控，在高速时屏蔽进度奖励。

## 训练后应该观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近来回震荡，无法稳定着陆。表现为 progress_reward 在0附近波动，landing_bonus 很少触发。此时需要调整 stability_penalty 权重或加入 time_penalty。

2. **high_reward_without_success**：智能体获得高奖励但从未触发 landing_bonus。可能原因是智能体学会了在目标附近快速移动但不着陆。需要收紧 landing_bonus 条件或增加着陆奖励权重。

3. **fast_crash_near_goal**：智能体高速冲向目标然后坠毁。表现为 progress_reward 很高但突然终止。需要增大 stability_penalty 中的速度惩罚权重，或加入安全门控。

4. **agent_afraid_to_move**：智能体原地不动，获得稳定的负奖励但不敢尝试。表现为 progress_reward 接近0，stability_penalty 很小。需要降低稳定性惩罚权重或增大 progress_reward 权重。
