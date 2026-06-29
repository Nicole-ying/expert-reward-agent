# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 使用欧几里得距离的负变化作为进度奖励
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 权重10，鼓励每一步更接近目标
    
    # 稳定/安全约束1：速度惩罚（轻量）
    # 惩罚过大的线速度，鼓励平稳接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty = -0.5 * speed  # 小权重，避免过度保守
    
    # 稳定/安全约束2：姿态角惩罚（轻量）
    # 惩罚过大的姿态角和角速度，鼓励稳定姿态
    angle_penalty = -0.3 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = -0.2 * abs(next_obs[5])  # 角速度惩罚
    stability_penalty = angle_penalty + angular_vel_penalty
    
    # 任务完成proxy：soft_landing_proxy（小权重）
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3  # 距离目标很近
    low_speed = speed < 0.2  # 速度很低
    stable_angle = abs(next_obs[4]) < 0.1  # 姿态接近水平
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)  # 双支撑接触
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0  # 小权重奖励，避免hack
    
    # 总奖励
    total_reward = progress_reward + speed_penalty + stability_penalty + soft_landing_bonus
    
    # 组件记录
    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号，权重10）
   - 基于每一步到目标距离的负变化（delta），鼓励智能体持续接近目标。
   - 这是核心引导信号，提供密集的、平滑的学习梯度。

2. **speed_penalty**（稳定约束，权重-0.5）
   - 惩罚过大的线速度，防止智能体高速冲向目标导致不稳定或坠毁。
   - 轻量权重，避免过度抑制探索。

3. **stability_penalty**（稳定约束，权重-0.3和-0.2）
   - 惩罚过大的姿态角和角速度，鼓励保持水平稳定姿态。
   - 这对着陆任务至关重要，防止翻滚或倾斜。

4. **soft_landing_bonus**（任务完成proxy，权重2.0）
   - 当智能体同时满足：接近目标、低速、姿态稳定、双支撑接触时，给予小奖励。
   - 这是对成功着陆的软性近似，但不伪造success flag。

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，info字典为空。
- 没有可靠的信号来区分成功/失败终止，使用终端奖励会引入误导性信号。
- 当前设计通过密集的进度奖励和稳定约束来引导学习，避免依赖不可靠的终止判断。

## 留到后续迭代的组件

- **energy_penalty**：当前未加入，因为v1应优先让智能体学会接近和稳定，过早优化燃料可能导致不敢动。
- **time_penalty**：未加入，避免鼓励冒险行为。
- **gated_reward**：未加入，门控逻辑可能过于复杂，v1保持简单。
- **terminal_success_reward**：等待wrapper明确暴露success flag后再加入。
- **terminal_failure_penalty**：等待能区分失败终止后再加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不稳定着陆。如果观察到，需要增加稳定约束权重或收紧soft_landing条件。
2. **high_reward_without_success**：智能体获得高奖励但从未成功着陆。需要检查是否过度依赖进度奖励而忽略最终着陆。
3. **fast_crash_near_goal**：智能体高速冲向目标然后坠毁。需要增加速度惩罚或加入安全门控。
4. **agent_afraid_to_move**：智能体因惩罚过重而不敢移动。如果观察到，应降低稳定约束权重。
