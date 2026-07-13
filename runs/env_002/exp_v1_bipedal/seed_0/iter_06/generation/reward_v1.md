# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]  # horizontal_velocity
    progress_reward = 2.0 * forward_velocity
    
    # 稳定性约束：惩罚身体倾斜和角速度
    hull_angle = next_obs[0]  # hull_angle
    hull_angular_velocity = next_obs[1]  # hull_angular_velocity
    stability_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_velocity ** 2)
    
    # 步态质量奖励：鼓励交替接触地面
    leg1_contact = next_obs[12]  # leg1_contact
    leg2_contact = next_obs[13]  # leg2_contact
    # 当两条腿交替接触时给予奖励（异或逻辑）
    gait_quality = abs(leg1_contact - leg2_contact)  # 1.0 when exactly one leg contacts
    gait_bonus = 0.3 * gait_quality
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + gait_bonus
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "gait_bonus": gait_bonus
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于前进速度的线性奖励，直接驱动 agent 向前行走。这是任务的核心目标，权重最大（2.0）。

2. **stability_penalty**（稳定/安全约束）：惩罚身体倾斜角度（hull_angle）和角速度（hull_angular_velocity），防止 agent 在快速前进时摔倒。使用平方形式使惩罚随偏离程度非线性增长。

3. **gait_bonus**（步态质量辅助信号）：鼓励两条腿交替接触地面（异或逻辑），促进产生自然的行走步态而非跳跃或拖行。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`：info 始终为空字典，无法获取成功标志
- `explicit_failure_flag_available=false`：同样无法获取失败标志
- 虽然环境有 success-like termination（到达地形终点）和 failure-like termination（摔倒），但无法在 reward 函数中区分这两种终止原因，因此不能使用终端奖励/惩罚

## 留到后续迭代的组件

- **energy_penalty**：v1 阶段 agent 应先学会稳定行走，过早加入能耗惩罚可能导致 agent 不敢移动
- **time_penalty**：同样会抑制探索，后续迭代当 agent 能稳定行走后再考虑效率优化
- **action_smoothness_penalty**：如果观察到动作抖动严重，可以在后续迭代加入
- **terminal_success_reward**：如果未来环境提供显式成功标志，可以加入

## 训练后应观察的 failure mode

1. **fast_then_fail**：agent 可能为了获得高前进速度奖励而忽略稳定性，导致快速摔倒。如果观察到这种情况，应增大 stability_penalty 权重或改用更保守的稳定性约束形式。

2. **hover_or_stand_still**：虽然本设计没有使用 alive_bonus，但 agent 可能通过原地小幅度摆动来获取 gait_bonus 而不前进。如果观察到前进速度持续接近零，应检查 gait_bonus 是否被 exploit。

3. **over_conservative_policy**：如果 stability_penalty 权重过大，agent 可能不敢移动。应观察训练初期是否出现 agent 保持直立不动的情况。

4. **gait hacking**：agent 可能通过快速交替接触地面（如抖动）来获取 gait_bonus，而不产生实际前进。如果观察到高频低幅度的接触切换，应限制 gait_bonus 的触发频率或改用更鲁棒的步态度量。