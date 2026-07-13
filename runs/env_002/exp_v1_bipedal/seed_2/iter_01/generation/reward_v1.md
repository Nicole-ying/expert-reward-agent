# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度奖励
    forward_velocity = next_obs[2]  # 水平速度
    forward_reward = 1.0 * forward_velocity
    
    # 存活奖励：鼓励保持站立
    # 通过检查主体角度和角速度判断是否还站立
    hull_angle = abs(next_obs[0])  # 主体角度绝对值
    hull_angular_vel = abs(next_obs[1])  # 主体角速度绝对值
    # 如果角度太大或角速度太大，认为可能摔倒，不给存活奖励
    is_alive = (hull_angle < 1.0) and (hull_angular_vel < 2.0)
    alive_bonus = 0.5 if is_alive else 0.0
    
    # 稳定性惩罚：轻量约束，防止过度倾斜和高速旋转
    stability_penalty = -0.1 * (hull_angle + 0.5 * hull_angular_vel)
    
    # 总奖励
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    components = {
        "forward_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **forward_reward**（主学习信号）：基于前向速度的稠密奖励，直接激励 agent 向前行走。这是 locomotion 任务最核心的学习信号，每步都有梯度。

2. **alive_bonus**（存活激励）：当 agent 保持直立姿态时给予小额奖励，鼓励 agent 不摔倒。通过主体角度和角速度判断是否存活，避免 agent 通过快速摔倒来逃避任务。

3. **stability_penalty**（稳定约束）：轻量惩罚项，抑制过大的倾斜角度和角速度，防止 agent 在快速前进时失去平衡。权重较小（0.1），不会压制探索。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`
- info 字典始终为空，无法区分终止原因是成功到达终点还是摔倒
- 使用终端奖励会导致 agent 无法获得正确的学习信号，可能误导策略

## 留到后续迭代的组件

- **energy_penalty**：v1 阶段 agent 应先学会稳定行走，过早加入能耗惩罚可能导致 agent 不敢移动
- **action_smoothness_penalty**：如果后续观察到动作抖动，可以加入
- **time_penalty**：效率优化应在 agent 能稳定完成任务后再考虑
- **gated_reward**：复杂门控机制在 v1 阶段可能过严，限制探索

## 训练后应观察的 failure mode

1. **fast_then_fail**：agent 可能学会快速前进但频繁摔倒。如果观察到这种情况，应增加稳定性惩罚权重或加入姿态约束。
2. **hover_or_stand_still**：agent 可能原地不动获取存活奖励而不前进。如果观察到这种情况，应降低存活奖励权重或增加前向速度奖励权重。
3. **over_conservative_policy**：稳定性惩罚过强可能导致 agent 不敢移动。如果观察到这种情况，应降低稳定性惩罚权重。