# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：奖励沿前进方向的速度
    w_forward = 1.0
    forward_velocity = next_obs[13]  # 前进速度（主体x方向速度）
    forward_reward = w_forward * forward_velocity

    # 约束1：惩罚横向运动，鼓励直线前进
    w_lateral = 0.5
    lateral_velocity = next_obs[14]  # 横向速度
    lateral_penalty = -w_lateral * (lateral_velocity ** 2)

    # 约束2：惩罚过高的角速度，保持姿态稳定
    w_angular = 0.5
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]
    yaw_rate = next_obs[18]
    angular_penalty = -w_angular * (roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2)

    # 约束3：惩罚剧烈的垂直运动，防止跳跃/坠落
    w_vertical = 0.1
    vertical_velocity = next_obs[15]  # 垂直速度
    vertical_penalty = -w_vertical * abs(vertical_velocity)

    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "vertical_velocity_penalty": vertical_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件与角色
- **forward_velocity_reward**（主学习信号）  
  直接奖励机器人的前进速度 `next_obs[13]`，驱动其学会向前移动。这是整个奖励的核心驱动力，每一步都提供稠密梯度。
- **lateral_velocity_penalty**（稳定约束）  
  惩罚横向速度的平方，抑制侧向漂移，引导机器人保持前进方向。
- **angular_velocity_penalty**（姿态稳定约束）  
  惩罚滚转、俯仰、偏航三个角速度的平方和，抑制剧烈的身体摇晃，降低摔倒风险。
- **vertical_velocity_penalty**（垂直运动约束）  
  惩罚垂直速度的绝对值，减少不必要的跳跃或剧烈起伏，帮助维持平稳步态。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境中 **explicit_success_flag_available = false**，**explicit_failure_flag_available = false**，info 字段为空。没有任何显式的成功或失败标志可供使用。
- 终止条件（身体高度超限或 NaN）在 `compute_reward` 中无法感知，因此无法设计基于 episode 结束的稀疏奖励或惩罚。

## 哪些组件留到后续迭代
- **energy_penalty（动作代价）**：v1 阶段不应引入，否则可能导致机器人不敢充分探索动作空间。
- **action_smoothness_penalty（动作平滑）**：需要访问历史动作，但当前接口无法获得上一步动作，暂不可用。
- **alive_bonus（存活奖励）**：需要 done/terminated 信号，当前不可用；且过早加入可能诱导原地不动。
- **更复杂的 stability penalty（如基于四元数的倾斜惩罚）**：后续迭代若姿态失稳严重，可引入。
- **gated_reward / dynamic_curriculum**：复杂门控与课程设计留待后续。

## 训练后应观察的 failure mode
1. **fast_then_fail**：机器人可能高速冲刺但姿态失控导致摔倒。需要观察 forward 奖励是否过高，是否需加强稳定性约束。
2. **hover_or_stand_still**：若无 forward 奖励或权重过低，可能学习到静止策略。当前设计直接奖励前进速度，应不会出现；但若稳定性惩罚过强，也可能抑制移动，需检查。
3. **over_conservative_policy**：若 lateral 或 angular 惩罚过重，机器人可能迈步极小、移动缓慢。后续可根据实际步速调整权重。
4. **侧向漂移或原地转圈**：若 lateral_velocity_penalty 或 angular_velocity_penalty 权重不足，可能出现偏航漂移。