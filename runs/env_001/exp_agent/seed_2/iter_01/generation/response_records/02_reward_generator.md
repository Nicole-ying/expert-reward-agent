# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标

    # 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = 0.2 * abs(next_obs[5])  # 角速度惩罚
    speed_penalty = 0.1 * speed  # 速度惩罚
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎的动作（action 1,2,3）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
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

1. **progress_reward**（主学习信号）：基于每一步到目标距离的变化量（progress_delta），奖励智能体更接近目标。这是核心引导信号，提供密集的、与任务目标直接相关的学习信号。

2. **stability_penalty**（稳定约束）：惩罚高速、大姿态角和大角速度，鼓励智能体以稳定、可控的方式接近目标。这对于着陆任务至关重要，防止智能体高速撞击或姿态失控。

3. **landing_bonus**（任务完成 proxy）：当智能体同时满足接近目标、低速、小姿态角和双接触时给予小奖励。这是一个软性的成功近似信号，鼓励智能体完成最终着陆，但不作为硬性成功条件。

4. **energy_penalty**（动作代价）：小权重惩罚使用引擎，鼓励智能体高效使用燃料。权重很小（-0.05），避免智能体因害怕使用引擎而不敢移动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 字典为空，没有可用的成功/失败标志。因此，在 v1 中不使用终端奖励，避免伪造信号或误判终止原因。

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：当 wrapper 能明确暴露成功/失败标志后再加入。
- **time_penalty**：如果智能体能接近目标但拖太久，后续可小权重加入。
- **gated_reward**：如果安全约束被进度奖励抵消，后续可加入门控机制。
- **potential_based_shaping**：如果需要更标准的塑形，后续可替换 progress_delta。

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡，无法稳定着陆。如果出现，需要增大 stability_penalty 或调整 landing_bonus 条件。
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆。如果出现，需要收紧 landing_bonus 条件或加入更严格的稳定约束。
3. **fast_crash_near_goal**：智能体高速冲向目标但坠毁。如果出现，需要增大速度惩罚或加入速度上限约束。
4. **agent_afraid_to_move**：智能体因惩罚过重而不敢移动。如果出现，需要降低 stability_penalty 或 energy_penalty 的权重。
