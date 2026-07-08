# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_support_contact, 7: right_support_contact
    # goal is at (0,0).

    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: penalize large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.01
    w_angle = 0.1
    w_angvel = 0.05
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: boost when almost on target, slow, upright, and both feet contact.
    dist = d_next
    contact_both = next_obs[6] * next_obs[7]   # 1.0 only when both feet touch
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_both *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 使用了哪些奖励组件

- **approach_reward**：主学习信号，稠密、每步有梯度。计算当前到目标的距离变化 `d_prev - d_next`，鼓励飞行器朝平台（坐标原点）靠近。
- **stability_penalty**：轻量稳定约束，惩罚水平/垂直速度、机体倾角和角速度。抑制高速、大角度飞行，促使软着陆和姿态平稳。
- **soft_landing_proxy**：任务完成近似信号（proxy）。当飞行器同时满足 **靠近平台、低速度、小倾角、双脚接触** 四个条件时，获得全局最大奖励。使用连续指数衰减函数组合，避免硬二值条件。

## 2. 每个组件的角色

| 组件 | 角色 | 学习功能 |
|------|------|----------|
| `approach_reward` | 主学习信号 | 告诉 agent 每一步“更靠近目标即得分”，是任务的核心驱动力。 |
| `stability_penalty` | 稳定/安全约束 | 抑制猛烈机动和大幅度摇摆，引导飞行器以平稳姿态接近平台。 |
| `soft_landing_proxy` | 任务完成锚点 | 在 agent 即将成功并稳定着陆时提供一个明确的全局正向激励，加速收敛到“稳定停驻”行为。 |

## 3. 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境 `info` 为空字典，显式成功/失败标志不可用（`explicit_success_flag_available = false`，`explicit_failure_flag_available = false`）。
- 环境终止原因（如 `body_not_awake_or_settled`、`crash`、`horizontal_position_outside_viewport`）只在 episode 结束时告知环境，并未暴露在 `info` 中，无法可靠地判定终止类型。
- 因此，不可能也不应构造依赖这些标志的奖励项。`soft_landing_proxy` 通过可观测的传感器信号组合来近似任务成功，而不伪造 success flag。

## 4. 哪些组件留到后续迭代

- **能量/动作代价（energy_penalty）**：待 agent 能稳定完成任务后，再引入对引擎使用的惩罚，以优化能耗。
- **时间效率约束（time_penalty）**：同样留待基本任务成功率达标后再加入，避免早期抑制探索。
- **动态课程 / 门控（gated_reward）**：若发现高速接近目标与稳定着陆之间存在冲突，可在后续阶段加入阶段门控，当前 v1 保持简洁。
- **显式成功奖励（terminal_success_reward）**：如果未来 wrapper 能暴露可靠的 `success` 标志，可考虑直接替代 proxy。

## 5. 训练后应该观察哪些 failure mode

- **high_reward_without_success（高分不落地）**：agent 在目标附近低振幅振荡，累积 `approach_reward` 但不真正稳定着陆。应检查 `stability_penalty` 权重是否需要增强，或者 `soft_landing_proxy` 的门限是否过严。
- **fast_crash_near_goal（接近目标时撞毁）**：agent 为快速接近而高速撞击平台或其他障碍。此时 `approach_reward` 可能会制造过度冒险行为。后续可考虑增加接近目标时的自动减速信号（如距离加权稳定性惩罚）。
- **contact_reward_hacking（接触奖励作弊）**：`soft_landing_proxy` 中的双脚接触条件可能诱使 agent 刻意“点地”弹跳而不稳定停驻。由于 proxy 同时要求低速、小倾角且位置接近 0，该风险较低，但仍应在定性分析中观察是否有反复轻触现象。
- **agent_afraid_to_move（推力恐惧）**：`stability_penalty` 若设得过强，agent 可能不敢启动引擎，停在起点附近。检查训练早期平均速度是否过低；若是，可适当调低 `w_speed`、`w_angle` 等系数。
