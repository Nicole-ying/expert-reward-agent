# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------
    # 1. 主学习信号：朝向目标的进度
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：抑制高速、大角度和高角速度，确保着陆平稳
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = - w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost

    # 3. 软着陆近似奖励：多条件组合，引导飞行器低速、低角度、双足接触着陆
    near_target = dist_next < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.5
    stable_angle = abs(next_obs[4]) < 0.1
    both_contacts = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contacts) else 0.0

    # 总奖励
    total_reward = progress + stability_penalty + soft_landing_bonus

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 奖励组件及角色

| 组件 | 角色 | 数学形式 |
|------|------|----------|
| `progress` (主信号) | 密集进度引导，每步奖励飞行器缩短与目标平台的距离 | `dist_curr - dist_next`，scale = 1.0 |
| `stability_penalty` (稳定约束) | 抑制水平/垂直速度、机体倾斜和角速度，促使平稳着陆 | `- w_vel * (|vx|+|vy|) - w_angle * |angle| - w_angvel * |angular_vel|`，所有权重极小 |
| `soft_landing_bonus` (任务完成近似) | 为“接近目标＋低速＋小角度＋双足接触”这一稳定着陆状态提供正向强化 | 多条件二值 bonus = 2.0，触发条件严格，避免 contact reward hacking |

## 2. 为何未使用 terminal_success_reward / terminal_failure_penalty

- **环境卡片明确声明**：`explicit_success_flag_available = false` 且 `explicit_failure_flag_available = false`。
- `info` 字典恒为空，无法获取任何成功/失败标志。
- 终止原因中的 `crash_or_body_contact` 含义模糊，既可能包含不安全碰撞也可能包含安全接触，不能直接当作成功或失败信号。
- 因此 v1 不依赖任何终端信号，而是通过密集进度 + 稳定性约束 + 软着陆条件组合来间接塑造成功行为。

## 3. 后续迭代预留

- **效率/燃油代价**（`energy_penalty`）：当前动作空间离散，但引擎使用会消耗燃料。v1 先保证学会稳定着陆，后续可加入小权重燃料惩罚。
- **时间效率**（`time_penalty`）：v1 暂不引入，避免 agent 因害怕步数惩罚而不敢调整姿态。
- **更强的稳定性约束或门控**：如果 `progress` 和 `soft_landing_bonus` 导致 agent 在目标附近快速震荡而不真正接触，可在 v2 收紧 `soft_landing_bonus` 的条件或加入 `progress` 的 clip。
- **terminal 信号**：如果后续 wrapper 能可靠提供 `info['success']`，可加入 terminal success reward 替代 `soft_landing_bonus`。

## 4. 训练后应观察的 failure mode

- **high_reward_without_success**：agent 在目标平台上方小幅度来回摆动，获得持续正向 `progress` 和较低的稳定性惩罚，但始终不接触。此时应监控 `soft_landing_bonus` 触发频率和 episode 终止情况。
- **fast_crash_near_goal**：agent 为快速接近目标而获得高 `progress`，但垂直速度过大导致 `crash_or_body_contact` 终止。需观察成功着陆次数和平均速度。
- **contact_reward_hacking**：若 `soft_landing_bonus` 条件太宽松，agent 可能反复轻触平台以重复获得 bonus。应监控同一 episode 内 bonus 触发次数，若 >1 则需在 v2 增加“仅首次成功”逻辑或更严格的速度/角度阈值。
- **agent_afraid_to_move**：稳定性惩罚过大可能导致 agent 完全不敢加速，悬停不动。可通过步均速度监控，若始终接近零则适当降低 `stability_penalty` 权重。