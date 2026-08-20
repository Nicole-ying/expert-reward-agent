# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: drive the agent to reach and settle on the central pad.
    Components:
      - progress_delta_reward: reward reduction in distance to the target (0,0)
      - stability_penalty: small penalty for high speed, tilt and angular velocity
      - soft_landing_proxy: bonus when near target, slow, upright and both feet in contact
    All components use only allowed signals (obs, next_obs, action). No terminal flags used.
    """
    # -- Helper: distance to target (target is at (0,0))
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (main driving signal)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (light constraint) on next state
    w_vel   = 0.01   # penalty per unit speed sum
    w_angle = 0.01   # penalty per radian of body angle
    w_angvel= 0.001  # penalty per angular velocity unit

    stability_penalty = (
        -w_vel   * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle * abs(next_obs[4])
        -w_angvel* abs(next_obs[5])
    )

    # -- 3. Soft landing proxy: conditions for being safely on the pad
    dist_threshold   = 0.5
    vel_threshold    = 0.3
    angle_threshold  = 0.2   # rad (~11.5 deg)

    near_target = dist_next < dist_threshold
    low_speed   = (abs(next_obs[2]) < vel_threshold) and (abs(next_obs[3]) < vel_threshold)
    upright     = abs(next_obs[4]) < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    w_proxy = 0.5
    soft_landing_proxy = w_proxy if (near_target and low_speed and upright and both_contact) else 0.0

    # -- Total reward
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **progress_reward（主学习信号）**  
   基于距离差的稠密奖励：每一步距离目标 `(0,0)` 的缩短 `d(obs) - d(next_obs)` 都获得正分，远离则负分。它直接告诉 agent“靠近中心就是好的”，是整个策略的主要驱动力。为了保证梯度连续，使用欧氏距离的差值，不加 clip（后续可加轻微截断以避免极端值）。

2. **stability_penalty（稳定约束）**  
   轻量惩罚高速、大倾角和快角速度，鼓励平稳飞行和着陆。权重故意设得很小（`w_vel=0.01, w_angle=0.01, w_angvel=0.001`），不会阻止 agent 必要的加速和姿态调整，但能抑制来回震荡和剧烈翻滚，减轻 `progress_reward` 诱导的目标附近振荡风险。

3. **soft_landing_proxy（任务完成近似信号）**  
   proxy 仅在接触正确且稳定时激活：同时满足 `距离<0.5, 速度<0.3, 角度<0.2 rad, 双足接触`。由于环境没有显式成功标志，该组件通过多条件连续组合近似“成功着陆”，为 agent 提供额外的锚点，鼓励它一旦到达就保持接触和低速。每步给予常数小奖励 `0.5`，不会过度膨胀，也不会替代 progress 信号。

## 为何未使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available = false`：info 为空，既没有 `success` 也没有明确的失败原因字段。因此严格遵循约束，不使用任何依赖显式终止标志的奖励。  
- 终止条件 `body_not_awake_or_settled` 可能表示稳定着陆，但也可能发生在平台之外，无法安全区分，所以不能用它来构建 `terminal_success_reward`。  
- 失败终止（碰撞或飞出边界）没有单独 flag，若强行使用 `terminal_failure_penalty` 必然误判，故不在 v1 引入。

## 后续迭代计划

- **效率/动作代价（energy_penalty）**：环境要求在完成目标的同时“尽量少使用引擎推力”，待 v1 能稳定着陆后，以小权重惩罚主引擎和姿态引擎的使用，引导节能飞行。  
- **dynamic_curriculum / 门控**：若 proxy 条件过于宽松导致 agent 仅靠近接触却不完全稳定，可在后续收紧阈值或引入需要连续 N 步满足才给奖励的门控。  
- **potential_based_shaping**：如果 `progress_reward` 与 `stability_penalty` 的交互造成某些奇怪行为，可以考虑使用标准势能塑形，进一步提高学习稳定性。  
- **terminal_success_reward**：当环境 wrapper 明确暴露 success 标志后（例如解包 `info` 增加字段），可加入大型成功奖励以加速收敛。

## 训练后应观察的 failure mode

1. **目标附近振荡不着陆**：  
   `progress_reward` 在接近目标时增量变小，agent 可能会来回小幅加速以赚取微小正 delta。`stability_penalty` 应能部分抑制，但如果权重过小，仍需观察。若出现，可在下一版增加接近时的额外减速鼓励或限制速度上限。

2. **reward hacking on soft_landing_proxy**：  
   agent 可能学会在平台边缘反复轻触（接触成立但速度尚未完全归零），以反复获得 proxy 奖励。本例中将 `velocity < 0.3` 与 `both_contact` 结合，大幅降低该风险，但仍需关注。若仍发生，可要求连续满足条件或只在首次触发时给一次性奖励。

3. **拒绝下降（freezing）**：  
   如果 `stability_penalty` 对速度惩罚过强，agent 可能选择几乎不动以避免速度惩罚。当前权重很小，但若出现，应先减小稳定性系数或增加 progress 权重。

4. **角度过大致损探索**：  
   `stability_penalty` 包含角度惩罚，可能让 agent 不敢使用姿态引擎调整姿态。但因为动作离散且需要转动才能横向移动，该惩罚不宜过大。如果观察到 agent 无法有效水平移动，应考虑移除角度项或进一步降低权重。