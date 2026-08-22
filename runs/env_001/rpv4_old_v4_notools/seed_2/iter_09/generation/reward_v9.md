**1. 诊断报告**

- **evidence**：
  - 最终策略 reward 组成中 `contact_bonus` 的 `magnitude_share` 高达 92.3%，`episode_sum_mean = 681.5`，而其他组件（progress 10.4, stable_contact_bonus 40.9）贡献极小。`contact_bonus` 的 `active_rate` 仅 13.2%，说明在一小部分步数中产生了极其巨大的累计奖励。
  - 训练过程中 `contact_bonus` 的 ep_sum 从 cp10% 的 9.25 飙升到 cp100% 的 548.5，同时 `orientation_penalty` 持续恶化（-0.64 → -2.14），表明 agent 正在拼命“刷”接触事件，而非收敛到一个干净的着陆动作。
  - 最终评估中 `terminated=4/20`，`truncated=16/20`，平均 episode 长度 896 步。绝大部分 episode 因超时被截断，agent 并没有频繁触发环境的成功终止条件。
  - 这些证据共同指向：**agent 学会了通过反复弹跳来反复触发接触事件奖励**，而不是稳定停靠。

- **behavior_diagnosis**：
  策略当前的行为模式是：快速接近平台后，反复轻微弹跳以反复获得 `contact_bonus`。由于每次弹跳都能获得 5～10 分的奖励，而稳定停留在平台上每步只能获得 0.1 分的 `stable_contact_bonus`，agent 没有动力保持稳定。这种模式导致 episode 长度极长、成功终止率极低，外部任务分数停滞在 ~80 分左右，是**不健康的 exploit 行为**。

- **signal_completeness**：
  必要的职责已经具备——`progress` 提供趋向目标的动力，`velocity_penalty`、`orientation_penalty` 提供安全约束，`stable_contact_bonus` 鼓励保持低触点低速度。但**缺失一个只给予一次（或极有限次）的“完成稳定着陆”事件信号**，而现有的 `contact_bonus` 实际上是一个可无限重复获取的**代理奖励陷阱**。

- **selected_level**：**L2**
  - 当前得分 78 距目标 200 差距 >10%，非微调区间。
  - `contact_bonus` 的数学形态（每步事件检测）直接导致了弹跳 exploit，需要从根本上改变其语义，不能仅靠调权重解决。

- **selected_intervention**：
  - **目标组件**：`contact_bonus` → 替换为 `stable_landing_bonus`。
  - **修改内容**：不再奖励任何一笔“腿从非接触到接触”的事件，而是奖励“从非稳定状态首次进入稳定着陆状态”的一次性事件：当满足 `至少一条腿接触`且`速度 < 0.2`且`倾斜角 < 0.1 rad`时，且上一帧不满足该组合条件时，给予 **+10.0** 的一次性奖励。
  - **联动调整**：将 `stable_contact_bonus` 从 0.1 提升到 0.5，以补偿总奖励的下降，并使 agent 在完成着陆后依然有动力维持稳定状态（服务于同一语义：从“鼓励反复接触”转变为“鼓励一次完成并保持稳定”）。

- **falsifiable_hypothesis**：
  取消可复现的接触事件奖励后，agent 将无法再从弹跳中获得净收益。`progress` 会推动它靠近目标，`stable_landing_bonus` 会引导它在正确时机一次性完成低速度、低角度的稳定接触，而增强的 `stable_contact_bonus` 会驱使它保持住该状态直至 episode 终止。这一改变预计会**大幅提高成功终止率，缩短 episode 长度，并改善 orientation_penalty**，从而推动外部任务分数上升。

- **expected_next_round**：
  下一轮策略的学习行为预计为：快速、平稳地接近平台，在接触瞬间满足低速度、低角度条件以获取一次性高额奖励，随后持续维持稳定接触直到 episode 终止。`terminated` 比率将明显上升，`episode_length` 下降，`orientation_penalty` 不再恶化。

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable landing event bonus (replaces per-step contact bonus)
    # Check if any leg is touching
    leg_prev = obs[6] > 0.5 or obs[7] > 0.5
    leg_next = next_obs[6] > 0.5 or next_obs[7] > 0.5

    vel_prev = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    vel_next = (vx ** 2 + vy ** 2) ** 0.5

    angle_prev = abs(obs[4])
    angle_next = abs(next_obs[4])

    # Stable condition: touching + low speed + upright
    prev_stable = leg_prev and vel_prev < 0.2 and angle_prev < 0.1
    next_stable = leg_next and vel_next < 0.2 and angle_next < 0.1

    stable_landing_bonus = 10.0 if (next_stable and not prev_stable) else 0.0

    # --- stable contact reward: encourage maintaining stable contact
    contact_active = next_obs[6] + next_obs[7]
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.5  # increased to balance total reward

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_landing_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_landing_bonus": stable_landing_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components
```