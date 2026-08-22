# Response Record

**1. 诊断报告**

- `evidence`：
  - 最终得分 -307.97，远低于目标 200；episode_length 平均数 956.8（接近最大步数 1000），terminated 仅 1/20，绝大多数被截断。说明策略未完成着陆任务，而是在徘徊。
  - `landing_reward` magnitude_share 99.9%，ep_sum_mean 高达 5647.93，是一个稠密的、始终为正的信号；`progress_reward` 贡献极微（-3.34），`crash_penalty` 全程为零。
  - Checkpoint 趋势显示 `landing_reward` 在训练中不仅持续为正，而且逐渐升高（从 453 增至 5906），而 `progress_reward` 始终负值或接近零，总得分仍为负，说明 agent 被 `landing_reward` 支配，学习到的是「在目标附近保持低速、小角度从而持续获得 proxy 奖励」的行为，而非真正着陆。
  - `landing_reward` 的构成包含 `prox`（距离）、`contact`（平均接触）、`angle_factor`、`angvel_factor`、`speed_factor` 的均值，即使无接触也可获得约 8.0 的步奖励，这使 agent 缺乏完成双接触的动力，反而通过永不结束的徘徊累积大量奖励。

- `behavior_diagnosis`：
  策略已经学会了一个「proxy exploitation」行为：在目标点附近维持低速、小角度，不触发双接触，从而不断获取 `landing_reward`。由于 episode 被截断而非因成功终止，总步数大，环境原始评分（或燃料消耗）拉低分数至 -307。这极不健康，与「稳定停靠在目标平台」的目标完全偏离。

- `signal_completeness`：
  缺失的职责是 **任务完成事件信号**。当前没有「成功双接触平台」的正面奖励，也没有终止 motivation（持续 proxy 反而抑制了完成任务）。`crash_penalty` 从未激活，可暂保留但不关键。`progress_reward` 方向正确但被淹没，需提升权重。

- `selected_level`：**L2**
  证据表明当前的主信号框架（持续 proxy）被 exploit，必须改变数学形态：移除持续的 `landing_proxy`，替换为一次性接触事件奖励。这不是简单调参（L1），而是语义级别的组件替换，但又未到全盘推倒主框架（L3），因为接近目标的方向信号（progress）仍然有价值。

- `selected_intervention`：
  **目标组件**：`landing_reward` → 替换为 `touchdown_bonus`（双接触一次性奖励）；同时提高 `progress_reward` 系数，使其成为过程主要驱动。
  **具体修改**：
  - 删除原有 `landing_proxy` 和其内在的五因子均值公式。
  - 新增 `touchdown_bonus`：当 `next_obs` 中的 `left_contact` 和 `right_contact` 同时为 1 时，给予 +200.0 的奖励（假设双接触会立即终止 episode，不会重复累积）。如果环境不一定终止，但仍假设双接触是唯一的成功标志，奖励一次性足够大，且之后 episode 终止的自然积累不会再有。
  - 将 `progress_reward` 系数从 2.0 提升到 5.0，使其在早期有足够力量引导接近。
  - 保留 `crash_penalty` 不变（防御性约束），尽管一直未激活。

- `falsifiable_hypothesis`：
  如果 agent 学会为获取一次性 `touchdown_bonus` 而真正完成双接触着陆，则 episode 长度将显著下降，terminated 比例上升，同时环境评分将向正方向移动（接近 200）。progress 系数提升可加速早期接近，不再被 proxy 淹没。

- `expected_next_round`：
  训练后 episode_length 会下降到成功着陆所需的步数（例如 200-400），terminated 比例 > 80%，score 转为正值并接近 200。`progress_reward` 的 ep_sum_mean 将变为正且主导前期，`touchdown_bonus` 在末步触发，`crash_penalty` 仍接近 0（因为任务不触发边界违规）。agent 的行为将是快速接近、减速、调整姿态、最终双接触着陆。

---

**2. 修改后的奖励函数代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- observations ----------
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # ---------- distances ----------
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. progress reward (approach incentive) ----------
    progress_reward = 5.0 * (dist - n_dist)  # positive when moving toward target

    # ---------- 2. touchdown bonus (task completion) ----------
    # Large one-time reward when both legs touch the platform.
    # Assuming the episode terminates immediately after this condition,
    # so no repeated accumulation of this bonus.
    both_contacts = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    touchdown_bonus = 200.0 if both_contacts else 0.0

    # ---------- 3. crash / boundary constraints ----------
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    v_violation = max(0.0, -ny_vel - 3.0)   # excessive downward speed
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + touchdown_bonus + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components
```
