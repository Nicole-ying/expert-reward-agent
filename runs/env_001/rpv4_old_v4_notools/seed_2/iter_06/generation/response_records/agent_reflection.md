# Response Record

**1. 诊断报告**

- `evidence`：
  - iter_5 最终 score=-64.2，early terminal (<150 steps & score<-50) 17/20，agent 大量早期失败（很可能是高速冲撞边界/地面）。
  - magnitude_share 显示 `proximity_pressure` 独占 82.8%，其 ep_sum_mean = -26.8 且全程几乎不变（checkpoint: -26.7 → -26.8）。
  - `progress`、`velocity_penalty`、`orientation_penalty` 也几乎常数，表明它们对学习几乎无贡献——整个优化被单一的负向距离压力支配。
  - `contact_state` active_rate 仅 3.4%，agent 极少碰到平台，更谈不上稳定停靠。
  - 行为模式推测：`proximity_pressure = -0.4*dist` 迫使 agent 不顾一切地减少距离，直接冲向中央，忽略了减速与姿态控制，导致绝大多数 episode 在接触平台前就因高速/倾覆而早期终止。

- `behavior_diagnosis`：
  当前策略是“盲冲中心”——它被单一的距离惩罚驱动快速接近目标，却未能学会安全着陆所需的减速与姿态稳定。这是一种不健康的 exploit：距离确实在减小（score 也从 -113 升至 -65），但最终行为仍以坠毁为主。

- `signal_completeness`：
  必要职责缺失：**到达目标的安全约束**（在接近平台时限制速度）以及**触地/着陆的正向成功信号**过弱。当前的 `contact_state` 量级太小且无质量区分，无法有效引导软着陆。

- `selected_level`：**L2** — 不是因为 L1 调参不行，而是当前的主信号（全程距离惩罚）从语义上就是有害的，必须改变数学形态：将其移除，用更强的进度奖励和高质量的接触奖励替代。

- `selected_intervention`：
  1. **删除 `proximity_pressure`** — 移除持续的负向距离压迫，避免强迫 agent 以任何代价冲向中心。
  2. **提升 `progress` 权重**（2.0→4.0）— 保持向目标移动的梯度，但不再附加惩罚恐惧。
  3. **重写接触奖励为事件+质量组合**：
     - 检测腿接触的上升沿事件，每次新接触给予较大奖励（`contact_event`）。
     - 若已接触且垂直速度很小，每步给予稳定接触奖励（`stable_contact_bonus`），鼓励减速并保持停靠。
  4. 保持 `velocity_penalty` 和 `orientation_penalty` 的结构，轻微调大速度惩罚系数（-0.02→-0.03）以适应量级平衡。

- `falsifiable_hypothesis`：
  移除压迫性距离惩罚后，agent 不再被迫盲冲，`progress` 和大幅提高的接触事件/停靠奖励将引导它学习“减速→接近→轻柔接触→稳定停靠”的完整行为，从而 early terminal 比例下降，`contact` 组件的 active_rate 和 ep_sum 上升，最终分数显著改善。

- `expected_next_round`：
  下一轮策略预计会存活更久（truncated 比例上升，early terminal 下降），在接近平台时有明显的减速动作，`contact_event` 和 `stable_contact_bonus` 的激活率与幅度提高，整体奖励分布由 progress 和接触奖励共同主导，score 有望提升 20-40 分（移除了 ~-26 的惩罚并加入正面着陆奖励）。

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (used for progress shaping, not for penalty)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- progress: strong shaping towards target
    progress = 4.0 * (dist_prev - dist_next)

    # --- velocity penalty: penalize high speed to encourage safe landing
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty: discourage large tilt
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact event reward: bonus for newly touching a leg to the platform
    leg_left_prev = obs[6]
    leg_right_prev = obs[7]
    leg_left_next = next_obs[6]
    leg_right_next = next_obs[7]

    contact_event = 0.0
    if leg_left_next > 0.5 and leg_left_prev < 0.5:
        contact_event += 1.0
    if leg_right_next > 0.5 and leg_right_prev < 0.5:
        contact_event += 1.0
    contact_bonus = 5.0 * contact_event

    # --- stable contact reward: small per-step reward for maintaining contact with low vertical speed
    contact_active = leg_left_next + leg_right_next
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 2.0

    # --- combine (no proximity_pressure, no continuous distance penalty)
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + contact_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_bonus": contact_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components
```
