# Response Record

## 1. 诊断报告

### evidence（关键数据与历史）

- 历史轨迹：`iter_1 score=-17.27`（无门控，向前摔倒，`forward_velocity_reward=222.6`）→ `iter_2 score=262.56`（门控 0.5）→ `iter_3 score=255.93`（门控收紧 0.4，惩罚加强）→ `iter_4 score=287.49`（门控放宽 0.6）。
- **本轮改善 >30 分的归因明确**：对比 `read_reward_code(3)` 与 `read_reward_code(4)`，iter_4 唯一实质修改是把直立门控宽度 `0.4 → 0.6`。证据链：
  - `forward_velocity_reward` ep_sum：`780.13 → 818.92`（+5.0%），说明 agent 获得了更高的水平速度奖励；
  - 外部分数：`255.93 → 287.49`（+31.56），说明更宽的直立门控让 agent 能采用中等前倾、更快的步态，而不是被门控过度压制前进；
  - `upright_stability_penalty` ep_sum：`-8.05 → -0.07`，active_rate 仍只有 4.1%，说明加强后的兜底惩罚几乎不被触发——agent 并没有重新滑向“向前摔倒”。
- 当前 `score=287.49`，目标 300，差距 12.51（约 4.2%），属于接近目标区间。
- `vertical_oscillation_penalty` 全程为 0，是死组件；`upright_stability_penalty` 量级几乎为 0 但作为兜底护栏保留；`forward_velocity_reward` 占 magnitude_share 100%，是唯一有效驱动信号。

### behavior_diagnosis

- 当前策略是**健康的快速前进步态**：`terminated=20/20`、`truncated=0/20`、`len=1207.5`，且没有早摔（early_terminal 0/20）。
- `upright_stability_penalty` 几乎为 0 说明 agent 已经学会在门控允许范围内保持躯干稳定，没有走“向前摔倒刷奖励”的老路。
- 这不是不健康行为，而是**前进速度/距离还差最后一小段**。分数区间 `[284.57, 289.27]` 很窄，说明各 episode 行为一致、接近但尚未触及 300 目标，瓶颈更可能是平均前进速度还不够快，而不是稳定性崩坏。

### signal_completeness

- 主信号 `forward_velocity_reward` 完备且有效；`upright_stability_penalty` 作为防摔倒护栏职责存在且几乎不触发（健康的兜底状态）；`vertical_oscillation_penalty` 目前是死组件，但无害、暂不必动。
- 不缺关键信号，当前不需要引入能量惩罚或步态对称性等次目标——那会与“前进主目标”混淆，尤其在接近目标时反而有风险。

### selected_level

- **L1**。理由：
  1. 当前距目标差距约 4.2%，符合“接近目标时优先 L1”的规则；
  2. 改善来源已经明确（门控 0.4→0.6），这是一个已经被验证为正向的方向；
  3. 本轮只需要在同一个被证明有效的方向上做系数微调，不需要改数学形态。

### selected_intervention

- 目标组件：`forward_velocity_reward` 中的 `upright_gate`。
- 具体修改：把直立门控宽度 `0.6 → 0.7`。
- 其余全部保持不变（`speed_reward` 的 1.5 温度与系数、`upright_stability_penalty` 的 -4.0/-0.4 兜底、死组件 vertical 项均不动）。

### falsifiable_hypothesis

- iter_4 相对 iter_3 的改善来自“更宽的直立门控 → 允许中等前倾 → 更快步态”。把门控从 0.6 进一步放宽到 0.7，应允许略微更大的自然前倾/蹬地角，从而进一步提高平均水平速度与前进距离，逼近 300；
- 同时，上一轮已加强的 `upright_stability_penalty` 仍作为兜底存在（在超过 0.15 rad 后以 -4.0·margin² 增长），足以阻止门控放宽后重新滑向“向前摔倒”。
- 若下一轮 `upright_stability_penalty` 的 active_rate 或 ep_sum 明显回升、分数反而下降，则说明 0.7 已越过了安全边界，应回退到 0.6 或 0.65。

### expected_next_round

- `forward_velocity_reward` ep_sum 预计继续小幅上升（速度提高）；
- `upright_stability_penalty` 预计仍保持接近 0（active_rate 可能略有上升但仍是小量级）；
- 分数预计向 290–300 收敛，`len` 可能略缩短或持平（走得更快），早摔数保持 0。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干保持直立时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：上一轮 0.4 -> 0.6 已带来显著改善（score 256 -> 287）。
    # 本轮继续沿同一方向微调（0.6 -> 0.7），允许略微更大的自然前倾换取更快步态，
    # 同时仍保留强力的兜底惩罚，防止回到"向前摔倒刷前进奖励"。
    hull_angle = float(next_obs[0])
    upright_gate = 2.718281828 ** (-(hull_angle / 0.7) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：保持上一轮加强后的力度（角度 -4.0，角速度 -0.4），
    # 作为放宽门控后的兜底护栏，防止向前摔倒。
    angle_margin = max(0.0, abs(hull_angle) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -4.0 * angle_margin ** 2
        - 0.4 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：保持不动。
    vertical_margin = max(0.0, abs(float(next_obs[3])) - 0.6)
    vertical_oscillation_penalty = -0.3 * vertical_margin ** 2

    total_reward = (
        forward_velocity_reward
        + upright_stability_penalty
        + vertical_oscillation_penalty
    )

    components = {
        "forward_velocity_reward": float(forward_velocity_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "vertical_oscillation_penalty": float(vertical_oscillation_penalty),
    }

    return float(total_reward), components
```
