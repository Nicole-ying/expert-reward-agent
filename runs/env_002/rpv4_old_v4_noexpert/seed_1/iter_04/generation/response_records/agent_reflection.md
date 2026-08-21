# Response Record

## 1. 诊断报告

### evidence（关键数据与历史）

- 历史轨迹：`iter_1 score=-17.27`（无门控，向前摔倒，`forward_velocity_reward=222.6`）；`iter_2 score=262.56`（best，门控宽 0.5，软惩罚）；`iter_3 score=255.93`（门控收紧 0.5→0.4，惩罚加强 -2.0→-4.0 / -0.15→-0.4）。
- iter_3 的收紧是负向修改：`forward_velocity_reward` 从 `814.5 → 780.1`，外部分数从 `262.56 → 255.93`，说明更窄的直立门控直接压低了前进速度。
- iter_3 最终策略 `upright_stability_penalty` active_rate 仅 `1.6%`，即 agent 已经非常直立、没有摔倒倾向，说明当前问题不是“不够稳”，而是“不够快”。
- `vertical_oscillation_penalty` 三轮 active_rate 全部为 0%，是死组件，但无害。
- iter_3 checkpoint 趋势显示训练中 forward 与 upright 交替波动（cp40 前进奖励骤升伴随惩罚上升），最终收敛到低惩罚、高前进奖励，说明结构健康，只是速度被门控卡住。

### behavior_diagnosis

当前策略已经学会稳定双足行走：`terminated=20/20`、`len≈1180`、躯干惩罚几乎不触发，且没有 iter_1 的“向前摔倒刷前进奖励”行为。这不是不健康策略，而是**速度被过度约束的健康慢速步态**。要冲 300 分，主前进信号需要允许更大的自然前倾、跑出更高水平速度。

### signal_completeness

必要职责基本完备：前进主信号（`forward_velocity_reward`）、躯干稳定约束（`upright_stability_penalty`）都有；vertical 组件死亡但属于次目标。当前缺少的是**更宽松的速度-姿态权衡**：0.4 的高斯门控对中等前倾惩罚过重，导致 agent 宁愿保持低速也不愿前倾加速。

### selected_level

**L1**。best=262.56 距目标 300 差距约 12.5%，接近目标；且上一轮 L2 式收紧已证明有害。本轮只调整门控宽度系数，不改数学形态。

### selected_intervention

目标组件 `forward_velocity_reward`：把 `upright_gate` 的宽度从 `0.4` 放宽到 `0.6`（`exp(-(hull_angle/0.6)^2)`）。保留 iter_3 加强后的 `upright_stability_penalty` 作为极端倾斜的兜底护栏，避免回到 iter_1 的摔倒刷分。

### falsifiable_hypothesis

更宽的高斯门控会降低中等前倾（约 0.3–0.5 rad）对前进奖励的惩罚，使“更快步态需要的前倾”变得有利可图，从而提升水平速度、提高外部分数。若下一轮 `forward_velocity_reward` 上升、score 突破 262.56 且 `upright_stability_penalty` 仍保持低 active_rate，则假设成立；若 penalty 骤升且 score 下降，说明门控过宽重新引入了摔倒刷分。

### expected_next_round

策略预计会跑出更快的步态：`forward_velocity_reward` 应高于 780（接近或超过 iter_2 的 814），episode 可能因更快到达地形末端而变短，`upright_stability_penalty` active_rate 会略有上升但仍处于低位，score 应高于 262.56 并向 300 靠近。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干保持直立时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：本轮放宽宽度（0.4 -> 0.6），允许中等前倾换取更快的步态，
    # 同时仍强烈抑制接近摔倒的大角度倾斜。
    hull_angle = float(next_obs[0])
    upright_gate = 2.718281828 ** (-(hull_angle / 0.6) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：保留上一轮加强后的力度（角度 -4.0，角速度 -0.4），
    # 作为放宽门控后的兜底护栏，防止回到"向前摔倒刷前进奖励"。
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
