# Response Record

## 1. 诊断报告

### evidence（关键数据与历史）

- iter_1 最终策略：`score=-17.27`，`len=375.45`，`terminated=20/20`，`truncated=0/20`，`score_range=[-92.27, 277.51]`。
- 最终评估分布：`early_terminal (<150 steps 且 score<-50): 14/20`。即最终策略大多数 episode 早摔。
- 奖励构成：`forward_velocity_reward` ep_sum=222.6，`magnitude_share=97.2%`，`active_rate=94.3%`；`upright_stability_penalty` ep_sum=-6.5，`magnitude_share=2.8%`；`vertical_oscillation_penalty` 全为 0（死组件）。
- checkpoint 总分趋势：`-95.1 → -55.4 → 82.8 → 106.1 → 138.0 → 89.4 → 128.5 → 181.2 → 212.3 → 46.7`。训练中段到后期分数整体上升，cp90% 达到 212.3，但 cp100% 骤降到 46.7，最终策略进一步崩到 -17。
- checkpoint 中 `forward_velocity_reward` 从 cp30% 起持续高位（422→773），cp100% 仍 434；`upright_stability_penalty` 量级全程很小（-17~-55）。

### behavior_diagnosis（当前行为模式）

策略被单一的 `forward_velocity_reward` 主导（97% 份额），学习到的核心行为是**追求高水平速度**。由于该奖励只依赖 `next_obs[2]` 的水平速度，**向前摔倒的过程同样会产生很大的水平速度**，因此“冲刺→失衡→摔倒”也能拿到大量正奖励。这解释了：训练后期总分先升到 212 再崩到 46，最终策略 14/20 早摔、得分 -17——不是不会走，而是**学成了“冲刺后摔倒”的可被 exploit 的步态**，代理最大化的是速度 proxy，而不是“稳定前进”这个真实目标。

`upright_stability_penalty` 量级太小（ep_sum 仅 -6.5），无法与每步最多 +1.5 的前进奖励抗衡，起不到约束作用。

### signal_completeness（信号完整性）

- 主职责 `forward_velocity_reward` 存在，但**缺少“稳定前进”的语义**：它奖励的是裸速度，没有约束躯干姿态，导致摔倒被误奖。
- `upright_stability_penalty` 存在但力度不足，且与前进奖励解耦。
- `vertical_oscillation_penalty` 是死组件（阈值 0.6 从未触发），本轮不动它，避免范围蔓延。
- 缺失的核心信号：**“前进必须同时保持直立”的对齐信号**，即把前进奖励与躯干姿态耦合。

### selected_level

**L2**：修改 `forward_velocity_reward` 的数学形态，而不是只调系数。理由：当前最佳 final score（-17）远低于目标（300），且问题本质是主信号的语义错误（裸速度可被摔倒 exploit），需要改变公式结构。虽然 cp90% 曾到 212，但那是训练中段的好趋势，最终策略已崩，不是“接近目标只差微调”的情形。

### selected_intervention

目标组件：`forward_velocity_reward`。

具体修改：在原有饱和速度奖励上，乘以一个基于 `next_obs[0]`（hull_angle）的**直立门控**：

```
upright_gate = exp(-(hull_angle / 0.5) ** 2)
forward_velocity_reward = 1.5 * (1 - exp(-max(vx,0) * 1.5)) * upright_gate
```

语义从“奖励水平速度”变为“**只在躯干保持直立时奖励水平速度**”。躯干越倾斜（接近摔倒），前进奖励越趋近 0，因此“向前摔倒”不再能套取前进奖励。保留原 `upright_stability_penalty` 与死组件 `vertical_oscillation_penalty` 不动。

### falsifiable_hypothesis（可证伪假设）

如果 iter_1 的崩坏根因是“摔倒时的水平速度被当作前进奖励 exploit”，那么把前进奖励与直立姿态耦合后，冲刺-摔倒路径的净收益会大幅下降，代理会被迫在“速度”与“稳定”之间权衡。预期下一轮：`early_terminal` 早摔占比明显下降，final score 应回到并超过 cp90% 的 212，且不会再出现 cp100% 式崩盘；`forward_velocity_reward` 在稳定 episode 中仍会较高，但在摔倒 episode 中会显著降低。

### expected_next_round（下一轮预期行为）

策略应学成**保持躯干直立的中高速步态**，而不是冲刺后摔倒；早摔 episode 数下降，episode 长度更一致，final score 与训练后期 checkpoint 分数之间的崩盘收窄。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干保持直立时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：躯干越倾斜，前进奖励越低；接近摔倒时趋近 0。
    hull_angle = float(next_obs[0])
    upright_gate = 2.718281828 ** (-(hull_angle / 0.5) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：只惩罚过大的躯干倾斜和躯干角速度。
    angle_margin = max(0.0, abs(hull_angle) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -2.0 * angle_margin ** 2
        - 0.15 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：轻微惩罚超过死区的垂直速度（本轮保持不动）。
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
