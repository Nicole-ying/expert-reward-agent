# 设计理由
这是第一轮迭代，当前骨架已基本接近目标（得分 290.58，达率 96.86%），但搜索目标要求达到 300 分。从训练反馈看，所有回合都能成功终止（20/20），速度中等级（len 930 表示步数充足），未出现早期摔倒。分析 agent 行为：存在中速巡航现象（forward_reward episode_sum_mean / len ≈ 0.54 m/s），线性奖励 `w * forward_vel` 对小幅速度提升的梯度激励不足，不能让 agent 突破至更高速度。

我实施 **Level 2 结构变换**：将前向速度奖励从 **线性形式** 切换为 **平方（保持方向）形式**，即使用 `forward_vel * |forward_vel|` 替代原有的 `forward_vel`。这是公式切换指南中的 **dense_state_signal 凸化**：当信号正值但偏小、停滞在低水平时，通过非线性（如平方）增加对高值的边际激励，同时保持方向性（后退仍为负）。这样，速度从 0.4 提升到 0.6 的增量奖励比线性大 50%，从而鼓励 agent 探索更高的冲刺速度。稳定性惩罚和能耗惩罚保持不变，但在新 forward 系数下调整权重以保持数值平衡。

校准：原 forward_reward episode_sum_mean = 504.3 / 930.4 step ≈ 0.542 per-step。 使用平方后，期望维持相近量级：设系数为 2.0，当速度为 0.54 时，奖励 ≈ 2 * 0.54^2 = 0.583，略高于原值；当速度提升到 0.7 时，奖励 = 2 * 0.49 = 0.98，边际激励增强。总惩罚负担不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：前进速度奖励（平方形式，凸化激励） ======
    # 使用 forward_vel * |forward_vel| 保持方向，同时平方增强高速激励
    forward_vel = next_obs[2]
    w_forward = 2.0
    forward_reward = w_forward * forward_vel * abs(forward_vel)

    # ====== 竖直/姿态稳定性惩罚 ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]
    w_angle = 0.5
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚 ======
    w_energy = 0.001
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 覆盖了前进驱动、姿态和能耗，但缺乏非线性激励和摔倒前兆门控；前进信号使用线性形态，可能陷入中速巡航。
- **behavior**: agent 已学会稳定行走（全部终止），但维持在约 0.54 m/s 的中等速度，未充分发挥快速前进的潜力。
- **signal**: 前进速度的线性奖励对速度提升的边际激励不足，需要凸化（平方）以鼓励探索更高速度。
- **level**: Level 2 — 结构变换（线性正奖励 → 平方正奖励）
- **hypothesis**: 凸化后的前进奖励将为高速行为提供更强的梯度激励，使 agent 突破当前中速瓶颈，向更高分数冲刺。
- **risk**: 凸化可能导致低速阶段的奖励更低，但 agent 已学会避免摔倒，不会显著恶化稳定性；若后期出现摔倒增加，可后续引入 gate 机制。