# Response Record

# 设计理由

本轮为正常模式（Normal Mode），干预一个组件：**stability_penalty**。

**证据链条**：
- 审计（§0）：稳定性惩罚 `active_rate=100%` 但 `episode_sum_mean=-0.18`，**几乎零贡献**。它在安全区域产生无意义的小惩罚，在危险区域又无法有效警示。
- 行为（§1）：agent 得分 276 分已经接近首轮 best（294），步长恢复 901，说明门控和动作惩罚的组合大体可行。它没有爆炸，而是**卡在高位瓶颈**——稳定性惩罚没有给到正确的边界信号。
- 累积记录（§3）：Iter 2 尝试放大稳定性惩罚系数（0.05→0.5），直接导致 score 下降 51 分；Iter 3 加动作惩罚，score 崩到 28。**盲目加大稳定性惩罚是有害的**，真正有用的是在接近摔倒边界时提供梯度，而非全时惩罚。
- 数学形态问题（§6）：当前 `-0.05 * hull_ang_vel^2` 是**无界二次惩罚**，每步都罚，但实际角速度很小 → 惩罚几乎为零，无法在关键时刻预警摔倒。
- 目标算子：**dense_state_signal (hinge)**——设在终止边界的 60-80% 处，安全性区域无需惩罚，风险区域给出线性增长且有界的警示。

**设计校准**：
- 主信号 `velocity_gated` 的 per-step ≈ 458.33/901 ≈ 0.51。
- 新惩罚 per-step 目标 ≤ 0.3 × 0.51 ≈ 0.15。
- 阈值：观察角速度值范围，设安全阈值 2.0（约 60-70% 危险边界），超出部分线性惩罚。
- 系数 0.05：在阈值+1.0（即 3.0）时 penalty ≈ -0.05 × 1.0 = -0.05 per-step，在阈值+3.0（即 5.0）时 ≈ -0.15 per-step。量级合理，不会支配主信号。
- 总惩罚负担：动作效率 per-step ≈ -18.82/901 ≈ -0.021，稳定性新惩罚平均 < 0.05 → 合计 < 0.08，远小于主信号 0.51 的 0.5 倍。

**改动内容**：将 stability_penalty 从全时二次转换为 **hinge 形态**：`-0.05 * max(0, |hull_ang_vel| - 2.0)`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度

    # ---------- 健康门控 ----------
    danger_angle = 0.8
    max_angle = 1.2
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：hinge 形态 ----------
    # 安全阈值 2.0：在此范围内不计入惩罚
    # 超出部分线性惩罚，为接近摔倒的高角速度提供梯度
    stability_threshold = 2.0
    excess_ang_vel = max(0.0, abs(hull_ang_vel_next) - stability_threshold)
    stability_penalty = -0.05 * excess_ang_vel

    # ---------- 动作效率 ----------
    w_action = 0.02
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 缺少有效边界风险预警信号——稳定性惩罚全时施加但几乎零贡献，无法在接近摔倒时提供梯度。
- **behavior**: agent 在安全区域稳定行走，但瓶颈期需要边界预警来突破；它不是乱走，而是过于安全但不够快。
- **signal**: 稳定性惩罚需要从无界二次改为**有界 hinge**，在危险区域给出线性梯度而非在安全区域浪费信号。
- **level**: Level 2
- **hypothesis**: hinge 形态在安全区域不惩罚 → 不干扰正常探索；在危险区域线性惩罚 → 提供清晰梯度警示，引导策略远离倾倒边界，从而释放更多速度探索空间，突破当前 276 分瓶颈。
- **risk**: 阈值 2.0 若偏高可能导致惩罚过晚触发，无效；若偏低可能恢复过度约束。当前 2.0 是基于常见角速度范围估算，最坏情况需要根据下一轮反馈微调阈值。
