# Response Record

# 设计理由

## 为什么以前都失败了？
历史记忆显示，所有迭代（除 iter 3 的极短 episode 碰巧获得 -9.56 外）均陷入**慢速苟活**模式：episode 长度恢复至 500-800 步，但实际得分却降至 -184 ~ -779，主信号 `gated_forward` 的每步均值远低于早期。根本原因在于**过度惩罚抑制了前进动机**：侧向惩罚、角速度惩罚、高度惩罚形成持续负反馈，agent 学到的策略是尽量站在原地或缓慢移动以减少惩罚，而非真正完成“快速稳定行走”任务。即使第 5 轮将前进系数放大 10 倍，也未能打破这一平衡，因为惩罚仍然是全局性的、无豁免区的。  

更重要的是，**门控乘算机制 (gated_forward)** 在身体姿态稍有偏差时就切断前进奖励，这使得 agent 更不敢加速——它无法在高速度与高稳定性之间获得连续的梯度信号，只能选择绝对安全（即慢速）。

## 新骨架选了什么算子，和已尝试过的有什么本质不同？
新骨架舍弃了所有门控乘积和全局持续性惩罚，改用**纯粹正向鼓励 + 边缘安全 hinge** 的简洁结构：

- **前进信号**：`dense_state_signal`，线性正奖励 `w * body_x_vel`，无任何门控。agent 每步都能直接从“更快前进”获得线性梯度，不再被姿态偏差阻断。
- **直立信号**：`dense_state_signal`，正整数 `0.5 * body_up_z`。这不是惩罚倾斜，而是用**微小但持续的奖励**暗示“保持竖直是好的”，同时不压制探索。
- **高度约束**：`dense_state_signal` 的 hinge 形式，只在 `body_z` 接近终止边界（0.25 / 0.90）时产生线性惩罚。agent 在安全区内完全自由，不会被持续的高度惩罚逼向某个固定值。

**本质不同**：旧骨架的核心是“要想拿前进奖励，必须先满足安全条件才能乘上 gate”，这导致奖励地形崎岖、梯度断裂。新骨架把前进奖励、直立奖励、高度安全**解耦相加**，形成平滑、可直接攀爬的奖励地形。高度安全仅作为边界保护存在，不再渗透到每一步的常规反馈中。

## 系数校准
- 前进系数 `2.0`：参考 final-policy 中 `gated_forward` 的去 gate 估计，每步约 4.0，保持主导地位。
- 直立系数 `0.5`：当 `body_up_z≈0.9` 时贡献约 0.45，仅为前进奖的 1/9，作为辅助信号，不会引发 agent 静止站立。
- 高度 hinge 系数 `-10.0`：越界时每步惩罚达到主信号的 2-3 倍，足够驱离边界，但因 active_rate 仅在 12% 以内，总惩罚负担远低于主信号的 0.5x。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    quat_x = next_obs[2]
    quat_y = next_obs[3]                     # quaternion components for upright projection
    body_x_vel = next_obs[13]                # forward velocity

    # ---------- 1. upright incentive ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)   # 1 when fully upright, 0 when horizontal
    upright_reward = 0.5 * body_up_z

    # ---------- 2. forward velocity (main progress) ----------
    forward_reward = 2.0 * body_x_vel

    # ---------- 3. height safety hinge ----------
    lower_safe = 0.25
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    height_hinge_penalty = -10.0 * height_penalty

    # ---------- combine ----------
    total_reward = forward_reward + upright_reward + height_hinge_penalty

    components = {
        "forward_reward": forward_reward,
        "upright_reward": upright_reward,
        "height_hinge_penalty": height_hinge_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失对前进速度的直接鼓励，惩罚过度覆盖正常行走范围，高度安全仅在边界触发属于合理覆盖，无需额外前兆信号。
- **behavior**: agent 在过去几轮中表现出典型的“保守苟活”模式，前进缓慢、惩罚主导，episode 不是被高效前进结束而是被截断。
- **signal**: 前进信号被不必要的姿态门控削弱；侧向/角速度惩罚制成全局负压，须移除。
- **level**: Level 3
- **hypothesis**: 解耦前进奖励与姿态门控，将奖励地形变得平滑连续，agent 将敢于加速并自发保持必要的稳定姿态，从而在存活时间提升的同时真正获得高额前进回报。
- **risk**: 直立奖励强度过弱可能不足以压制极端倾斜诱发的早期终止，必要时可在后续迭代中将直立信号升级为 bounded hinge 或微调系数。
