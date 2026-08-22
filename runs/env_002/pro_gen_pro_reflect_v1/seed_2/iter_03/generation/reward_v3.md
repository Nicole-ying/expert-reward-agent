# 设计理由
**问题诊断**：第 2 轮将主奖励从 `forward_vel` 改为 `forward_vel·|forward_vel|`（凸化），意图增强高速目标的梯度——但实际导致 len 从 930 → 490、score 从 290 → 88，终局 terminated=20/20。凸化对低速 (<1.0) 的梯度被压缩（斜率 2·|v| → 低速更弱），而对高速产生过强激励，agent 学会“冲刺即摔倒”的 exploit。

**修改方案（Level 1 尺度修复）**：
- **恢复线性前进奖励**：`forward_reward = 1.0 * forward_vel`，保持正向梯度均衡。观察 iter1 的骨架能稳定到 290 分，说明线性形态是有效的。
- **放松稳定性惩罚**：当前 hull_angle 系数 0.5 过约束——iter1 中稳定性 penalty per-step 约 -0.011 已足够，但配合凸化高速后可能过度抑制 recovery。降到 0.3，保留有效约束但不打断探索。
- **强化能量惩罚**：当前系数 0.001 使 penalty per-step 仅 -0.002（占 0.2%），几乎无效。提高到 0.01，使 per-step ≈ -0.02（占主奖励的 8%），但仍然远低于 30% 上限。

**校准**：
- 主信号 per-step = 0.29（iter1）→ 稳定性 penalty per-step ≈ 0.01（OK）、能量 penalty per-step ≈ 0.02（OK）
- 总惩罚负担 < 0.05 ≪ 0.5 × 0.29，不会压制探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：线性前进速度奖励（恢复） ======
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel  # 回到 iter1 的线性形式

    # ====== 姿态稳定性惩罚（系数放松） ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]
    w_angle = 0.3        # 原 0.5 → 0.3，避免过罚小幅摆动
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚（适度加强） ======
    w_energy = 0.01       # 原 0.001 → 0.01，使能耗有感知力
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
- **audit**: 缺前兆信号（摔倒无软惩罚→由稳定性惩罚承担）、缺步态接触利用，但核心问题是主信号形态畸形；当前三个职责组件均已存在。
- **behavior**: agent 被凸化前进奖励推向“高速冲刺→摔倒”的 exploit，长度减半、分数暴跌。
- **signal**: 前进速度凸化（`v·|v|`）在低速区梯度压缩、高速区过激励，是 iter2 失败的根因。
- **level**: Level 1
- **hypothesis**: 恢复 iter1 的线性前进奖励骨架，放松稳定性约束以容忍高速 recovery 摆动，轻度提高能耗惩罚使其可感知——应使 len 回升至 700+，score 回到 200+。
- **risk**: 能量惩罚略增可能略微降低平均速度，但在 290 分时 eps 已有盈余，per-step 能量负担 < 10% 可控。