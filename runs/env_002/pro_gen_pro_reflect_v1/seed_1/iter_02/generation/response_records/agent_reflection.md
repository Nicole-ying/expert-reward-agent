# Response Record

# 设计理由

**审计与诊断**：
- 信号覆盖审计：缺少动作能量惩罚（任务目标明确要求最小化能量消耗），stability_penalty 贡献微乎其微（episode_sum_mean=-0.09，per-step约-0.0001），远不足以形成有效梯度。
- 行为诊断：agent 能稳定行走（len=911.2，terminated=0/20），但效率尚未最优（score=294.82 vs target=300）。步态可能存在不必要的晃动，而当前稳定性惩罚太弱无法抑制。
- 干预目标：增强稳定性约束，让 agent 学习更节能、更稳定的步态。先解决现有组件的尺度问题，暂不引入动作惩罚（避免同时改动多个组件）。
- 干预层级：Level 1 尺度修复。stability_penalty 系数 0.05 太小，相对于 velocity_gated 的 per-step 主信号（~0.544）几乎可以忽略。将系数提升到 0.5，理论 per-step 约 -0.001（仍 << 主信号的 0.3x），但比当前提升 10 倍，足以形成有效梯度引导。

**系数校准**：
- velocity_gated per-step ≈ 495.5/911.2 ≈ 0.544
- 新 stability per-step ≈ -0.5 × (hull_ang_vel≈0.05)² ≈ -0.00125（主信号的 0.23%）
- 门控保持不变：danger_angle=0.8, max_angle=1.2，线性衰减。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：增强角速度惩罚，抑制剧烈晃动 ----------
    w_ang_vel = 0.5  # 从 0.05 提升到 0.5，增强 10 倍
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少动作能量惩罚，但优先修复现有僵尸组件——stability_penalty 贡献近乎为零（-0.09/911.2步）。
- **behavior**: agent 能稳定行走但可能存在不必要的晃动，因为稳定性信号太弱无法引导优化。
- **signal**: stability_penalty 系数 0.05 过弱，per-step 贡献仅主信号的 0.02%。
- **level**: Level 1
- **hypothesis**: 提升稳定性惩罚系数 10 倍（0.05→0.5）将引导策略减少无效晃动，步态更平滑，从而提高行走效率，接近目标 300 分。系数仍在安全范围（per-step < 主信号的 0.3x）。
- **risk**: 极低。新系数 per-step 仅为主信号的 0.23%，不会过度抑制探索。若提升后 len 下降，则表明过度惩罚了必要的平衡调整，需回退。
