# Response Record

# 设计理由

本轮诊断：连续 6 轮预判 ❌，score 从 -18 降至 -61，len 从 411 降至 240 且全部 terminated（摔倒）。现有奖励仅使用 3/24 维观测，严重缺失**倾倒预判信号**。当前门控只能在躯干倾角已经很大之后才衰减奖励，此时 agent 已经处于不可逆的摔倒过程中，缺乏提前遏制旋转的趋势信号。此外，hull_angular_velocity 和 leg_ground_contact 完全未被使用，恰好能提供“正在旋转倾倒”和“双脚已离地（典型空中摔倒特征）”的早期检测。因此，本次修改采用 **Level 2 — 结构变换 (add 新组件)**，新增 `falling_risk_penalty`，利用 `hull_angular_velocity`（角速度）和双腿离地状态提前施加惩罚，让 agent 学到主动抑制旋转并保持至少一脚触地。这是对信号缺口的直接填补，而非继续修补原有门控的系数。同时，遵循惩罚负担校准：目标 per-step ≤ 主信号 per-step 的 0.3 倍。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    hull_angular_vel = next_obs[1]   # body angular velocity (rad/s)  ← 新使用
    horizontal_speed = next_obs[2]   # forward velocity
    vertical_speed = next_obs[3]     # vertical velocity
    leg1_contact = next_obs[12]      # left leg ground contact (0 or 1)
    leg2_contact = next_obs[13]      # right leg ground contact (0 or 1)

    # 主进展信号：水平速度
    progress_raw = horizontal_speed

    # 身体倾角门控（保持原有）
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控（保持原有）
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控（几何平均，保持原有）
    health_gate = ((angle_gate * vertical_gate) + 1e-8) ** 0.5

    # 基础奖励：进度 × 健康
    base_reward = progress_raw * health_gate

    # ── 新增：摔倒风险惩罚 ──
    # (1) 角速度惩罚：超过 1.0 rad/s 的部分线性惩罚，系数 0.05
    angular_vel_penalty = max(0.0, abs(hull_angular_vel) - 1.0) * 0.05

    # (2) 双脚离地惩罚：当两腿均未触地时给予轻度惩罚，系数 0.03
    #     使用连续乘积避免二值突变： (1-leg1)*(1-leg2) 仅在双脚离地时 ≈1
    air_penalty = (1.0 - leg1_contact) * (1.0 - leg2_contact) * 0.03

    falling_risk_penalty = angular_vel_penalty + air_penalty

    total_reward = base_reward - falling_risk_penalty

    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
        'falling_risk_penalty': falling_risk_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失 — 仅使用 3/24 维观测，未使用角速度与触地接触，无法预判摔倒；全部 episode 以摔倒终止但无直接惩罚。
- **behavior**: agent 持续中速前进但最终失控摔倒，len 240，score 持续负增长。
- **signal**: 缺少对旋转倾倒和双脚离地的早期惩罚信号，门控反应滞后。
- **level**: Level 2
- **hypothesis**: 引入角速度与双脚离地惩罚后，agent 将在倾斜角扩大前主动抑制旋转、保持至少一脚触地，从而减少摔倒次数，延长存活，提升真实任务得分。
- **risk**: 惩罚可能迫使 agent 过于保守（降低速度），进而减少 progress_reward 累积；但校准后的惩罚系数极低（合计约 0.02~0.05/step），预计不会压垮前进动机。
