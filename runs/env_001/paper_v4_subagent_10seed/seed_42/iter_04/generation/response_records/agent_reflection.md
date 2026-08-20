# Response Record

# 设计理由
本轮累积记录显示：iter 3 将 orientation_penalty 系数从 iter 1 的轻微水平（约 -0.001 per‑step）提升到 -0.1 和 -0.05，导致 agent 极度保守，在 1000 步内始终无法接近目标（dist 从未 <0.3），soft_landing 僵尸化。同时 safe_progress 尽管正向但过于微弱，整体只能徘徊存活至超时。

行为诊断指出：agent 不敢倾斜或转动身体，以免触发巨大的角度/角速度惩罚，从而牺牲了向目标靠拢的能力。问题在于 **orientation_penalty 过强**，而非进度信号结构。干预方向：仅降低惩罚系数 10 倍（尺度修复），恢复到与主进度信号匹配的级别，让 agent 敢于朝目标机动，同时保留 safe_progress 的速度门控防止失控（该门控在低速安全区域不影响奖励）。

干预层级为 **Level 1（尺度修复）**，改动单一组件 orientation_penalty。系数选择使其预期 per‑step 惩罚 ≤ 主信号 per‑step 的 0.3 倍，避免再次支配整个奖励。

soft_landing 暂时不动，因为一旦 agent 能靠近目标，当前阈值 0.3 有望被触发，届时再校准。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Scale-fix: orientation_penalty weakened 10× to unbind approach capability.
    Keeps safe_progress and soft_landing unchanged; active-rate drop expected.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ──────────────────────────────────────────────
    dist_prev  = (px0**2 + py0**2) ** 0.5
    dist_next  = (px1**2 + py1**2) ** 0.5
    speed      = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Safe progress (speed-gated advancement) ───────────────────
    raw_progress = dist_prev - dist_next   # positive when approaching
    progress     = max(0.0, raw_progress)

    k_target      = 1.5
    gate_strength = 3.0

    expected_speed = k_target * dist_next
    excess_speed   = max(0.0, speed - expected_speed)
    speed_gate     = 1.0 / (1.0 + gate_strength * excess_speed**2)

    safe_progress  = progress * speed_gate

    # ── 2. Orientation / stability penalties (weakened 10×) ─────────
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (unchanged) ─────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * safe_progress
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing
    )

    components = {
        "safe_progress":       safe_progress,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均已使用，但惩罚过强导致接近行为完全被抑制，soft_landing 僵尸化。
- **behavior**: agent 在 1000 步内缓慢徘徊，从不接近目标 0.3 内，依靠极小进度奖励存活，最终因超时截断。
- **signal**: 角度/角速度惩罚（per‑step ~ -0.00052）小幅压倒主进度信号（~0.00137），剥夺了向目标机动的动力。
- **level**: Level 1
- **hypothesis**: 降低 penalty 系数 10 倍后，agent 将敢于倾斜和转动，从而有效接近目标，触发 soft_landing，并恢复类似 iter 1 的成功模式。
- **risk**: 若角度约束过弱，可能导致身体大幅摆动甚至诱发 crash（类似 iter 2），但速度门控仍会限制 reckless 行为，风险可控。
