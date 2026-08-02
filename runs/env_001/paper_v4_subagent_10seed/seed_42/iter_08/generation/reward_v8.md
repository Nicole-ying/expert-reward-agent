# 设计理由

**修改组件**：`stability_penalty` → `attitude_penalty`（移除速度惩罚，仅保留姿态惩罚）。

**问题诊断**：第 0 步审计显示所有观测信号均被使用，但 `soft_landing` 活跃度为 0%（阈值 `dist < 0.3` 从未被触发），说明 agent 一直没有接近目标。Iter 7 引入的 `velocity_penalty = -0.08 * speed / (1+dist)` 虽然在远处很小，但在接近目标时权重急剧增大，反而抵消了 `proximity_reward` 的吸引力，导致 agent 学会在安全距离徘徊以规避惩罚，最终所有 episode 均超时（len=1000）且无任何成功着陆迹象。Iter 6 的配置（proximity_reward + 极小的 angle/angvel 惩罚 + soft_landing）取得了 194.63 分，证实移除速度约束后 agent 能够接近并触发 soft_landing。

**变换类型**：Level 2 — 结构变换（删除妨碍主导信号的惩罚项）。将 gated speed penalty 从组件中完全移除，回归到只对姿态施加极小惩罚的简单形式。

**系数校准**：姿态惩罚系数沿用 iter 6 的 `-0.003 * angle²` 和 `-0.001 * angvel²`，per-step 量级约 −0.003，远小于 proximity_reward per-step 的 0.747（仅约 0.4%），不会破坏前进动力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Proximity reward + attitude-only penalty + soft landing gate.
    Removed velocity penalty that was preventing the agent from getting close
    enough to trigger soft_landing.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px1, py1 = next_obs[0], next_obs[1]  # position
    vx1, vy1 = next_obs[2], next_obs[3]  # velocity (unused now, kept for clarity)
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_next = (px1**2 + py1**2) ** 0.5
    speed = (vx1**2 + vy1**2) ** 0.5    # only used in soft_landing now

    # ── 1. Proximity reward ──────────────────────────────────────────
    proximity_reward = 1.0 / (1.0 + dist_next)

    # ── 2. Attitude penalty (replaces stability_penalty) ─────────────
    # Only penalize tilt and spinning — no speed penalty.
    angle_penalty  = -0.003 * (angle1 ** 2)
    angvel_penalty = -0.001 * (angvel1 ** 2)
    attitude_penalty = angle_penalty + angvel_penalty

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
        1.0 * proximity_reward
        + 1.0 * attitude_penalty
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward":   proximity_reward,
        "attitude_penalty":   attitude_penalty,
        "soft_landing":       soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，但 `soft_landing` 为僵尸组件（dist 从未 < 0.3），`velocity_penalty` 的 gated 形式在近处抑制接近。
- **behavior**: agent 存活 1000 步但始终未进入着陆区域，在远处通过 `proximity_reward` 获取平稳回报但不敢靠近。
- **signal**: 速度惩罚在接近目标时迅速增大，抵消了 proximity 的吸引力。
- **level**: Level 2（移除妨碍主导信号的结构性惩罚项）
- **hypothesis**: 移除速度约束后，agent 将像 iter 6 一样敢于接近目标，触发 `soft_landing` 并恢复 190+ 分数。
- **risk**: 着陆阶段可能速度偏高导致不稳定，但 `soft_landing` 内的 `speed_factor` 仍会在接近时鼓励减速。