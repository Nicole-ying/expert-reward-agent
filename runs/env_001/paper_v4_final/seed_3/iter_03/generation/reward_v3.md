# 设计理由
agent 快速失败（len=68.7，均提前终止），说明当前奖励信号不足以引导其学会稳定飞行并着陆。上一轮引入的角度门控虽然方向正确，但 **landing 组件处于僵尸状态**（active_rate=0.7%），因为它要求 **双腿同时触碰** 才能获得奖励。任务成功需要的着陆姿态——平稳落下并稳定在平台上——完全没有任何有效引导信号，agent 从未获得足够的着陆反馈，只能依赖 progress 和惩罚盲目探索，导致过早摔出或偏离。

本轮只修改一个组件：**将 landing_bonus 中的接触条件从二值 `min(leg, leg)` 改成连续 bounded factor `(left+right)/2`**。这样，单腿接地时因子为 0.5，双腿接地时为 1.0，中间无塌缩。其他垂直速度、姿态条件保持不变，仍保持 bounded。系数 3.0 不变，但实际生效范围扩大，使 agent 有机会在腿部触地（哪怕仅单腿）时获得正向梯度，进而学会稳定落地并延长 episode。该修改属于 **Level 2 结构变换**（二值→连续化），符合 active_rate < 5% 的标准处理方式。

## 系数校准
- 主信号 progress_gated per-step ≈ 0.154；着陆奖励在活跃后 per-step 可控制在 0.05~0.1 范围（若 active_rate 升至 10%，每步奖励约 0.05），不会压倒 progress。
- 无新增惩罚，不进一步抑制探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)

    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target decreasing ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new

    w_progress = 10.0

    # ---------- 2. Attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Angular velocity penalty: small auxiliary smoothing ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. Landing bonus: soft continuous proxy ----------
    # Modified: use mean contact instead of min, so single-leg touch gives partial reward
    contact_level = (n_left + n_right) / 2.0            # 0.0, 0.5, or 1.0
    vertical_ok = max(0.0, 1.0 - abs(nvy) / 0.3)       # 1.0 when vy≈0
    attitude_ok = max(0.0, 1.0 - abs_angle / 0.15)     # 1.0 when angle≈0

    landing_factor = contact_level * vertical_ok * attitude_ok
    landing_bonus = 3.0 * landing_factor                # up to 3.0, now feasible with single leg

    # Combine: progress is gated by attitude, then penalties and bonus added
    total_reward = (w_progress * progress * angle_gate
                    + lateral_drift
                    + angvel_penalty
                    + landing_bonus)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_drift_penalty": lateral_drift,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号覆盖齐全，但 landing_bonus 因双腿接触的二值条件几乎全程不触发（active_rate 0.7%），成为僵尸组件，缺乏有效着陆引导。
- **behavior**: agent 快速失败，平均 length=68.7，全部提前终止，未接近着陆状态。
- **signal**: 缺失连续的着陆接触信号，单腿接地时完全无奖励，无法引导循序渐进的降落行为。
- **level**: Level 2
- **hypothesis**: 将接触条件从 `min` 改为均值，使单腿触地也能获得部分奖励，提供连续梯度；agent 将开始尝试保持触地并逐步实现双腿着地，从而延长 episode 并提高最终得分。
- **risk**: 若单腿接地奖励过强，可能诱发单腿滑行等不稳定姿态；当前系数 3.0 及因子 0.5 时单腿贡献约 1.5，与 progress 量级可比但不会喧宾夺主，风险可控。