# Response Record

# 设计理由

本轮审计发现**信号缺失**：观测维度 `obs[6]`（左腿接触）和 `obs[7]`（右腿接触）未被使用。当前 20/20 episode 全部 crash 终止，agent 从未获得任何“怎样才算成功着陆”的信息。`proximity_delta` 只是驱近目标，`velocity_danger` 和 `orientation_penalty` 虽然充当约束，但量级不足以阻止“冲过去然后坠毁”的策略。**缺少软着陆的正向信号是根因**。

因此，本轮唯一修改是**新增 `soft_landing_bonus` 组件**，利用腿接触标志提供着陆成功引导。当至少一条腿接触地面时，根据接近目标程度、速度大小和车身姿态，给出连续化的奖励乘积。这使 agent 首次获得“减速、放平、轻触”的明确梯度信号。其余组件保持不变，以保持归因单一。

- **数学形式**：`bonus = w_land * contact * dist_factor * yvel_factor * xvel_factor * angle_factor`
  - `contact = max(obs[6], obs[7])`（0 或 1）
  - `dist_factor = 2.718281828 ** (-dist / 0.3)`
  - `yvel_factor = max(0, 1 - abs(y_vel) / 0.3)`
  - `xvel_factor = max(0, 1 - abs(x_vel) / 0.3)`
  - `angle_factor = max(0, 1 - abs(angle) / 0.3)`
- **系数校准**：`w_land = 80`，使得完美着陆单步 bonus ≈ 80。当前主信号 `proximity_delta` 的 episode sum 约 56，80 的单步上限不超过 2× 总和，且仅在着陆步骤触发，不影响巡航阶段。
- **门控不塌缩**：各因子在“不理想但安全”区域（如速度 0.25、角度 0.25）仍能给出 >0.17 的乘积，保证梯度存在。
- **惩罚负担不变**：未新增惩罚，总惩罚负担与上一轮一致。

预期效果：agent 在接近目标时会尝试减速并对准，以触发 `soft_landing_bonus`，从而减少 crash，提升 len 和 score。

风险：如果 agent 仍然从未触发过腿接触（始终在触地前 crash），则该 bonus 无法提供有效梯度，需要后续轮次进一步强化 velocity_danger 或 orientation_penalty。但目前 crash 模式表明 agent 是接触了地面（因为大部分 crash 是坠毁），所以很可能存在腿接触的瞬间。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward — add soft_landing_bonus using leg contact to signal successful touchdown.
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # ── distance to pad (target at 0, 0) ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_land = 80.0            # soft landing bonus weight

    # ── 1. Proximity delta (unchanged) ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger (unchanged) ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty (unchanged) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft landing bonus (NEW) ──
    contact = max(left_contact, right_contact)  # 0.0 or 1.0

    # distance factor: exponential decay as distance to target increases
    dist_factor = 2.718281828 ** (-dist_next / 0.3)

    # velocity and angle factors: linear ramp from 1 at 0 to 0 at threshold 0.3
    yvel_factor = max(0.0, 1.0 - abs(vy_cur) / 0.3)
    xvel_factor = max(0.0, 1.0 - abs(vx_cur) / 0.3)
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.3)

    landing_bonus = w_land * contact * dist_factor * yvel_factor * xvel_factor * angle_factor

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty + landing_bonus

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 腿接触信号（obs[6], obs[7]）完全未使用，缺失软着陆成功信息，当前 crash 模式无正向梯度。
- **behavior**: agent 冲向目标后 crash，len 68 且 score -110，仅 proximity_delta 驱动。
- **signal**: 缺少“怎样算安全着陆”的指导，velocity_danger 和 orientation_penalty 量级不足以制止 crash。
- **level**: Level 2（新增组件，利用未使用的观测信号填补缺口）
- **hypothesis**: 引入 soft_landing_bonus 后，agent 会尝试在接近目标时减速、对齐姿态并触发腿接触，从而获得首个成功信号，逐渐摆脱纯 crash 策略。
- **risk**: 若当前所有 episode 中腿接触均未触发（罕见），bonus 将保持 0，修改无效果；后续需在 velocity_danger 的近地惩罚或 orientation_penalty 中补强。但根据 crash 模式判断，接触很可能发生。
