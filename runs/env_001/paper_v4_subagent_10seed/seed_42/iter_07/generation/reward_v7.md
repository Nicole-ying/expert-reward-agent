# 设计理由
**修改组件**：将几乎无关紧要的 `orientation_penalty` 替换为 `stability_penalty`，在保留原有体态角惩罚的同时增加**接近加权速度惩罚**。  
**为什么**：agent 已能高效接近原点（19/20 终止，推测为成功着陆），但距离目标分差 5.37，主要是缺少对**接近目标时减速**的引导——当前只有在 `dist < 0.3` 的少数步内才通过 `soft_landing` 间接惩罚高速，导致进入软着陆窗口时速度仍偏高，未能充分累积 soft_landing 奖励。  
**数学形式**：`speed * 1/(1 + dist)`，远处 penalty 极小，近处自然增大，要求减速；体态角罚保持不变，系数微调以降低扰动。  
**系数校准**：  
- 主信号 `proximity_reward` 的 per‑step 均值 ≈ 0.75  
- 速度惩罚系数 0.08，最大可能 per‑step ≈ −0.08（speed≈1, dist≈0），远小于 0.75×0.3 = 0.225，安全。  
- 角罚系数略微降低（0.005 → 0.003, 0.002 → 0.001），防止叠加后过大。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Proximity reward + stability penalty (speed * nearness + orientation)
    + soft landing gate.  Speed penalty is gated by 1/(1+dist) to encourage
    deceleration only when close to the target.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px1, py1 = next_obs[0], next_obs[1]  # position
    vx1, vy1 = next_obs[2], next_obs[3]  # velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_next = (px1**2 + py1**2) ** 0.5
    speed = (vx1**2 + vy1**2) ** 0.5
    nearness = 1.0 / (1.0 + dist_next)          # ∈ (0,1], 1 when at origin

    # ── 1. Proximity reward (unchanged) ─────────────────────────────
    proximity_reward = 1.0 / (1.0 + dist_next)

    # ── 2. Stability penalty (replaces orientation_penalty) ──────────
    # Speed penalty gated by proximity: punish speed only when near.
    velocity_penalty = -0.08 * speed * nearness

    # Small orientation penalties to keep the craft upright.
    angle_penalty  = -0.003 * (angle1 ** 2)
    angvel_penalty = -0.001 * (angvel1 ** 2)

    stability_penalty = velocity_penalty + angle_penalty + angvel_penalty

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
        + 1.0 * stability_penalty
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward":    proximity_reward,
        "stability_penalty":   stability_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均已使用；19/20 终止推断为成功着陆，信号齐全，问题在于靠近目标时的速度校准不足。  
- **behavior**: agent 在 proximity 驱动下快速接近原点，但进入软着陆区域时速度仍然较高，导致软着陆步数少、累加奖励低。  
- **signal**: 缺少“越近越需减速”的连续引导，使得 agent 无法自主压低近区速度来充分收割 soft_landing。  
- **level**: Level 2 – 组件结构变换（姿态罚 → 姿态+速度联合罚）。  
- **hypothesis**: `speed * 1/(1+dist)` 惩罚将在近距离自然压低速度，延长 soft_landing 有效窗口，从而提升 soft_landing 累积，并促成更平稳的终止，总分有望突破 200。  
- **risk**: 速度惩罚可能使 agent 在接近原点时过度保守（减速过猛），轻微拉长 episode；但 proximity_reward 的持续吸引力会抑制这一倾向，总体偏向正面。