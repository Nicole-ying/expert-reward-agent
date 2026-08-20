# Response Record

# 设计理由
正常模式下只修改一个组件：将 `proximity_reward` 的数学形式从 `1.0 / (1.0 + dist_next)` 改为 `1.0 / (1.0 + dist_next**2)`。

**为什么改这个？**
- 第0步审计：所有观测已被使用，无信号缺失；soft_landing 完全未触发（active_rate=0%），说明 agent 从未到达距离 < 0.3 的目标邻域。当前策略停留在 avg dist≈0.39，靠 proximity_reward 每步约 0.72 稳定获取收益，陷入局部最优。
- `1/(1+dist)` 在 0.3–0.4 区间梯度平缓（约 -0.52），不足以驱动 agent 进一步冒险接近目标。
- 替换为 `1/(1+dist^2)` 后，同一位置的梯度变为约 -1.18，近区奖励提升更明显（0.39 → 0.869），形成更强的“拉向中心”的力，从而有望将 agent 拉入 `dist<0.3` 区域内，使 soft_landing 被激活并开始提供精细的着陆引导。
- 该改动属于 Level 2 结构变换：将平缓的线性分式替换为凸化的二次分式，增加信号区分度，符合 formula guide 的“凸化”建议。

**系数量级**：保持系数 1.0，当前平均每步 `1/(1+0.39²)≈0.869`，仅比原来 +8%，总和仍在安全范围内，且无惩罚上涨，不会造成训练崩溃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Proximity reward (convexified) + attitude penalty + soft landing gate.
    Changed proximity_reward to 1/(1+dist^2) to strengthen gradient near target.
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

    # ── 1. Proximity reward (convexified) ───────────────────────────
    #  1/(1+dist) → 1/(1+dist^2) to increase gradient at ~0.3–0.4.
    proximity_reward = 1.0 / (1.0 + dist_next**2)

    # ── 2. Attitude penalty ──────────────────────────────────────────
    angle_penalty  = -0.003 * (angle1 ** 2)
    angvel_penalty = -0.001 * (angvel1 ** 2)
    attitude_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance ─────────────────────────────────────
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
- **audit**: 信号齐全但校准不足——soft_landing 死区是因为 agent 从未进入 dist<0.3，而非观测缺失。
- **behavior**: agent 在 avg dist 0.39 处徘徊，从平缓 proximity 获取足够回报，没有动力接近和着陆。
- **signal**: proximity_reward 在 0.3–0.4 区间梯度过小，缺乏足够驱动力拉向目标；soft_landing 完全闲置。
- **level**: Level 2
- **hypothesis**: 凸化 proximity 函数能产生更强向心梯度，将 agent 拉入 soft_landing 活跃区域，进而引导稳定着陆。
- **risk**: agent 在初期若距离较大（>1）时奖励偏低，可能放慢探索；但 proximity 仍然有界且 soft_landing 一旦激活将迅速提供高额正向引导，总体风险可控。
