现在我有了完整的图景。让我整理诊断并设计 v4。

---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_1 (best)** | score=-2.98 final eval, **cp100%=122.7** checkpoint; len=813; soft_landing_bonus 在 cp80% 爆到 256.5 — 被 bounce exploit；proximity_progress=30.0 系数主导 (82.9% magnitude) |
| **iter_2** | score=-125.32; len=68; terminated=20/20; speed_penalty 占 46.3% magnitude — agent 学会快速坠毁来最小化惩罚 |
| **iter_3** | score=-114.09; len=68; terminated=20/20; 所有组件 checkpoint 趋势**完全平坦**：approach_reward 从 3.340→3.354（几乎不变），score 从 -124→-114（几乎不学习） |
| **iter_3 逐组件** | approach_reward ep_sum=3.35 意味着净接近距离仅 3.35/3.0≈**1.12 单位**（68步几乎原地不动）；altitude_bonus 全程 0.44（死组件）；landing_reward active_rate 从 4.1% **降到** 3.1%（agent 在学会避开着陆信号） |
| **iter_1 vs iter_3 对比** | iter_1 的 proximity_progress=30.0 产生了明显学习信号（cp30% score 就从 -4 跳到 104）；iter_3 的 approach_reward=3.0 太弱，奖励被负向组件淹没 |

### behavior_diagnosis

**当前策略学到的是"尽快坠毁"**：
- 20/20 terminated，平均 68 步终止 — agent 没有在尝试着陆，而是在快速触发终止条件（crash 或飞出视口）
- 每步净奖励 ≈ (3.35 - 1.26 - 1.02 + 0.86 + 0.44 - 0.20 + 0.54) / 68 ≈ **+0.04/步**，几乎为零——agent 收不到有意义的梯度信号
- approach_reward 系数 3.0 太弱：iter_1 用 30.0 时 agent 学会了着陆（cp60-90 分数 104-165），但 iter_3 用 3.0 时 agent 连"向 pad 靠近"都学不会
- speed_penalty 虽然做了 near_ground 条件化，但仍占 magnitude 的 16.2%，作为负向信号压制了本已微弱的正向信号
- altitude_bonus（0.02 系数）完全无效——所有 checkpoint 值 0.44，是死组件
- landing_reward 的 active_rate 在训练中**下降**（4.1%→3.1%），说明 agent 在主动避开接触（因为接触时伴随 crash 惩罚）

**根因**：自 iter_2 起，奖励函数引入了过多惩罚项（speed_penalty、step_penalty、fuel_penalty），使得每步净奖励趋近于零甚至为负。agent 无法在 68 步的短生存窗口内发现"向 pad 靠近→软着陆"这条正收益路径。iter_1 证明 agent 能学会着陆——只需回归那个框架并修复 exploit。

### intervention

**策略：回归 iter_1 的 potential-based shaping 主干，修复 landing exploit，移除死组件。**

核心改变：
1. **approach_reward 从 3.0 → 12.0**：恢复强梯度，参考 iter_1 的 30.0 但更温和，避免 magnitude 垄断
2. **移除 altitude_bonus**：死组件，全程平坦无贡献
3. **speed 改为「近地慢速奖励 + 近地超速轻罚」双轨**：正向引导优于纯惩罚，且只在近地处生效
4. **landing_reward 改为 new-contact-based**：`max(0, curr_contact - prev_contact)`，只在腿**首次**接触时给奖励，杜绝 iter_1 的 bounce exploit
5. **新增 centering_penalty**：轻量水平偏离惩罚，防止 agent 漂出视口
6. **fuel_penalty 降到 -0.02**：几乎不压制探索
7. **无 step_penalty**：iter_2 证明这只会加速坠毁

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v4 — return to potential-based shaping backbone (iter_1 proven path),
    with exploit-proof landing reward and dead-component cleanup.

    Core changes from v3:
    - approach_reward: 3.0 → 12.0 — strong gradient that iter_1 proved works.
    - REMOVED altitude_bonus — dead component, flat 0.44 across all checkpoints.
    - Speed handling: replaced pure penalty with slow_bonus (reward low speed
      near ground) + mild speed_penalty (only near ground). Net-positive framing.
    - landing_reward: now NEW-CONTACT-BASED. Uses max(0, curr-prev) so that
      sustained contact gives zero additional reward — prevents the bounce
      exploit that caused iter_1's soft_landing_bonus to spike to 256.5.
    - ADDED centering_penalty: light horizontal-deviation cost to prevent
      the agent drifting out of the viewport.
    - Fuel penalty: 0.05 → 0.02 — minimal disincentive, just enough to
      prefer action=0 when hovering.
    - No step_penalty — iter_2 showed this accelerates crashing.
    """

    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad centre
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. approach_reward ─────────────────────────────────────────
    # Potential-based shaping: reward closing Euclidean distance to pad.
    # Coefficient 12.0 is a middle ground: iter_1 used 30.0 (worked but
    # dominated 83% of magnitude), iter_3 used 3.0 (too weak, agent
    # closed only ~1.1 units of distance over the whole episode).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ───────────────────────────────────────
    # Explicit reward for downward progress.  Supplements approach_reward
    # so that even if horizontal drift dilutes the distance signal,
    # descending is always rewarded.
    descent_incentive = 0.5 * max(0.0, oy - y)

    # ── 3. centering_penalty ───────────────────────────────────────
    # Light quadratic penalty for horizontal deviation from pad centre.
    # Prevents the agent from drifting sideways out of the viewport
    # (a failure mode seen in iter_2/3 where episodes terminate early).
    centering_penalty = -0.08 * (x ** 2)

    # ── 4. speed guidance (dual: slow bonus + excess penalty) ──────
    # Positive framing: reward LOW speed near the ground (where it
    # matters for safe landing).  Mild penalty for excess speed near
    # ground as a backstop.  Both decay exponentially with altitude
    # so high-altitude fast descent is unpenalised.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-2.5 * y)   # 1.0 at y=0, ~0.08 at y=1, ~0.007 at y=2
    slow_bonus = 0.15 * ground_factor / (1.0 + speed)
    speed_penalty = -0.06 * speed * ground_factor

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Moderate coefficients —
    # strong enough to matter, not so strong the agent fears using
    # orientation engines entirely.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 6. landing_reward (new-contact-based) ──────────────────────
    # KEY FIX: reward only NEW contact events, not sustained contact.
    # This prevents the bounce exploit where the agent repeatedly
    # taps the pad to farm contact bonuses (iter_1 soft_landing_bonus
    # reached 256.5 at cp80% due to this).
    #
    # new_contact ∈ {0, 1, 2}: how many legs *just now* made contact.
    # Once both legs are down, subsequent steps give ZERO landing reward.
    prev_contact = obs[6] + obs[7]              # 0.0, 1.0, or 2.0
    curr_contact = left_contact + right_contact  # 0.0, 1.0, or 2.0
    new_contact = max(0.0, curr_contact - prev_contact)

    # Quality multiplier: higher bonus for soft, upright, centred contact
    contact_quality = 1.0 / (
        (1.0 + 5.0 * speed)
        * (1.0 + 3.0 * (angle ** 2) ** 0.5)
        * (1.0 + 2.0 * (x ** 2) ** 0.5)
    )
    landing_reward = 8.0 * new_contact * contact_quality

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Minimal penalty — just enough to break ties in favour of action=0
    # when other signals are neutral.  -0.02 is 6x smaller than iter_2's
    # -0.25, so engines can be used freely for braking and control.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + centering_penalty
        + slow_bonus
        + speed_penalty
        + attitude_penalty
        + landing_reward
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "descent_incentive": descent_incentive,
        "centering_penalty": centering_penalty,
        "slow_bonus":        slow_bonus,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "landing_reward":    landing_reward,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components
```