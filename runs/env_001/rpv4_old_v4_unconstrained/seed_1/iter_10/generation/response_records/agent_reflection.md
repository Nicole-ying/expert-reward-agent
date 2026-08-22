# Response Record

现在我对完整情况有了清晰的认识。让我综合所有证据，给出诊断和改进方案。

---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_9 最终反馈** | score=234.89, len=453.7, terminated=15/20, truncated=5/20；slow_bonus=259.7 (**82.9%** magnitude share)，landing_reward=15.6 (5.0%, active_rate仅0.4%) |
| **iter_9 checkpoint 趋势** | slow_bonus 从 cp10%=64 → cp70%=**549.8**（score仍为-8.2，agent在hovering不降落）→ cp80%骤降至142（agent发现landing，score跳至255.1）→ cp100%=229。landing_reward 在cp70%前为0，cp80%=16.6 |
| **iter_4 对照** | landing_reward=922 (85.6%, active_rate=14.1%) — bouncing exploit导致landing_reward膨胀，slow_bonus仅113.7 (10.6%)。iter_9用contact_loss=-8.0成功反制bouncing |
| **iter_8 对照** | slow_bonus=0.947 (4.9%) — 系数0.15太弱，agent从未学会刹车，撞毁 |
| **score 方差** | [150.8, 282.3]，跨度131.5，说明部分episode表现显著更差（可能对应5个truncated） |
| **环境卡片** | 次要目标：节约燃料 + 缩短飞行时间。当前453.7步明显偏长 |

### behavior_diagnosis

当前策略的模式是**"hover-then-land"**：

1. **cp10%-cp70%（score从-210升至-8）**：agent学会了用approach_reward导航靠近地面，然后用slow_bonus在近地处减速，但**迟迟不降落**——slow_bonus在cp70%膨胀到549.8，score却仍为-8.2，说明agent在地面附近悬停收集slow_bonus，速度/燃料/姿态惩罚将其抵消到接近零分。

2. **cp80%（转折点）**：agent偶然发现了landing，landing_reward从0跳到16.6，score从-8.2跳到255.1。slow_bonus从549骤降至142——因为着陆后episode终止，不再能累积slow_bonus。

3. **cp80%-cp100%**：agent逐渐平衡"hover收集slow_bonus"与"最终landing"，slow_bonus回升至229-346。但**25%的episode仍然truncated**（悬停到超时未降落）。

**核心问题**：slow_bonus（82.9%份额）已成为事实上的目标函数——agent最大化它的方式是延长近地悬停时间。这与"尽快着陆+节约燃料"的次要目标直接冲突。slow_bonus应该只是刹车提示，不应成为得分主力。

### 干预方案

| 组件 | 当前 | 修改后 | 理由 |
|---|---|---|---|
| `slow_bonus` | 0.80 × e^(-1.5y) | **0.35 × e^(-2.0y)** | 系数减半，刹车窗口收窄；预计ep_sum从260降至~70，份额从83%降至~30% |
| `speed_penalty` | -0.30 × speed × e^(-1.5y) | **-0.10 × speed × e^(-2.0y)** | 与slow_bonus同比缩放，保持刹车梯度对称 |
| `landing_reward` | 10.0 | **18.0** | 让着陆成为明确的"高价值目标"，引导agent尽快着陆而非悬停 |
| `contact_loss_penalty` | -8.0 | **-14.0** | 保持与landing_reward的比例（14/18≈0.78，与原来8/10=0.80持平），维持反bounce硬度 |
| `step_penalty` | 无 | **-0.02/步** | 新增时间压力，直接惩罚悬停拖延；450步约-9，足以让agent倾向于尽早触发终止 |
| `approach_reward` | 12.0 | **15.0** | 小幅增强以补偿更短的episode，确保导航梯度仍然充足 |
| `fuel_penalty` | -0.02 | **-0.03** | 轻微提升节油压力 |

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v10 — rebalance: slow_bonus from score-source → brake-hint,
    landing_reward promoted to primary objective, +time pressure.

    DIAGNOSIS (iter_9, score=234.89):
    - slow_bonus 82.9% magnitude share (259.7 ep_sum). Checkpoint trend shows
      agent hovering near ground accumulating massive slow_bonus (cp70%=549.8)
      before discovering landing (cp80%: slow_bonus drops to 142, score jumps
      from -8.2 to 255.1). The slow_bonus is no longer a braking hint — it IS
      the objective function, incentivising the agent to delay landing.
    - 5/20 truncated: agent sometimes never lands, just hovers.
    - landing_reward only 5.0% share. Landing should be the clear primary goal.
    - Episode length 453.7 — unnecessarily long given the task demands speed.

    STRATEGY (v10):
    1. HALVE slow_bonus (0.80→0.35) and TIGHTEN ground window (e^(-1.5y)→e^(-2.0y)).
       Reduces slow_bonus to a gradient hint, not a score mine.
    2. BOOST landing_reward (10.0→18.0) as the primary objective.
    3. PROPORTIONAL contact_loss (-8.0→-14.0) to maintain anti-bounce hardness.
    4. NEW step_penalty (-0.02/step): time pressure, directly counters the
       "hover to collect slow_bonus" exploit. Agent is rewarded for landing
       and terminating, not for lingering.
    5. SLIGHT approach_reward boost (12.0→15.0) to compensate for shorter episodes.
    6. SCALE speed_penalty down (-0.30→-0.10) to match slow_bonus reduction.
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
    # Potential-based Euclidean distance shaping.  Boosted 12.0→15.0
    # to compensate for shorter episodes (step_penalty pressure).
    # Stable backbone across all successful iterations.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 15.0 * (old_dist - new_dist)

    # ── 2. slow_bonus (REDUCED + TIGHTENED braking window) ────────
    # Positive reward for low speed near ground.  Coefficient halved
    # (0.80→0.35) and ground window tightened (e^(-1.5y)→e^(-2.0y)).
    #
    # Per-step values (illustrative):
    #   y=2.0, speed=3: 0.35*0.018/4.0 ≈ 0.002  (negligible)
    #   y=1.0, speed=2: 0.35*0.135/3.0 ≈ 0.016  (gentle hint)
    #   y=0.5, speed=1: 0.35*0.368/2.0 ≈ 0.064  (building gradient)
    #   y=0.1, speed=0.5: 0.35*0.819/1.5 ≈ 0.191 (strong, but landing imminent)
    # Cannot explode: max at y=0,speed=0 is 0.35, episode terminates on settle.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-2.0 * y)
    slow_bonus = 0.35 * ground_factor / (1.0 + speed)

    # ── 3. speed_penalty (SCALED DOWN to match slow_bonus) ────────
    # Negative penalty for excess speed near ground.  Reduced from
    # -0.30 to -0.10, proportional to slow_bonus reduction.
    # At y=0.5, speed=1: penalty ≈ -0.10*1*0.368 = -0.037
    # Net brake signal at y=0.5, speed=1: 0.064 - 0.037 = +0.027 (mild reward)
    # Net brake signal at y=0.5, speed=2: 0.043 - 0.074 = -0.031 (mild penalty)
    speed_penalty = -0.10 * speed * ground_factor

    # ── 4. descent_incentive ───────────────────────────────────────
    # Explicit reward for downward progress.  Unchanged from iter_9.
    descent_incentive = 0.5 * max(0.0, oy - y)

    # ── 5. landing_reward + contact_loss_penalty ──────────────────
    # BOOSTED landing reward (10.0→18.0) as the PRIMARY objective.
    # contact_loss scaled proportionally (-8.0→-14.0) to maintain
    # anti-bounce hardness (break-even quality ≈ 0.78, same as before).
    #
    # Bounce economics (per leg):
    #   Quality=0.4: gain=7.2, loss=14.0 → net -6.8  (bounce punished)
    #   Quality=0.6: gain=10.8, loss=14.0 → net -3.2  (still punished)
    #   Quality=0.8: gain=14.4, loss=14.0 → net +0.4  (near-break-even)
    #   Quality=0.9: gain=16.2, loss=14.0 → net +2.2  (very clean landing
    #     could be net-positive, but this requires near-perfect conditions)
    #
    # Proper landing: gain once per leg → episode terminates → no loss.
    # Net heavily positive (18-32 for decent quality, both legs).
    prev_contact_sum = obs[6] + obs[7]
    curr_contact_sum = left_contact + right_contact
    new_contact = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    contact_quality = 1.0 / (
        (1.0 + 3.0 * speed)
        * (1.0 + 1.0 * abs(angle))
        * (1.0 + 1.0 * abs(x))
    )

    landing_reward = 18.0 * new_contact * contact_quality
    contact_loss_penalty = -14.0 * contact_loss

    # ── 6. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Proven coefficients.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 7. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation.
    centering_penalty = -0.08 * (x ** 2)

    # ── 8. fuel_penalty ────────────────────────────────────────────
    # Slightly increased (-0.02→-0.03) for mild efficiency pressure.
    fuel_penalty = -0.03 if action != 0 else 0.0

    # ── 9. step_penalty (NEW) ──────────────────────────────────────
    # Constant per-step penalty to create time pressure.  Directly
    # counteracts the "hover near ground to farm slow_bonus" exploit
    # by making every extra step costly.
    # At -0.02/step: 300-step episode costs -6, 450-step costs -9.
    # The agent is incentivised to land and terminate sooner rather
    # than linger.  This penalty is small enough that it won't cause
    # the agent to crash (crashing at step 70 costs only -1.4, far
    # outweighed by lost slow_bonus + landing_reward).
    step_penalty = -0.02

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + slow_bonus
        + speed_penalty
        + descent_incentive
        + landing_reward
        + contact_loss_penalty
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
        + step_penalty
    )

    components = {
        "approach_reward":      approach_reward,
        "slow_bonus":           slow_bonus,
        "speed_penalty":        speed_penalty,
        "descent_incentive":    descent_incentive,
        "landing_reward":       landing_reward,
        "contact_loss_penalty": contact_loss_penalty,
        "attitude_penalty":     attitude_penalty,
        "centering_penalty":    centering_penalty,
        "fuel_penalty":         fuel_penalty,
        "step_penalty":         step_penalty,
    }

    return float(total_reward), components
```
