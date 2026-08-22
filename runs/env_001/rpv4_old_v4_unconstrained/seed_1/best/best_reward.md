Now I have a comprehensive picture. Let me compile the diagnosis.

---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4 (best, score=88.62)** | len=909, landing_reward ep_sum=922 (85.6% mag), slow_bonus=113.7 (10.6%); cp30% peak score=230.8 → 被 bouncing exploit 破坏；但核心结构 **slow_bonus + strong landing_reward 确实教会了着陆** |
| **iter_6 (score=-41)** | len=976.6, soft_landing ep_sum=306.7 (87.3% mag); dense per-step 奖励导致 **hovering exploit**，agent 悬浮不降落 |
| **iter_7 (score=-100.82)** | len=71.5, 删除了 slow_bonus，contact_gain 仅 2.2% active, ep_sum=0.75 — **contact 信号从未被发现**，18/20 early terminal |
| **iter_8 (score=-93.26)** | len=70.4, 恢复了 slow_bonus(0.15) 但保留 iter_7 的 contact_gain(3.0)/loss(-4.0)；slow_bonus ep_sum=0.95（iter_4 的 1/120），contact_gain 仍仅 2.3% active — **刹车信号太弱，contact 仍未被发现，17/20 early terminal** |
| **组件历史对比** | slow_bonus: iter_4=113.7 vs iter_8=0.95（120× 差距因 episode 长度 909 vs 70）；contact_gain: iter_7=0.75, iter_8=0.61 — 两轮均未被发现 |
| **iter_8 checkpoint 趋势** | 所有组件几乎**完全平坦**：approach_reward 13.2→13.3, slow_bonus 0.84→0.92, speed_penalty -1.15→-1.17。agent **没有在学习**，只是重复同样的 crash 轨迹 |

### behavior_diagnosis

**当前策略行为：高速俯冲 → 撞击地面 → 立即终止。** 

根因链条：
1. `approach_reward=12.0` 是唯一显著的每步信号（~0.19/step），推动 agent 快速下降
2. `slow_bonus=0.15` 基数太小，在 y=0.5 处仅 0.014/step — 完全无法与 approach_reward 竞争，刹车梯度不存在
3. `contact_gain_reward=3.0` 太小 + `contact_loss_penalty=-4.0` 超过最大可能 gain → **尝试接触的期望净值为负**，agent 学会回避接触
4. 结果：agent 被 approach_reward 驱动冲向地面，没有刹车信号，没有接触激励 → crash → 短 episode → slow_bonus 永远无法积累 → 死循环

**iter_4 为什么成功**：`landing_reward=8.0` 系数足够大（是 iter_8 的 2.7×），且没有 loss penalty → agent 敢尝试接触；`slow_bonus` 在 909 步长 episode 中有充分时间积累。虽然最终被 bouncing exploit，但 cp30% 时 score=230.8 证明了核心结构正确。

**v9 策略**：恢复 iter_4 的核心强度 + 加固反 exploit 机制：
- **landing_reward 提升到 10.0**（iter_4 是 8.0，iter_8 是 3.0）— 确保 contact 可被发现
- **slow_bonus 从 0.15 提升到 0.8** — 使刹车信号能与 approach_reward 竞争
- **ground_factor 从 e^(-2.5y) 放宽到 e^(-1.5y)** — 刹车信号在更高海拔激活，给 agent 更长的减速窗口
- **speed_penalty 从 -0.06 提升到 -0.3** — 作为刹车 backstop
- **contact_loss_penalty=-8.0** — 使 bouncing 始终净负（单腿: gain≤10.0, loss=8.0, 质量<0.8 时净负），但 quality gate 放宽（speed_factor=3.0, angle_factor=1.0, x_factor=1.0）使优质着陆奖励丰厚
- **descent_incentive 恢复到 iter_4 的 0.5**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v9 — restore iter_4 landing strength + hardened anti-bounce + amplified brakes.

    DIAGNOSIS (iter_8 failure, score=-93.26):
    - slow_bonus=0.15 too weak: ~0.014/step near ground vs approach ~0.19/step.
      Agent never learned to brake, crashed at step ~70, slow_bonus never accumulated.
    - contact_gain=3.0 + loss=-4.0: expected value of attempting contact is NEGATIVE
      for any quality < 0.75. Agent learned to AVOID contact. Active rate flatlined at 2.3%.
    - All checkpoint trends FLAT: agent repeated the same crash trajectory every episode.

    STRATEGY (v9):
    1. AMPLIFY slow_bonus: 0.15 → 0.8 base, widen ground_factor from e^(-2.5y) → e^(-1.5y).
       This gives the agent a longer braking window (activates at y≈2 instead of y≈1)
       and makes the brake signal competitive with approach_reward near the ground.
    2. RESTORE strong landing_reward: 10.0 * new_contact * quality (iter_4 used 8.0).
       Large enough to be discovered even at moderate quality.
    3. HARDEN anti-bounce: loss penalty -8.0/leg ensures ANY bounce is net-negative
       given realistic quality values (quality < 0.8 for any imperfect landing).
       Proper landing: gain once → sustain → episode terminates → net positive.
    4. RELAX quality gate: speed_factor=3.0 (was 4.0), angle_factor=1.0 (was 1.5),
       x_factor=1.0 (was 1.0). Wider tolerance so first successful contacts give
       meaningful gradient.
    5. STRONGER speed_penalty: -0.3 (was -0.06), proportional to the amplified
       slow_bonus as a backstop against excess speed near ground.
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
    # Potential-based Euclidean distance shaping. 12.0 is the proven
    # backbone across all iterations. Stable, never exploited.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. slow_bonus (AMPLIFIED) ──────────────────────────────────
    # Positive reward for low speed near the ground.  THE critical brake
    # signal.  Amplified 5.3× from iter_8 (0.15 → 0.80), and ground_factor
    # widened from e^(-2.5y) → e^(-1.5y) so the gradient activates earlier
    # (meaningful at y≈2 instead of y≈1), giving agent time to decelerate.
    #
    # Per-step values (illustrative):
    #   y=2.0, speed=3: 0.80*0.050/4.0 = 0.010  (just activating)
    #   y=1.0, speed=2: 0.80*0.223/3.0 = 0.059  (gradient building)
    #   y=0.5, speed=1: 0.80*0.472/2.0 = 0.189  (strong incentive to slow)
    #   y=0.1, speed=0.5: 0.80*0.861/1.5 = 0.459 (dominant signal)
    # Cannot explode: max at y=0,speed=0 is 0.80, and episode terminates on settle.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-1.5 * y)
    slow_bonus = 0.80 * ground_factor / (1.0 + speed)

    # ── 3. speed_penalty (AMPLIFIED backstop) ──────────────────────
    # Negative penalty for excess speed near ground.  Amplified 5× from
    # iter_8 (-0.06 → -0.30), proportional to slow_bonus amplification.
    # At y=1.0, speed=2: penalty ≈ -0.30*2*0.223 = -0.134.
    # Net brake signal at y=1.0, speed=2: 0.059 - 0.134 = -0.075 (penalty).
    # Net brake signal at y=0.5, speed=1: 0.189 - 0.142 = +0.047 (reward).
    # Gradient: fast near ground → penalised; slow near ground → rewarded.
    speed_penalty = -0.30 * speed * ground_factor

    # ── 4. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress.  Back to iter_4 level (0.5,
    # was 0.3 in iter_7/8).  Supplements approach_reward so descending is
    # always locally rewarded even when horizontal drift dilutes the
    # Euclidean distance signal.
    descent_incentive = 0.5 * max(0.0, oy - y)

    # ── 5. landing_reward + contact_loss_penalty ──────────────────
    # RESTORED strong landing reward (10.0, iter_4 used 8.0) with
    # HARDENED anti-bounce loss penalty (-8.0 per leg).
    #
    # Bounce economics (per leg, quality-gated gain, flat loss):
    #   Quality=0.4 (speed≈0.5, angle≈0.2, x≈0.3): gain=4.0, loss=8.0 → net -4.0
    #   Quality=0.7 (speed≈0.15, angle≈0.05, x≈0.1): gain=7.0, loss=8.0 → net -1.0
    #   ANY realistic quality < 0.8 → bouncing is NET NEGATIVE.
    #
    # Proper landing economics:
    #   0→1 leg: gain 10.0*q1, 1→2 legs: gain 10.0*q2, then episode terminates.
    #   No loss fires.  Net strongly positive (10-20 for decent quality).
    #
    # Quality gate RELAXED vs iter_8: speed_factor=3.0 (was 4.0),
    # angle_factor=1.0 (was 1.5).  Wider tolerance → first contacts give
    # meaningful gradient → agent can discover the landing signal.
    prev_contact_sum = obs[6] + obs[7]               # 0.0, 1.0, or 2.0
    curr_contact_sum = left_contact + right_contact   # 0.0, 1.0, or 2.0
    new_contact = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    contact_quality = 1.0 / (
        (1.0 + 3.0 * speed)
        * (1.0 + 1.0 * abs(angle))
        * (1.0 + 1.0 * abs(x))
    )

    landing_reward = 10.0 * new_contact * contact_quality
    contact_loss_penalty = -8.0 * contact_loss  # per leg, flat (no quality gate on loss)

    # ── 6. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Proven coefficients across
    # iter_4/6/7/8: ep_sum ~-1.0 to -2.0, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 7. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents
    # drifting out of viewport (failure mode in iter_2/3).
    centering_penalty = -0.08 * (x ** 2)

    # ── 8. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is cheap enough that
    # braking and attitude control are essentially free, but the agent
    # prefers action=0 when other signals are neutral.
    fuel_penalty = -0.02 if action != 0 else 0.0

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
    }

    return float(total_reward), components
```