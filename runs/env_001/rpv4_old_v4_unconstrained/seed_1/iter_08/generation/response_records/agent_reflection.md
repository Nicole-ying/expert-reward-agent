# Response Record

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4 (best, score=88.62)** | cp30% 峰值 score=**230.8**；slow_bonus 100% active，ep_sum 从 cp10%=4.5 增长到 cp100%=101.3；approach_reward 稳定 ~16。landing_reward 从 cp30%=203 膨胀到 cp100%=**927** (15.3% active) — bouncing exploit |
| **iter_7 (当前, score=-100.82)** | 全程负分 (-111~-86)；contact_gain 仅 2.2% active, contact_loss 仅 0.3% active；**early_terminal=18/20, len=71.5**；descent_incentive 94.4% active → agent 在下降但 crash 前从未触垫 |
| **iter_7 vs iter_4 对比** | iter_7 **删除了 slow_bonus**，speed_penalty 放大 5× (-0.3 vs -0.06)；iter_4 的 slow_bonus 是**唯一的近地低速正向激励**，iter_7 只剩纯惩罚 |
| **iter_6 soft_landing** | ep_sum=306.7, 87.3% magnitude → dense per-step 正向奖励被**悬停 exploit** |
| **approach_reward 跨迭代** | 在所有迭代中稳定 (13-16 ep_sum)，从未被 exploit — 是唯一可靠的 backbone |
| **speed_penalty 历史** | iter_2: -11.35 (46% mag, 导致 crash); iter_7: -6.09 (24% mag); iter_4: -3.57 (0.3% mag, 配合 slow_bonus) |

### behavior_diagnosis

**当前策略行为：高速下坠 → 撞击地面 → 提前终止。**

原因链条：
1. `approach_reward` (12.0×距离缩短) + `descent_incentive` (向下奖励) 驱动 agent **加速接近目标垫**
2. iter_7 删除了 iter_4 中的 `slow_bonus`（近地低速正向奖励），**没有任何信号告诉 agent "接近地面时应该减速"**
3. `speed_penalty` 是纯负向（≤0），agent 学到的最优策略是：**快速冲下去 → 尽早 crash → 最小化累积惩罚**（因为 crash 后 episode 终止，不再累积 speed_penalty）
4. 18/20 episodes 在 71.5 步内提前终止，contact 组件 barely active → agent **从未成功触垫**

**根因：缺乏近地减速的正向激励。** iter_4 的 slow_bonus（`+0.15 * ground_factor / (1+speed)`）提供了 "近地处 speed 越低越好" 的梯度，agent 学会了 brake。这是 iter_4 能到达垫子而 iter_7 不能的关键差异。

### 干预策略

1. **恢复 slow_bonus**（来自 iter_4 的成功结构）：近地低速获得正向奖励，教会 agent 减速进场
2. **保留 anti-bounce contact 机制**（gain reward + loss penalty），但调整参数：
   - 降低 contact_gain 系数（3.0，而非 iter_4 的 8.0）防止 exploit 膨胀
   - 提高 contact_loss_penalty（-4.0/leg）确保 bouncing 始终 net negative
   - 适度放宽 contact_quality gate（speed 因子从 5.0→4.0）
3. **speed_penalty 回归 iter_4 水平**（-0.06，而非 -0.3），因为 slow_bonus 承担主要减速引导
4. 保持 approach_reward=12.0 及其他 regularizer 不变

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v8 — restore iter_4 slow_bonus + hardened anti-bounce contact.

    DIAGNOSIS (iter_7 failure):
    - iter_7 removed slow_bonus and relied solely on speed_penalty (negative).
      Agent learned: "rush down → crash early → minimise cumulative penalty."
      18/20 early terminal, contact components barely active (2.2% / 0.3%).
    - iter_4 (best score=88.62, cp30%=230.8) succeeded because slow_bonus
      taught the agent to BRAKE near the ground, enabling gentle contact.

    STRATEGY (v8):
    1. RESTORE slow_bonus from iter_4: positive reward for low speed near ground.
       This is the critical missing gradient — without it, the agent never
       learns to decelerate before impact.
    2. KEEP anti-bounce contact mechanism (gain + loss), with balanced params:
       - contact_gain_reward at 3.0 (lower than iter_4's 8.0 to limit exploit ceiling)
       - contact_loss_penalty at -4.0 per leg (ensures bouncing ALWAYS net negative:
         max 2-leg gain = 3.0*2*1.0 = 6.0, max 2-leg loss = 8.0)
    3. RELAX contact_quality: speed factor 4.0 (was 5.0), angle factor 1.5 (was 2.0)
       → stronger gradient for the agent when it does make contact.
    4. MILD speed_penalty (-0.06, as iter_4) as backstop; slow_bonus does the
       heavy lifting for deceleration guidance.
    5. Keep approach_reward at 12.0 (proven backbone across all iterations).
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
    # Potential-based Euclidean distance shaping. Coefficient 12.0 is the
    # proven sweet spot (iter_4 cp30%=230.8, iter_6, iter_7). Stable, never
    # exploited, provides the backbone gradient toward the pad.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. slow_bonus (RESTORED from iter_4) ──────────────────────
    # POSITIVE reward for low speed near the ground.  This is the critical
    # component that iter_7 deleted — it provides the gradient that teaches
    # the agent to BRAKE before touchdown.
    #
    # Properties:
    # - Max 0.15 per step (at y=0, speed=0).  Cannot explode because
    #   the episode terminates on settle at y≈0, speed≈0.
    # - ground_factor = e^(-2.5y): at y=1.0 → 0.082, at y=2.0 → 0.007.
    #   Effectively zero at altitude, active only in final approach.
    # - Always ≥0: best case is speed=0 → bonus=0.15*gf.  No exploit
    #   possible because the agent can't get more than this.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-2.5 * y)
    slow_bonus = 0.15 * ground_factor / (1.0 + speed)

    # ── 3. speed_penalty (mild backstop, as iter_4) ────────────────
    # Mild negative penalty for speed near ground.  Coefficient -0.06
    # matches iter_4 (where it coexisted successfully with slow_bonus).
    # At speed=2, y=0.5: penalty ≈ -0.06*2*0.287 = -0.034 per step.
    # The slow_bonus is the primary brake signal; this is a backstop.
    speed_penalty = -0.06 * speed * ground_factor

    # ── 4. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress. Supplements approach_reward
    # so descending is always locally rewarded.
    descent_incentive = 0.3 * max(0.0, oy - y)

    # ── 5. contact handling: gain reward + loss penalty ────────────
    # Anti-bounce mechanism: reward NEW contact (gain), penalise LOSING
    # contact (loss).  The loss penalty exceeds the maximum possible gain
    # so that ANY bounce cycle (gain → loss → gain) is net NEGATIVE.
    #
    # Per-leg analysis:
    #   Max gain per leg = 3.0 * 1.0 (quality=1.0) = 3.0
    #   Loss per leg = -4.0
    #   One-leg bounce:  net ≤ 3.0 - 4.0 = -1.0  (always negative)
    #   Two-leg bounce:  net ≤ 6.0 - 8.0 = -2.0  (always negative)
    #
    # Proper landing: gain once → sustain contact → episode terminates.
    # Net = +gain (positive), no loss triggered.
    prev_contact_sum = obs[6] + obs[7]               # 0.0, 1.0, or 2.0
    curr_contact_sum = left_contact + right_contact   # 0.0, 1.0, or 2.0
    contact_gain = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    # Quality gate: rewards only gentle, upright, centred touchdowns.
    # Relaxed vs iter_7: speed factor 4.0 (was 5.0), angle factor 1.5 (was 2.0).
    # This gives a stronger gradient for the agent's first successful contacts.
    contact_quality = 1.0 / (
        (1.0 + 4.0 * speed)
        * (1.0 + 1.5 * abs(angle))
        * (1.0 + abs(x))
    )

    contact_gain_reward = 3.0 * contact_gain * contact_quality
    contact_loss_penalty = -4.0 * contact_loss  # per leg, stronger than max possible gain

    # ── 6. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Proven coefficients across
    # iter_4/5/6/7: ep_sum ~-1.0 to -2.0, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 7. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents
    # drifting out of viewport.
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
        + contact_gain_reward
        + contact_loss_penalty
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":     approach_reward,
        "slow_bonus":          slow_bonus,
        "speed_penalty":       speed_penalty,
        "descent_incentive":   descent_incentive,
        "contact_gain_reward": contact_gain_reward,
        "contact_loss_penalty": contact_loss_penalty,
        "attitude_penalty":    attitude_penalty,
        "centering_penalty":   centering_penalty,
        "fuel_penalty":        fuel_penalty,
    }

    return float(total_reward), components
```
