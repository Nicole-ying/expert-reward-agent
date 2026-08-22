# Response Record

现在我已经掌握完整的图景。让我整理最终诊断并设计 v7。

---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4 (best score=88.62)** | cp30% 达到峰值 score=230.8，此时 landing_reward=202.8 (active=5.7%), slow_bonus=47.5, approach=16.3。到 cp100% 时 landing_reward 膨胀到 927.0 (active=15.3%, 4.6× 增长)，score 降至 114.3。**transition-based contact reward 被 bouncing exploit** |
| **iter_6 (当前, score=-41.00)** | soft_landing=306.7 (87.3% magnitude), len=976.7, terminated=1/20。cp30% peak=36.8 → cp100%=-44.4。**per-step dense soft_landing 被 hovering exploit**：agent 在 y≈0 附近悬停，不接触地面就能收取 soft_landing reward |
| **iter_5 (score=-120.43)** | settled_reward 是 binary 阈值型 (speed<0.4 & angle<0.25 & x<0.8 & both_legs)，active_rate=0.4%，ep_sum=0.75。**死组件**：agent 从未发现这个稀疏信号 |
| **iter_2 (score=-125.32)** | speed_penalty 全局施加，46.3% magnitude，episode 全在 68 步内 crash。**全局速度惩罚导致 agent 学会快速坠毁** |
| **所有迭代的共同规律** | (1) approach_reward (potential-based distance shaping, coef=12.0) 是唯一稳定有效的正向信号；(2) 任何 near-target per-step 正向奖励（slow_bonus, soft_landing, 或 transition-based landing_reward）最终都被 exploit；(3) 纯稀疏奖励是死组件 |

### behavior_diagnosis

- **iter_6 行为**：agent 学会了在 pad 上方悬停（y≈0 但不接触），收取 soft_landing=306.7 的 per-step 奖励。episode 极长 (976 步)，仅 1/20 terminated。这是经典的 **reward exploitation**：soft_landing 的 `contact_factor=0.3 + 0.7*(L+R)/2` 使得无接触时也有 0.3 的基础因子，agent 可以不接触地面就获得正向奖励。
- **iter_4 行为**：agent 在 cp30% 时学会了真正的着陆（score=230.8），但随后发现 bouncing（反复接触→离地→再接触）可以 farming transition-based landing_reward。到 cp100% landing_reward=927，score 降到 114.3。
- **核心矛盾**：per-step positive reward near target → hovering exploit；transition-based reward → bouncing exploit；sparse reward → dead component。

### 干预策略

**核心创新：contact_gain_reward + contact_loss_penalty（反 bouncing 机制）**

- 奖励**新获得**的接触（contact_gain），quality-gated，鼓励轻柔着陆
- **惩罚接触丢失**（contact_loss），且惩罚 > 奖励。 bouncing = 获得接触(小奖励) → 丢失接触(大惩罚) = **净负**。正确着陆 = 获得接触(奖励) → 保持接触 → episode terminate = **净正**
- 去掉所有无接触时的 per-step 正向奖励（slow_bonus, soft_landing 的 0.3 floor）
- 用 **speed_penalty（仅近地面）** 替代 slow_bonus：始终 ≤0，悬停时 speed=0 → penalty=0，不可 exploit
- 保留 approach_reward=12.0 和 descent_incentive=0.3 作为骨干

---

## 奖励函数 v7

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v7 — anti-bounce contact mechanism replaces exploitable soft_landing.

    CORE INSIGHT from all 6 iterations:
    - Any per-step POSITIVE reward near the target state gets exploited:
      * iter_4: transition-based landing_reward → bouncing exploit (927 ep_sum)
      * iter_6: dense per-step soft_landing → hovering exploit (306 ep_sum)
      * iter_1: per-step soft_landing_bonus → spike to 256.5
    - Sparse/binary rewards die (iter_5 settled_reward: 0.4% active).
    - approach_reward (potential-based distance shaping) is the only
      consistently stable positive signal across all iterations.

    STRATEGY (v7):
    - Keep approach_reward at 12.0 (proven backbone).
    - REPLACE soft_landing with contact_gain_reward + contact_loss_penalty:
      * Reward NEW contact (gain), quality-gated by speed/angle/x.
      * Penalize LOSING contact (loss) MORE than the gain reward.
      * Net effect: bouncing (gain→loss→gain) is net NEGATIVE.
        Proper landing (gain→sustain→settle) is net POSITIVE.
    - Use speed_penalty (≤0, near ground only) instead of slow_bonus:
      can't be exploited — best case is speed=0 → penalty=0.
    - Remove all per-step positive rewards that don't require contact.
    - Keep mild regularizers (attitude, centering, fuel).
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
    # proven sweet spot across iter_1/4/5/6. Stable, never exploited.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress. Supplements approach_reward
    # so descending is always locally rewarded even when horizontal
    # movement dilutes the Euclidean distance signal.
    descent_incentive = 0.3 * max(0.0, oy - y)

    # ── 3. contact handling: gain reward + loss penalty ────────────
    # KEY INNOVATION: penalize losing contact MORE than rewarding gain.
    # This makes bouncing net-negative while proper landing is net-positive.
    #
    # Bouncing cycle:  gain ~1.0 → lose -2.0 → net negative per bounce.
    # Proper landing:  gain ~1.5 → sustain (no loss) → episode terminates.
    #
    # Contact quality gates ensure only gentle touchdowns get high reward.
    prev_contact_sum = obs[6] + obs[7]          # 0.0, 1.0, or 2.0
    curr_contact_sum = left_contact + right_contact  # 0.0, 1.0, or 2.0
    contact_gain = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Quality: high when speed low, angle small, centered.
    # Speed factor dominates: at speed=2.0 quality drops to ~1/11 ≈ 0.09.
    contact_quality = 1.0 / (
        (1.0 + 5.0 * speed)
        * (1.0 + 2.0 * abs(angle))
        * (1.0 + abs(x))
    )

    contact_gain_reward = 3.0 * contact_gain * contact_quality
    contact_loss_penalty = -2.0 * contact_loss  # per leg lost, independent of quality

    # ── 4. speed_penalty (near ground only) ─────────────────────────
    # Penalize high speed near the ground to encourage braking.
    # Always ≤0: at speed=0 penalty=0 — no positive reward to exploit.
    # Decays exponentially with height: at y=3 ground_factor < 0.001.
    ground_factor = 2.718281828 ** (-2.5 * y)
    speed_penalty = -0.3 * speed * ground_factor

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Proven coefficients from
    # iter_4/5/6: ep_sum ~-1.0 to -1.7, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 6. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents
    # drifting out of viewport (failure mode in iter_2/3).
    centering_penalty = -0.08 * (x ** 2)

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is just enough to
    # prefer action=0 when hovering, but cheap enough that braking
    # and attitude control are essentially free.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + contact_gain_reward
        + contact_loss_penalty
        + speed_penalty
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":     approach_reward,
        "descent_incentive":   descent_incentive,
        "contact_gain_reward": contact_gain_reward,
        "contact_loss_penalty": contact_loss_penalty,
        "speed_penalty":       speed_penalty,
        "attitude_penalty":    attitude_penalty,
        "centering_penalty":   centering_penalty,
        "fuel_penalty":        fuel_penalty,
    }

    return float(total_reward), components
```
