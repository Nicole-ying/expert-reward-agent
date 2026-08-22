Now I have all the evidence I need. Let me compile the diagnosis.

---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4** | score=88.62 (cp30% 峰值=230.8), len=909, landing_reward ep_sum=927 (85.6% magnitude) — 被 exploit 膨胀了 4.5× 从 cp30% 到 cp100% |
| **iter_4 landing_reward 轨迹** | cp10%=0.5 → cp30%=202 → cp100%=927. 新接触机制 (new_contact * quality) 被 bouncing 利用：接触→弹起→再接触=更多 new_contact 事件 |
| **iter_5** | score=-120.43, len=68.3, terminated=20/20 — 全部提前终止 |
| **iter_5 settled_reward** | ep_sum=0.75, active_rate=0.4% — 几乎是死组件。二元条件 (speed<0.4 & angle<0.25 & x<0.8 & both_legs) 太过严格，agent 几乎从未触发 |
| **iter_5 approach_reward** | ep_sum=13.4, magnitude_share=79.1% — 唯一活跃信号，但它是势能差分的，到目标附近后归零，无法引导软着陆 |
| **iter_5 attitude_penalty 轨迹** | cp10%=-3.55 → cp100%=-1.70，在改善但缺乏着陆引导 |
| **iter_5 height_penalty** | 全程恒定 -0.636，线性惩罚太弱无法驱动下降 |

### behavior_diagnosis

**iter_5 的失败模式是「信号真空」**：settled_reward 的二元条件导致 0.4% 的 active_rate，agent 在整个训练过程中几乎从未体验过着陆奖励。approach_reward（势能差分）是唯一有效的信号——它能把 agent 引向目标附近，但一旦到达 (x≈0, y≈0) 附近，差分归零，agent 失去所有引导。没有软着陆信号，agent 只能随机飘荡直到 crash/越界终止（68 步，20/20 terminated）。

**iter_4 的失败模式是「exploit」**：new_contact 机制让每次「新接触」都给大额奖励（8.0 * quality），agent 学会 bouncing 来反复触发。但 iter_4 在 cp30% 达到 230.8 分的峰值，说明其核心结构（approach_reward + slow_bonus + 接触信号）是有效的——只是 transition-bonus 形式被利用了。

**核心教训**：着陆信号必须是 **(a) 稠密的**（每步都给，不像 settled_reward 那样稀疏），且 **(b) 非跃迁的**（per-step 连续值，不像 new_contact 那样在接触瞬间跳跃），这样才能既提供梯度又不被 bouncing 利用。

### 策略：用连续「着陆质量」替换二元 settled_reward

核心思路：定义一个从 0 到 1 的连续「settled 质量分」，每步都给。它是多个连续因子的乘积：
- `ground_factor`：离地越近越高（指数衰减）
- `speed_factor`：速度越低越高
- `angle_factor`：姿态越平越高
- `x_factor`：越居中越高
- `contact_factor`：接触腿数越多越高（但无接触时也有基础分，保持梯度）

因为这是一个 **per-step 连续值**而非 transition bonus，bouncing 不会产生额外收益（弹起→失去接触→质量分下降）。同时因为它是 **稠密的**（每步都非零），agent 在整个下降过程中都能感受到梯度。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v6 — dense continuous landing-quality reward replaces sparse settled_reward.

    CORE INSIGHT from iter_4 & iter_5:
    - iter_4: landing_reward (transition-based new_contact * quality) was exploitable
      via bouncing — agent farmed 927.0 ep_sum. But the STRUCTURE was right:
      approach_reward (12.0) + slow_bonus (continuous near-ground speed reward)
      achieved score=230.8 at cp30% before exploits took over.
    - iter_5: settled_reward (binary: speed<0.4 & angle<0.25 & x<0.8 & both_legs)
      was a DEAD COMPONENT — 0.4% active_rate, 0.75 ep_sum. Binary thresholds
      created a reward cliff the agent could never discover.

    STRATEGY (v6):
    - Keep approach_reward at 12.0 (proven backbone, iter_4 cp30%).
    - REPLACE binary settled_reward with DENSE, CONTINUOUS soft_landing reward:
      per-step quality = ground_factor * speed_factor * angle_factor * x_factor * contact_factor
      Each factor is a smooth function → gradient everywhere, no dead zones.
      Per-step (not transition-based) → bouncing REDUCES reward, can't be farmed.
    - Keep descent_incentive (explicit vertical progress signal).
    - Keep attitude_penalty, centering_penalty, fuel_penalty as mild regularizers.
    - REMOVE height_penalty (redundant with descent_incentive).
    - REMOVE contact_discovery (tiny seed bonus, made obsolete by dense soft_landing).
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
    # Potential-based Euclidean distance shaping. 12.0 coefficient is the
    # sweet spot from iter_4 (score=230 at cp30% before landing_reward
    # exploitation took over).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. soft_landing (DENSE, CONTINUOUS) ─────────────────────────
    # Per-step reward proportional to how "settled-like" the state is.
    #
    # KEY PROPERTIES:
    # - DENSE: active every step (not 0.4% like settled_reward).
    # - CONTINUOUS: smooth factors, no binary thresholds → gradient everywhere.
    # - PER-STEP: not transition-based → bouncing REDUCES reward because
    #   breaking contact lowers contact_factor immediately.
    #
    # Each factor ∈ [0, 1] or [0.3, 1.0] for contact.  Multiplicative
    # combination means ALL conditions must be partially met for high reward,
    # providing a clear gradient toward the target state.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    abs_angle = abs(angle)
    abs_x = abs(x)

    # Ground proximity: 1.0 at y=0, decays exponentially with height.
    # At y=2: ~0.14, at y=5: ~0.007 → landing reward is negligible at altitude.
    ground_factor = 2.718281828 ** (-y)

    # Speed quality: 1.0 when stationary, ~0.2 at speed=1.0.
    speed_factor = 1.0 / (1.0 + 4.0 * speed)

    # Angle quality: 1.0 when level, ~0.25 at angle=1.0 rad (~57°).
    angle_factor = 1.0 / (1.0 + 3.0 * abs_angle)

    # Horizontal centering: 1.0 when x=0, ~0.33 at x=1.0.
    x_factor = 1.0 / (1.0 + 2.0 * abs_x)

    # Contact factor: scales from 0.3 (no contact) to 1.0 (both legs).
    # The 0.3 floor ensures gradient exists even before first contact.
    contact_factor = 0.3 + 0.7 * (left_contact + right_contact) / 2.0

    # Combined quality: product of all factors.
    # Perfect landing (y=0, speed=0, angle=0, x=0, both legs):
    #   3.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 = 3.0 per step.
    # Hovering near ground without contact (y≈0, speed≈0, no legs):
    #   3.0 * 1.0 * 1.0 * 1.0 * 1.0 * 0.3 = 0.9 per step — modest, won't
    #   incentivize hovering because episode terminates on settle.
    soft_landing = 3.0 * ground_factor * speed_factor * angle_factor * x_factor * contact_factor

    # ── 3. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress. Supplements approach_reward
    # so descending is always locally rewarded, even when horizontal
    # movement dilutes the Euclidean distance signal.
    # Coefficient 0.3 (slightly reduced from iter_4's 0.5 to avoid
    # competing with soft_landing for magnitude share).
    descent_incentive = 0.3 * max(0.0, oy - y)

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Proven coefficients from
    # iter_4/iter_5: ep_sum ~-1.3 to -1.7, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents the
    # agent from drifting sideways out of the viewport (failure mode
    # seen in iter_2/3). Coefficient 0.08 from iter_4, non-intrusive.
    centering_penalty = -0.08 * (x ** 2)

    # ── 6. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is just enough to
    # prefer action=0 when hovering, but cheap enough that braking
    # and attitude control are essentially free.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + soft_landing
        + descent_incentive
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "soft_landing":      soft_landing,
        "descent_incentive": descent_incentive,
        "attitude_penalty":  attitude_penalty,
        "centering_penalty": centering_penalty,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components
```