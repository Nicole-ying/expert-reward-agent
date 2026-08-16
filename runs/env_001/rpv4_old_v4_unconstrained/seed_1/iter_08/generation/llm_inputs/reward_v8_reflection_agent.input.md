下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_7, score=-100.821003)
# Training Feedback

## Final-policy outcome
score=-100.821003, len=71.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-202.913329, 25.948439]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.531166 | 53.8% | 55.9% | 100.0% |
| speed_penalty | -6.092345 | -24.2% | 24.2% | 100.0% |
| attitude_penalty | -2.878651 | -11.5% | 11.5% | 100.0% |
| contact_gain_reward | 0.753144 | 3.0% | 3.0% | 2.2% |
| contact_loss_penalty | -0.500000 | -2.0% | 2.0% | 0.3% |
| descent_incentive | 0.431431 | 1.7% | 1.7% | 94.5% |
| centering_penalty | -0.263810 | -1.0% | 1.0% | 100.0% |
| fuel_penalty | -0.169000 | -0.7% | 0.7% | 11.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估（训练过程中各组件的变化趋势）

每个 checkpoint 在训练到 N% 时用当前模型评估 20 集，记录每个组件的 ep_sum_mean。
**趋势比单点数值更重要：**
- 组件全程为 0 → 死组件，agent 在训练中忽略了它
- 组件从 0 逐渐增长 → 正在引导学习
- 组件骤升 → 可能被 exploit
- 组件骤降 → agent 在学习抑制它

## Overall score trend
| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -111.6 | -91.2 | -109.2 | -99.6 | -101.6 | -97.4 | -92.7 | -88.7 | -90.3 | -106.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 13.403 | 13.324 | 13.363 | 13.352 | 13.384 | 13.331 | 13.309 | 13.303 | 13.385 | 13.361 |
| attitude_penalty | -1.261 | -1.572 | -2.371 | -1.667 | -1.364 | -1.547 | -1.476 | -1.484 | -1.392 | -1.699 |
| centering_penalty | -0.223 | -0.228 | -0.223 | -0.223 | -0.216 | -0.222 | -0.226 | -0.238 | -0.236 | -0.228 |
| contact_gain_reward | 0.870 | 0.542 | 0.459 | 0.535 | 0.801 | 0.682 | 0.541 | 0.668 | 0.679 | 0.711 |
| contact_loss_penalty | -0.200 | -0.100 | -0.300 | -0.100 | 0.000 | -0.100 | -0.200 | -0.100 | -0.100 | -0.100 |
| descent_incentive | 0.430 | 0.430 | 0.430 | 0.430 | 0.430 | 0.430 | 0.429 | 0.430 | 0.431 | 0.430 |
| fuel_penalty | -0.132 | -0.285 | -0.180 | -0.190 | -0.167 | -0.189 | -0.239 | -0.243 | -0.197 | -0.126 |
| speed_penalty | -5.722 | -5.787 | -5.767 | -5.766 | -5.751 | -5.806 | -5.794 | -5.919 | -5.846 | -5.826 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| centering_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_gain_reward | 2.0% | 2.0% | 2.0% | 2.1% | 1.9% | 2.0% | 2.0% | 2.2% | 1.8% | 2.1% |
| contact_loss_penalty | 0.1% | 0.1% | 0.2% | 0.1% | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% |
| descent_incentive | 94.3% | 94.4% | 94.3% | 94.3% | 94.3% | 94.4% | 94.4% | 94.4% | 94.4% | 94.4% |
| fuel_penalty | 9.6% | 20.6% | 13.1% | 13.8% | 12.2% | 13.7% | 17.2% | 17.3% | 14.2% | 9.1% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
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

---

以上是本轮数据。以下信息可通过工具自主查阅：
- `read_memory()` — 全局记忆表，简单记录每一轮迭代iteration的socre，len长度，奖励函数组件组成
- `read_training_feedback(N)` — 任意迭代的详细训练反馈，了解以前的iteration有过什么好的或者坏的行为，可以避免重蹈覆辙
- `read_reward_code(N)` — 任意迭代的奖励函数代码，方便你深入查看比较好的奖励函数的具体设计
- `read_past_reflection(N)` — 任意迭代的诊断和干预记录，可以查看由好到坏或者由坏到好的记录，吸收学习经验
- `get_component_history(name)` — 组件在所有迭代中的演化，能看清楚某一组件的好的或者坏的演化过程
- `read_environment_card()` — 环境的观察空间、动作空间、终止条件，用来不确定的时候理解每个输入的观测和动作空间的维度信息
- `read_checkpoint_trend(N)` — 训练过程中各组件的趋势，可以看到从训练过程中看出组件是否发挥作用，这样就不仅只看最终反馈，更全面

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

现在开始调查，分析后输出诊断和代码。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。

**2. 完整 Python 代码**（```python ... ``` 代码块）。