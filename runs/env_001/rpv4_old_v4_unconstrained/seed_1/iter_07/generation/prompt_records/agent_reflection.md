# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件。
- `read_checkpoint_trend(iteration)` — 查看某轮训练中各组件随训练的演化轨迹。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。


```

## User Prompt

```markdown
下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=-40.995321)
# Training Feedback

## Final-policy outcome
score=-40.995321, len=976.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-86.074082, -2.576422]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 306.664896 | 87.3% | 87.3% | 100.0% |
| fuel_penalty | -19.235000 | -5.5% | 5.5% | 98.5% |
| approach_reward | 12.357417 | 3.5% | 5.1% | 100.0% |
| centering_penalty | -6.084178 | -1.7% | 1.7% | 100.0% |
| attitude_penalty | -1.058117 | -0.3% | 0.3% | 100.0% |
| descent_incentive | 0.414252 | 0.1% | 0.1% | 67.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
| -108.7 | -92.4 | 36.8 | -58.7 | -30.5 | 2.9 | -26.3 | -37.6 | -38.0 | -44.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 13.322 | 7.293 | 13.023 | 11.056 | 8.377 | 9.879 | 7.364 | 7.059 | 8.465 | 12.446 |
| attitude_penalty | -1.796 | -0.966 | -1.409 | -1.979 | -1.850 | -1.911 | -2.196 | -2.564 | -1.422 | -1.233 |
| centering_penalty | -0.223 | -4.038 | -1.080 | -1.001 | -5.001 | -5.082 | -7.211 | -18.879 | -19.172 | -6.338 |
| descent_incentive | 0.429 | 0.373 | 0.440 | 0.435 | 0.434 | 0.437 | 0.451 | 0.464 | 0.438 | 0.427 |
| fuel_penalty | -0.143 | -2.605 | -3.714 | -2.388 | -4.950 | -5.557 | -5.644 | -9.113 | -12.829 | -18.448 |
| soft_landing | 3.847 | 41.500 | 99.618 | 25.178 | 70.596 | 91.227 | 73.315 | 151.391 | 173.883 | 306.764 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 99.1% | 100.0% | 99.6% | 99.4% | 99.4% | 99.5% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| centering_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| descent_incentive | 94.3% | 55.2% | 78.3% | 95.9% | 91.9% | 91.3% | 93.1% | 83.3% | 75.9% | 67.6% |
| fuel_penalty | 10.4% | 56.9% | 63.3% | 58.8% | 59.9% | 70.4% | 72.4% | 66.8% | 84.0% | 96.3% |
| soft_landing | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
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
```
