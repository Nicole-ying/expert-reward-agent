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

# 上一轮训练反馈 (iter_4, score=88.619535)
# Training Feedback

## Final-policy outcome
score=88.619535, len=908.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[16.011625, 267.138425]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 922.079418 | 85.6% | 85.6% | 14.1% |
| slow_bonus | 113.704663 | 10.6% | 10.6% | 100.0% |
| approach_reward | 16.397651 | 1.5% | 1.8% | 99.8% |
| fuel_penalty | -16.073000 | -1.5% | 1.5% | 88.4% |
| speed_penalty | -3.566617 | -0.3% | 0.3% | 100.0% |
| attitude_penalty | -1.502817 | -0.1% | 0.1% | 100.0% |
| descent_incentive | 0.828957 | 0.1% | 0.1% | 55.9% |
| centering_penalty | -0.539452 | -0.1% | 0.1% | 100.0% |

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
| -85.2 | -20.9 | 230.8 | 212.4 | 133.1 | 96.2 | 114.2 | 126.9 | 146.4 | 114.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 8.160 | 9.721 | 16.280 | 15.170 | 16.095 | 16.093 | 15.727 | 15.838 | 15.758 | 16.182 |
| attitude_penalty | -0.985 | -1.837 | -1.154 | -1.625 | -1.158 | -1.391 | -1.748 | -1.742 | -1.606 | -1.302 |
| centering_penalty | -2.743 | -8.460 | -2.301 | -1.254 | -0.813 | -0.758 | -1.010 | -0.851 | -0.899 | -0.582 |
| descent_incentive | 0.658 | 0.740 | 0.726 | 0.717 | 0.786 | 0.815 | 0.835 | 0.811 | 0.785 | 0.801 |
| fuel_penalty | -1.580 | -9.840 | -6.967 | -4.262 | -14.617 | -15.756 | -14.914 | -14.586 | -9.047 | -13.405 |
| landing_reward | 0.520 | 3.746 | 202.772 | 100.041 | 410.886 | 632.287 | 716.905 | 598.343 | 659.166 | 926.995 |
| slow_bonus | 4.493 | 58.824 | 47.481 | 26.114 | 99.067 | 113.339 | 105.144 | 94.138 | 84.955 | 101.307 |
| speed_penalty | -1.999 | -3.789 | -2.196 | -1.568 | -2.838 | -3.806 | -3.403 | -3.281 | -2.970 | -3.028 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 99.9% | 97.2% | 97.1% | 99.6% | 99.9% | 99.4% | 99.3% | 99.2% | 99.4% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| centering_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| descent_incentive | 64.3% | 81.8% | 78.4% | 77.7% | 54.8% | 54.0% | 58.9% | 58.4% | 58.3% | 57.3% |
| fuel_penalty | 56.4% | 74.3% | 66.2% | 62.3% | 88.9% | 85.4% | 86.6% | 92.6% | 63.6% | 81.8% |
| landing_reward | 0.5% | 0.3% | 5.7% | 4.4% | 8.0% | 10.1% | 12.7% | 11.2% | 13.6% | 15.3% |
| slow_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 99.8% | 99.8% | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% | 100.0% |

# 上一轮奖励函数代码
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
