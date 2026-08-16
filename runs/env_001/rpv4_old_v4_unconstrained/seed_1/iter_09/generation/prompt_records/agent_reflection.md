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

# 上一轮训练反馈 (iter_8, score=-93.264022)
# Training Feedback

## Final-policy outcome
score=-93.264022, len=70.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.385890, -46.322790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.224344 | 67.9% | 70.8% | 100.0% |
| attitude_penalty | -1.836411 | -9.4% | 9.4% | 100.0% |
| speed_penalty | -1.173432 | -6.0% | 6.0% | 100.0% |
| slow_bonus | 0.947411 | 4.9% | 4.9% | 100.0% |
| contact_gain_reward | 0.606460 | 3.1% | 3.1% | 2.3% |
| descent_incentive | 0.429514 | 2.2% | 2.2% | 94.5% |
| centering_penalty | -0.248010 | -1.3% | 1.3% | 100.0% |
| fuel_penalty | -0.245000 | -1.3% | 1.3% | 17.4% |
| contact_loss_penalty | -0.200000 | -1.0% | 1.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
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
| -120.0 | -117.4 | -122.6 | -114.8 | -113.2 | -103.2 | -92.4 | -93.8 | -92.6 | -94.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 13.377 | 13.341 | 13.373 | 13.335 | 13.336 | 13.193 | 13.245 | 13.327 | 13.177 | 13.250 |
| attitude_penalty | -2.019 | -2.152 | -2.078 | -1.990 | -2.119 | -1.848 | -1.635 | -1.801 | -1.585 | -1.923 |
| centering_penalty | -0.220 | -0.224 | -0.216 | -0.224 | -0.225 | -0.247 | -0.243 | -0.227 | -0.259 | -0.241 |
| contact_gain_reward | 0.777 | 0.588 | 0.777 | 0.846 | 0.797 | 0.533 | 0.588 | 0.583 | 0.701 | 0.566 |
| contact_loss_penalty | -0.200 | -0.600 | 0.000 | 0.000 | 0.000 | -0.400 | -0.400 | 0.000 | -0.200 | 0.000 |
| descent_incentive | 0.430 | 0.429 | 0.429 | 0.429 | 0.429 | 0.430 | 0.430 | 0.429 | 0.430 | 0.429 |
| fuel_penalty | -0.213 | -0.125 | -0.081 | -0.128 | -0.124 | -0.199 | -0.226 | -0.205 | -0.267 | -0.222 |
| slow_bonus | 0.840 | 0.838 | 0.826 | 0.840 | 0.841 | 0.915 | 0.951 | 0.918 | 0.973 | 0.916 |
| speed_penalty | -1.149 | -1.152 | -1.143 | -1.152 | -1.149 | -1.187 | -1.171 | -1.156 | -1.182 | -1.170 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| centering_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_gain_reward | 2.0% | 2.2% | 2.2% | 2.0% | 2.3% | 2.2% | 2.3% | 2.4% | 2.3% | 2.1% |
| contact_loss_penalty | 0.1% | 0.2% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.0% |
| descent_incentive | 94.2% | 94.3% | 94.3% | 94.3% | 94.3% | 94.4% | 94.5% | 94.4% | 94.5% | 94.4% |
| fuel_penalty | 15.6% | 9.1% | 5.9% | 9.3% | 9.0% | 14.3% | 16.0% | 14.7% | 18.9% | 15.9% |
| slow_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
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
