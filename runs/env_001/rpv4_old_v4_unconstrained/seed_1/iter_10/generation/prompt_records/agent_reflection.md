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

# 上一轮训练反馈 (iter_9, score=234.887724)
# Training Feedback

## Final-policy outcome
score=234.887724, len=453.700000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[150.809826, 282.347116]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| slow_bonus | 259.723753 | 82.9% | 82.9% | 100.0% |
| approach_reward | 16.160866 | 5.2% | 5.3% | 97.8% |
| landing_reward | 15.589665 | 5.0% | 5.0% | 0.4% |
| speed_penalty | -9.792360 | -3.1% | 3.1% | 99.8% |
| fuel_penalty | -6.480000 | -2.1% | 2.1% | 71.4% |
| contact_loss_penalty | -2.800000 | -0.9% | 0.9% | 0.1% |
| attitude_penalty | -1.150950 | -0.4% | 0.4% | 100.0% |
| descent_incentive | 0.712589 | 0.2% | 0.2% | 61.6% |
| centering_penalty | -0.420942 | -0.1% | 0.1% | 100.0% |

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
| -210.4 | -226.0 | -42.5 | -28.0 | -7.1 | -20.6 | -8.2 | 255.1 | 213.8 | 245.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | -1.255 | -5.311 | 10.271 | 11.977 | 14.553 | 14.280 | 15.828 | 16.275 | 16.516 | 16.258 |
| attitude_penalty | -2.716 | -2.561 | -1.696 | -1.135 | -1.151 | -0.993 | -1.301 | -1.014 | -0.926 | -1.077 |
| centering_penalty | -11.558 | -12.109 | -3.579 | -1.823 | -1.139 | -2.799 | -0.714 | -0.557 | -0.567 | -0.419 |
| contact_loss_penalty | 0.000 | 0.000 | 0.000 | 0.000 | -1.200 | 0.000 | 0.000 | -1.200 | -1.200 | -4.000 |
| descent_incentive | 0.232 | 0.063 | 0.498 | 0.570 | 0.677 | 0.679 | 0.721 | 0.712 | 0.712 | 0.713 |
| fuel_penalty | -7.872 | -7.018 | -12.262 | -12.827 | -19.249 | -19.958 | -19.969 | -5.682 | -8.515 | -6.013 |
| landing_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 | 0.000 | 16.607 | 17.269 | 17.247 |
| slow_bonus | 63.947 | 37.290 | 302.610 | 381.746 | 470.646 | 487.630 | 549.752 | 142.065 | 346.588 | 229.318 |
| speed_penalty | -6.239 | -3.837 | -8.784 | -10.470 | -14.238 | -16.025 | -17.379 | -10.004 | -10.435 | -9.797 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 95.9% | 98.5% | 97.4% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% |
| centering_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_loss_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% |
| descent_incentive | 61.3% | 34.2% | 73.4% | 69.3% | 72.1% | 73.6% | 74.3% | 88.4% | 64.3% | 66.2% |
| fuel_penalty | 62.1% | 62.8% | 61.3% | 64.1% | 99.1% | 99.8% | 99.8% | 86.3% | 71.7% | 72.2% |
| landing_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.6% | 0.3% | 0.4% |
| slow_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.9% | 99.8% |

# 上一轮奖励函数代码
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
