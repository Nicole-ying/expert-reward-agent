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

# 上一轮训练反馈 (iter_5, score=-120.427683)
# Training Feedback

## Final-policy outcome
score=-120.427683, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.678292, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.396807 | 76.4% | 79.1% | 100.0% |
| attitude_penalty | -1.696977 | -9.7% | 9.7% | 100.0% |
| settled_reward | 0.750000 | 4.3% | 4.3% | 0.4% |
| height_penalty | -0.635954 | -3.6% | 3.6% | 100.0% |
| contact_discovery | 0.510000 | 2.9% | 2.9% | 1.9% |
| fuel_penalty | -0.076500 | -0.4% | 0.4% | 3.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -115.1 | -120.5 | -123.5 | -121.4 | -115.3 | -119.6 | -121.3 | -121.8 | -124.7 | -121.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 13.327 | 13.363 | 13.363 | 13.376 | 13.364 | 13.397 | 13.394 | 13.385 | 13.368 | 13.395 |
| attitude_penalty | -3.551 | -2.243 | -2.306 | -2.156 | -2.101 | -1.903 | -1.914 | -1.771 | -2.093 | -1.703 |
| contact_discovery | 0.510 | 0.510 | 0.510 | 0.495 | 0.525 | 0.510 | 0.510 | 0.495 | 0.495 | 0.510 |
| fuel_penalty | -0.176 | -0.125 | -0.143 | -0.144 | -0.180 | -0.132 | -0.108 | -0.093 | -0.082 | -0.081 |
| height_penalty | -0.638 | -0.636 | -0.636 | -0.636 | -0.636 | -0.636 | -0.636 | -0.636 | -0.636 | -0.636 |
| settled_reward | 0.375 | 0.500 | 0.625 | 0.500 | 0.375 | 0.750 | 0.625 | 0.750 | 0.500 | 0.750 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_discovery | 2.2% | 2.2% | 2.2% | 2.0% | 2.0% | 2.0% | 2.0% | 2.0% | 2.1% | 2.0% |
| fuel_penalty | 8.4% | 6.0% | 6.9% | 7.0% | 8.8% | 6.4% | 5.3% | 4.5% | 4.0% | 4.0% |
| height_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settled_reward | 0.2% | 0.3% | 0.4% | 0.3% | 0.2% | 0.4% | 0.4% | 0.4% | 0.3% | 0.4% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v5 — settled-state reward replaces exploitable transition bonuses.

    CORE INSIGHT from iter_1 & iter_4:
    - Any contact-transition bonus (new_contact-based) gets exploited via bouncing.
    - Any unconditional near-ground low-speed bonus (slow_bonus) encourages hovering.
    - iter_4 cp30% hit score=230.8 with approach shaping alone, then exploits took over.

    STRATEGY:
    - Keep the approach_reward backbone (12.0 coefficient, proven at cp30%).
    - REPLACE landing_reward + slow_bonus + speed_penalty with a single
      **settled_reward**: per-step reward ONLY when the agent is actually in the
      target state (dual contact + low speed + level + centered).
    - This is inherently bounce-proof: bouncing breaks the settled condition,
      so the agent wants to STAY settled, not bounce.
    - REPLACE descent_incentive + centering_penalty (dead-weight components,
      each <2% magnitude share) with a mild height_penalty to discourage hovering.
    - Tiny contact_discovery seed reward to help the agent discover pad contact.
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
    # proven sweet spot: iter_1 (30.0, dominated 83%), iter_3 (3.0, too weak),
    # iter_4 (12.0, worked — score=230 at cp30% before exploits kicked in).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. settled_reward ──────────────────────────────────────────
    # Per-step reward for being in the actual target state.
    # This REPLACES: landing_reward (exploitable transition bonus),
    # slow_bonus (exploitable hovering reward), and speed_penalty.
    #
    # KEY: sustained settled state IS the task goal. Bouncing breaks the
    # condition (loses contact or gains speed), so the agent wants to stay.
    # This is inherently aligned with the external evaluator.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    both_legs_down = (left_contact > 0.5 and right_contact > 0.5)
    is_settled = (
        both_legs_down
        and speed < 0.4           # nearly stationary
        and abs(angle) < 0.25     # roughly level (≈14°)
        and abs(x) < 0.8          # near pad centre
    )
    settled_reward = 2.5 if is_settled else 0.0

    # ── 3. contact_discovery ───────────────────────────────────────
    # TINY seed bonus for any new leg contact. Helps the agent discover
    # that touching the pad is good, without being worth farming:
    # max 0.6 per dual-touchdown vs 2.5 per settled step.
    # Once settled, the agent never breaks contact, so this becomes silent.
    prev_contact = obs[6] + obs[7]
    curr_contact = left_contact + right_contact
    new_contact = max(0.0, curr_contact - prev_contact)
    contact_discovery = 0.3 * new_contact

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Same coefficients as iter_4
    # (proven non-intrusive, ep_sum ~-1.3).
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. fuel_penalty ────────────────────────────────────────────
    # Mild disincentive for engine use. -0.03 is slightly stronger than
    # iter_4's -0.02 to better discourage unnecessary firing (which enables
    # bouncing). Still low enough that braking and attitude control are cheap.
    fuel_penalty = -0.03 if action != 0 else 0.0

    # ── 6. height_penalty ──────────────────────────────────────────
    # Mild linear penalty for altitude. Replaces the dead centering_penalty
    # and descent_incentive. Provides constant downward pressure so the agent
    # doesn't loiter at high altitude. At y=5 this is -0.05/step, negligible
    # against approach_reward's ~0.5-1.0 per step during active descent.
    height_penalty = -0.01 * y

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + settled_reward
        + contact_discovery
        + attitude_penalty
        + fuel_penalty
        + height_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "settled_reward":    settled_reward,
        "contact_discovery": contact_discovery,
        "attitude_penalty":  attitude_penalty,
        "fuel_penalty":      fuel_penalty,
        "height_penalty":    height_penalty,
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
