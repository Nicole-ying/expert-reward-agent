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

# 上一轮训练反馈 (iter_3, score=-114.093212)
# Training Feedback

## Final-policy outcome
score=-114.093212, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.982445, -94.359884]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 3.354443 | 43.0% | 44.6% | 100.0% |
| speed_penalty | -1.264842 | -16.2% | 16.2% | 100.0% |
| attitude_penalty | -1.016799 | -13.0% | 13.0% | 100.0% |
| descent_incentive | 0.860983 | 11.0% | 11.0% | 94.3% |
| landing_reward | 0.536060 | 6.9% | 6.9% | 3.1% |
| altitude_bonus | 0.440928 | 5.7% | 5.7% | 100.0% |
| fuel_penalty | -0.200000 | -2.6% | 2.6% | 5.8% |

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
| -124.0 | -119.3 | -119.7 | -124.9 | -122.1 | -120.2 | -119.9 | -116.0 | -115.9 | -114.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_bonus | 0.446 | 0.441 | 0.442 | 0.440 | 0.440 | 0.439 | 0.439 | 0.441 | 0.441 | 0.441 |
| approach_reward | 3.340 | 3.342 | 3.344 | 3.341 | 3.343 | 3.349 | 3.349 | 3.353 | 3.353 | 3.354 |
| attitude_penalty | -1.409 | -1.365 | -1.384 | -1.209 | -1.222 | -1.126 | -1.101 | -1.017 | -1.014 | -1.015 |
| descent_incentive | 0.857 | 0.858 | 0.857 | 0.858 | 0.859 | 0.860 | 0.859 | 0.861 | 0.861 | 0.861 |
| fuel_penalty | -0.225 | -0.173 | -0.133 | -0.113 | -0.110 | -0.130 | -0.143 | -0.158 | -0.188 | -0.195 |
| landing_reward | 0.452 | 0.466 | 0.475 | 0.488 | 0.501 | 0.524 | 0.484 | 0.540 | 0.552 | 0.523 |
| speed_penalty | -1.276 | -1.262 | -1.255 | -1.265 | -1.267 | -1.259 | -1.259 | -1.269 | -1.266 | -1.265 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| descent_incentive | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% |
| fuel_penalty | 6.6% | 5.0% | 3.9% | 3.3% | 3.2% | 3.8% | 4.2% | 4.6% | 5.5% | 5.7% |
| landing_reward | 4.1% | 3.7% | 3.6% | 3.5% | 3.4% | 3.0% | 2.9% | 3.0% | 3.1% | 3.1% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — fixed net-negative-per-step bug from v2.

    Core changes from v2:
    - speed_penalty is now conditional on near_ground proximity (was global).
      Fast descent at high altitude is efficient, not penalised.
    - Added descent_incentive: explicit per-step reward for moving downward,
      supplementing the potential-based approach_reward.
    - Reduced fuel_penalty from -0.25 to -0.05 so the agent dares to use engines.
    - Removed step_penalty entirely — it incentivised early termination.
    - Boosted landing_reward coefficient to create a strong terminal attractor.
    - All coefficients rebalanced so the approach phase is net-positive.
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
    # Modest coefficient (was 30.0 in v1, 5.0 in v2) — enough to guide,
    # not so large it drowns out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 3.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ───────────────────────────────────────
    # Explicit reward for each unit of downward progress.
    # Supplements approach_reward: even if horizontal drift dilutes the
    # distance signal, descending is always rewarded.
    descent_incentive = 0.6 * max(0.0, oy - y)

    # ── 3. altitude_bonus ──────────────────────────────────────────
    # Small bonus for staying at low altitude.  Creates a gentle
    # "basin" near the pad without being large enough to cause hovering.
    altitude_bonus = 0.02 / (1.0 + 3.0 * (y ** 2) ** 0.5)

    # ── 4. speed_penalty ───────────────────────────────────────────
    # Penalise high speed ONLY when close to the ground, where it risks
    # a crash.  At high altitude, speed is unpenalised — fast efficient
    # descent is actually desirable.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    near_ground = 1.0 / (1.0 + 8.0 * (y ** 2) ** 0.5)   # ~1 at y=0, ~0 when high
    speed_penalty = -0.08 * speed * near_ground

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating throughout the trajectory.
    # Moderate coefficients — strong enough to matter, not so strong
    # that the agent fears using orientation engines.
    attitude_penalty = -0.2 * (angle ** 2) - 0.05 * (angular_vel ** 2)

    # ── 6. landing_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed, upright attitude, and proximity to pad centre.
    # Continuous n_contact (not binary product) gives partial credit
    # for single-leg contact, providing a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    landing_quality = 1.0 / (
        (1.0 + 4.0 * speed)
        * (1.0 + 3.0 * (angle ** 2) ** 0.5)
        * (1.0 + 2.0 * (x ** 2) ** 0.5)
    )
    landing_reward = 0.8 * n_contact * landing_quality

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Small penalty for engine use (actions 1,2,3).
    # Kept intentionally light so the agent is willing to use engines
    # for braking and attitude control.  -0.05 vs v2's -0.25.
    fuel_penalty = -0.05 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + altitude_bonus
        + speed_penalty
        + attitude_penalty
        + landing_reward
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "descent_incentive": descent_incentive,
        "altitude_bonus":    altitude_bonus,
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
