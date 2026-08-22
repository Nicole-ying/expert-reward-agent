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
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_9, score=185.785275)
# Training Feedback

## Final-policy outcome
score=185.785275, len=680.100000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[77.743736, 260.996126]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 289.491832 | 56.5% | 56.5% | 100.0% |
| contact_reward | 204.195000 | 39.9% | 39.9% | 54.9% |
| action_reward | -13.443000 | -2.6% | 2.6% | 72.9% |
| progress_reward | 2.547018 | 0.5% | 0.5% | 99.2% |
| vel_damping | -1.430503 | -0.3% | 0.3% | 19.5% |
| orientation_penalty | -0.809572 | -0.2% | 0.2% | 100.0% |

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
| -160.3 | -162.3 | -54.1 | -29.2 | -21.2 | -21.4 | 155.6 | 220.9 | 187.1 | 200.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -11.859 | -15.723 | -28.365 | -27.378 | -31.488 | -31.487 | -20.700 | -10.842 | -12.964 | -14.034 |
| contact_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 241.905 | 43.320 | 42.930 | 110.370 |
| orientation_penalty | -1.411 | -0.884 | -0.408 | -0.465 | -0.470 | -0.544 | -0.560 | -0.755 | -0.952 | -0.716 |
| progress_reward | 0.308 | 0.260 | 1.638 | 2.070 | 2.325 | 2.366 | 2.698 | 2.613 | 2.485 | 2.481 |
| proximity_reward | 60.641 | 65.665 | 305.844 | 352.679 | 387.748 | 415.391 | 375.514 | 153.533 | 177.074 | 235.017 |
| vel_damping | -0.900 | -0.263 | -0.211 | -0.297 | -0.265 | -0.512 | -1.826 | -1.694 | -0.878 | -1.564 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 63.0% | 61.8% | 84.3% | 79.2% | 100.0% | 99.9% | 100.0% | 84.4% | 88.9% | 88.8% |
| contact_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 61.4% | 18.8% | 15.9% | 43.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 96.9% | 97.9% | 99.4% |
| proximity_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_damping | 46.0% | 19.5% | 6.1% | 8.0% | 7.1% | 9.7% | 16.5% | 33.9% | 27.4% | 22.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: Continuous proximity replaces discrete landing bonus.

    ROOT CAUSE OF v4-v8 FAILURES:
    Every iteration with a discrete "landing bonus" (per-step or one-time) was exploited:
    - v4: agent sat on pad collecting 50/step → landing_bonus=2627, 97.4% magnitude
    - v8: agent bounced to re-trigger one-time bonus → landing_bonus=922, 94.7% magnitude
    Discrete events are inherently exploitable — the agent always finds a way to re-trigger them.

    v9 APPROACH:
    - NO discrete landing bonus. NO touchdown detection. NO "just_touched_down" triggers.
    - proximity_reward: continuous Gaussian centered on pad (0,0). Always active, smooth
      gradient, max 0.5/step at center. Cannot be exploited: bouncing AWAY reduces reward.
    - On-pad: proximity + contact ≈ 1.0/step. Far away: ≈ 0.05/step. 20:1 natural gradient.
    - All components are continuous → no exploit cliff, no collapse-recovery cycle.
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (navigation signal, proven in all iterations) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Proximity: continuous Gaussian pull toward pad center ──
    # Replaces the exploitable discrete landing bonus.
    # max 0.5 at (0,0), decays smoothly: 0.37 at dist=0.5, 0.14 at dist=1.0, ~0 at dist=2.0
    # Always active → agent always feels gradient toward pad. Bouncing reduces reward.
    proximity_reward = 0.5 * (2.718281828 ** (-dist ** 2 / 0.8))

    # ── 3. Orientation penalty: stable, always active, proven scales ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping near pad: prevent hard landings ──
    near_pad = 2.718281828 ** (-dist ** 2 / 0.5)
    safe_speed = 0.25
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * near_pad * excess_speed

    # ── 5. Contact: reward ground interaction for discoverability ──
    contact_reward = 0.3 * (left_contact + right_contact)

    # ── 6. Action efficiency: gentle penalty, glide is free ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
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
