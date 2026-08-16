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
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=252.826018)
# Training Feedback

## Final-policy outcome
score=252.826018, len=386.650000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[179.071889, 294.371685]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 50.122683 | 70.0% | 70.0% | 26.7% |
| velocity_management | -13.804043 | -19.3% | 19.3% | 98.3% |
| progress | 4.149977 | 5.8% | 5.9% | 96.3% |
| orientation_penalty | -3.462125 | -4.8% | 4.8% | 100.0% |

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
| -107.9 | -359.9 | 24.7 | 191.1 | -11.0 | 258.0 | 231.4 | 254.5 | 248.4 | 259.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.000 | 0.000 | 4.491 | 84.483 | 0.000 | 19.513 | 67.933 | 48.403 | 128.703 | 29.370 |
| orientation_penalty | -2.304 | -2.880 | -3.898 | -3.990 | -6.155 | -3.700 | -4.021 | -3.156 | -3.193 | -3.180 |
| progress | 0.974 | -4.657 | 3.596 | 4.067 | 3.873 | 4.089 | 4.003 | 4.118 | 4.159 | 4.137 |
| velocity_management | -8.163 | -6.740 | -9.051 | -13.507 | -12.761 | -13.729 | -13.776 | -14.042 | -14.200 | -13.779 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.0% | 0.0% | 1.0% | 25.0% | 0.0% | 13.6% | 36.3% | 38.3% | 55.5% | 18.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 99.6% | 98.2% | 100.0% | 95.2% | 97.3% | 96.6% | 97.7% | 95.6% |
| velocity_management | 100.0% | 100.0% | 99.8% | 99.1% | 100.0% | 97.5% | 98.7% | 98.3% | 98.8% | 97.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Distance to target (origin at 0, 0) ──
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # ── 1. Progress: PRIMARY driver toward target ──
    # potential-based: sum over episode = dist_start - dist_end
    progress = dist_prev - dist_next          # >0 when approaching target
    progress_reward = 3.0 * progress          # was 0.5 → 6x boost

    # ── 2. Velocity management ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    speed = speed_sq ** 0.5

    # 2a. General speed penalty — encourages efficiency, prevents oscillation
    speed_penalty = -0.08 * speed_sq

    # 2b. Hard-landing prevention: penalise downward speed near ground
    #     exponential decay with distance → strong only when close to target
    near_ground = 2.718281828 ** (-3.0 * dist_next)
    descent_penalty = -0.8 * max(0.0, -vy) * near_ground

    velocity_management = speed_penalty + descent_penalty

    # ── 3. Orientation stabilisation ──
    orientation_penalty = -0.15 * abs(next_obs[4]) - 0.05 * (next_obs[5] ** 2)

    # ── 4. Contact guidance: tight, conditional, modest ──
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0 .. 2.0

    # Tight proximity: only within 0.3 units of target (was 5.0!)
    proximity_ok = max(0.0, 1.0 - dist_next / 0.3)

    # Tight speed gate: total speed < 0.2 (was vy-only < 0.5)
    speed_ok = max(0.0, 1.0 - speed / 0.2)

    # Modest coefficient 0.3 × tighter gates → much lower effective magnitude
    contact_guidance = 0.3 * contact_val * proximity_ok * speed_ok

    # ── Combine ──
    total_reward = (
        progress_reward
        + velocity_management
        + orientation_penalty
        + contact_guidance
    )

    components = {
        "progress": progress_reward,
        "velocity_management": velocity_management,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
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
