# Duplicate reward retry
The previous generation duplicated iter 9 (runs/env_001/rpv4_old_v4_notools/seed_2/iter_09/generation/reward_v9.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable landing event bonus (replaces per-step contact bonus)
    # Check if any leg is touching
    leg_prev = obs[6] > 0.5 or obs[7] > 0.5
    leg_next = next_obs[6] > 0.5 or next_obs[7] > 0.5

    vel_prev = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    vel_next = (vx ** 2 + vy ** 2) ** 0.5

    angle_prev = abs(obs[4])
    angle_next = abs(next_obs[4])

    # Stable condition: touching + low speed + upright
    prev_stable = leg_prev and vel_prev < 0.2 and angle_prev < 0.1
    next_stable = leg_next and vel_next < 0.2 and angle_next < 0.1

    stable_landing_bonus = 10.0 if (next_stable and not prev_stable) else 0.0

    # --- stable contact reward: encourage maintaining stable contact
    contact_active = next_obs[6] + next_obs[7]
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.5  # increased to balance total reward

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_landing_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_landing_bonus": stable_landing_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components
```

下面是环境描述：
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

目标得分：200

### 第一原则：黑盒实验，不是代码重构

你无法预知新奖励函数会引导出什么行为，只能训练完后看结果。
**看到好趋势 → 先理解为什么好，再精准微调。**
大规模重写等于丢弃已有实验证据，重新掷骰子。

**硬约束：如果本轮得分比上一轮改善超过 30 分，你必须先回答：**
上一轮的哪个修改促成了这个改善？证据是什么？
**如果你回答不了，本轮只做 L1，不改数学形态。**

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

### 本轮数据

# 上一轮训练反馈 (iter_9, score=128.161049)
# Training Feedback

## Final-policy outcome
score=128.161049, len=841.600000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[24.893681, 242.229440]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_bonus | 494.000000 | 71.4% | 71.4% | 5.9% |
| stable_contact_bonus | 182.325000 | 26.3% | 26.3% | 43.3% |
| progress | 10.235151 | 1.5% | 1.8% | 99.2% |
| orientation_penalty | -2.245467 | -0.3% | 0.3% | 100.0% |
| velocity_penalty | -0.959756 | -0.1% | 0.1% | 99.1% |

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
| -38.6 | -2278.1 | -260.4 | -45.8 | -12.1 | 15.0 | 19.8 | 123.2 | 152.3 | 121.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| orientation_penalty | -2.140 | -1.895 | -2.220 | -3.413 | -3.901 | -2.737 | -2.582 | -2.623 | -2.661 | -1.995 |
| progress | 5.750 | -138.243 | -9.307 | 6.790 | 6.972 | 8.121 | 8.723 | 9.697 | 9.342 | 10.293 |
| stable_contact_bonus | 28.250 | 0.000 | 0.000 | 60.850 | 119.550 | 182.275 | 188.500 | 155.025 | 161.350 | 155.600 |
| stable_landing_bonus | 6.000 | 0.000 | 0.000 | 103.500 | 305.500 | 645.000 | 679.500 | 482.000 | 438.500 | 522.000 |
| velocity_penalty | -0.965 | -44.986 | -0.490 | -0.822 | -1.163 | -1.195 | -1.251 | -1.181 | -1.237 | -0.955 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.1% | 99.1% | 99.2% |
| stable_contact_bonus | 8.1% | 0.0% | 0.0% | 14.6% | 32.4% | 38.4% | 41.0% | 42.0% | 46.0% | 38.6% |
| stable_landing_bonus | 0.1% | 0.0% | 0.0% | 1.2% | 4.1% | 6.8% | 7.4% | 6.5% | 6.2% | 6.5% |
| velocity_penalty | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.1% | 98.8% | 99.2% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable landing event bonus (replaces per-step contact bonus)
    # Check if any leg is touching
    leg_prev = obs[6] > 0.5 or obs[7] > 0.5
    leg_next = next_obs[6] > 0.5 or next_obs[7] > 0.5

    vel_prev = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    vel_next = (vx ** 2 + vy ** 2) ** 0.5

    angle_prev = abs(obs[4])
    angle_next = abs(next_obs[4])

    # Stable condition: touching + low speed + upright
    prev_stable = leg_prev and vel_prev < 0.2 and angle_prev < 0.1
    next_stable = leg_next and vel_next < 0.2 and angle_next < 0.1

    stable_landing_bonus = 10.0 if (next_stable and not prev_stable) else 0.0

    # --- stable contact reward: encourage maintaining stable contact
    contact_active = next_obs[6] + next_obs[7]
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.5  # increased to balance total reward

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_landing_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_landing_bonus": stable_landing_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components
```

---

以上是本轮数据。现在开始调查，做出一个语义方向上的修改。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。
- `signal_completeness`：必要职责是否完备？缺什么信号？
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修预计改善策略？
- `expected_next_round`：下一轮策略学习到的行为模式预计是什么？

**2. 完整 Python 代码**（```python ... ``` 代码块）。