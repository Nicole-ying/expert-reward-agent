# Duplicate reward retry
The previous generation duplicated iter 4 (runs/env_001/rpv4_old_v4_notools/seed_2/iter_04/generation/reward_v4.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward *increase* in leg contact, not the absolute contact state
    prev_contact = float(obs[6] + obs[7])
    next_contact = float(next_obs[6] + next_obs[7])
    contact_improvement = max(0.0, next_contact - prev_contact)   # only positive changes

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_improvement * proximity_factor * velocity_ok

    # combine all components
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
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

# 上一轮训练反馈 (iter_4, score=29.558091)
# Training Feedback

## Final-policy outcome
score=29.558091, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-8.525842, 77.881773]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 66.304464 | 96.6% | 96.6% | 19.1% |
| orientation_penalty | -1.263707 | -1.8% | 1.8% | 100.0% |
| progress | 0.679175 | 1.0% | 1.2% | 100.0% |
| velocity_penalty | -0.245721 | -0.4% | 0.4% | 100.0% |

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
| -88.9 | -37.3 | 71.6 | 55.6 | 66.2 | 45.0 | 80.4 | 51.0 | 35.7 | 43.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.206 | 3.951 | 17.411 | 19.123 | 39.936 | 49.559 | 49.289 | 53.185 | 63.419 | 59.637 |
| orientation_penalty | -0.887 | -0.813 | -1.566 | -2.778 | -1.489 | -1.287 | -1.376 | -1.456 | -1.851 | -1.246 |
| progress | 0.277 | 0.283 | 0.569 | 0.570 | 0.680 | 0.690 | 0.686 | 0.694 | 0.670 | 0.673 |
| velocity_penalty | -0.507 | -0.402 | -0.269 | -0.270 | -0.230 | -0.212 | -0.225 | -0.237 | -0.276 | -0.241 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.4% | 3.5% | 8.4% | 7.9% | 13.1% | 14.8% | 16.0% | 15.4% | 18.5% | 18.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% | 99.9% | 99.8% | 100.0% | 99.9% | 99.9% |
| velocity_penalty | 100.0% | 99.9% | 99.9% | 100.0% | 99.7% | 99.9% | 99.7% | 100.0% | 99.9% | 99.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward *increase* in leg contact, not the absolute contact state
    prev_contact = float(obs[6] + obs[7])
    next_contact = float(next_obs[6] + next_obs[7])
    contact_improvement = max(0.0, next_contact - prev_contact)   # only positive changes

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_improvement * proximity_factor * velocity_ok

    # combine all components
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
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