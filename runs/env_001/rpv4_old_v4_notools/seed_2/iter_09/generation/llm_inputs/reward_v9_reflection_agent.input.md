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

# 上一轮训练反馈 (iter_8, score=78.110493)
# Training Feedback

## Final-policy outcome
score=78.110493, len=896.050000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[13.296212, 264.246128]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 681.500000 | 92.3% | 92.3% | 13.2% |
| stable_contact_bonus | 40.930000 | 5.5% | 5.5% | 45.7% |
| progress | 10.360270 | 1.4% | 1.7% | 100.0% |
| orientation_penalty | -1.709633 | -0.2% | 0.2% | 100.0% |
| velocity_penalty | -1.014798 | -0.1% | 0.1% | 100.0% |

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
| -40.5 | -38.0 | -14.0 | 78.4 | -1.5 | -5.6 | 110.2 | 56.3 | 89.2 | 88.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 9.250 | 9.500 | 10.250 | 11.250 | 236.250 | 170.000 | 232.250 | 382.000 | 530.750 | 548.500 |
| orientation_penalty | -0.639 | -0.880 | -1.151 | -1.460 | -1.384 | -1.556 | -1.864 | -2.798 | -2.206 | -2.142 |
| progress | 8.541 | 9.781 | 9.691 | 9.299 | 9.845 | 9.799 | 9.604 | 9.461 | 9.503 | 9.731 |
| stable_contact_bonus | 0.445 | 0.070 | 4.400 | 16.740 | 17.460 | 10.250 | 26.210 | 28.190 | 35.420 | 35.680 |
| velocity_penalty | -2.413 | -2.043 | -1.935 | -1.706 | -1.627 | -1.315 | -1.406 | -1.426 | -1.196 | -1.061 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.7% | 1.4% | 0.8% | 0.6% | 6.0% | 3.9% | 7.5% | 9.4% | 11.0% | 11.7% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 99.4% | 100.0% | 100.0% | 99.4% | 100.0% | 100.0% | 99.4% | 99.6% | 99.9% | 99.9% |
| stable_contact_bonus | 4.8% | 0.7% | 21.8% | 44.5% | 25.3% | 13.3% | 46.1% | 39.3% | 41.5% | 42.9% |
| velocity_penalty | 99.2% | 100.0% | 99.9% | 99.3% | 100.0% | 100.0% | 99.3% | 99.6% | 99.9% | 99.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (used for progress shaping, not for penalty)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- progress: strong shaping towards target
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty: penalize high speed to encourage safe landing
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty: discourage large tilt
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact event reward: bonus for newly touching a leg to the platform
    leg_left_prev = obs[6]
    leg_right_prev = obs[7]
    leg_left_next = next_obs[6]
    leg_right_next = next_obs[7]

    contact_event = 0.0
    if leg_left_next > 0.5 and leg_left_prev < 0.5:
        contact_event += 1.0
    if leg_right_next > 0.5 and leg_right_prev < 0.5:
        contact_event += 1.0
    contact_bonus = 5.0 * contact_event

    # --- stable contact reward: small per-step reward for maintaining contact with low vertical speed
    # Keep it small to avoid encouraging unnecessary lingering on the platform.
    contact_active = leg_left_next + leg_right_next
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.1   # reduced further to discourage hovering

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + contact_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_bonus": contact_bonus,
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