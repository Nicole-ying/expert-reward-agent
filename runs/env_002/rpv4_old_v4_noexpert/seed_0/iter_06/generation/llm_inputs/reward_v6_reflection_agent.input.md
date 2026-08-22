# Duplicate reward retry
The previous generation duplicated iter 4 (runs/env_002/rpv4_old_v4_noexpert/seed_0/iter_04/generation/reward_v4.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1: revert iter_5's slope micro-tune (0.62 -> 0.60).
    # iter_4 with tanh(0.60*v) produced the best score so far (298.56).
    # iter_5's only change was the slope, and the score regressed to 254.16
    # with LOWER forward ep_sum (565 vs 587), i.e. the intended speed-up did not
    # materialize. Restore the proven concave saturation operating point.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep the proven asymmetric backward penalty: backward motion is
        # punished harder than forward motion is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1/iter_4's proven coefficients:
    # they were not the exploit source, and changing them risks trading away
    # forward progress (as seen in iter_3).
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components
```

下面是环境描述：
本环境是一个匿名 2D 双足地形行走任务。主目标是控制双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时保持稳定步态，不发生摔倒。次目标是最小化能量消耗，并保持髋、膝关节动作协调平滑。  
需要注意：该任务不是稀疏导航到唯一目标点的任务，也不是纯粹的平衡存活任务；能量消耗和动作平滑是辅助目标，不应与“向前行走”这一主目标混合成多目标问题。

目标得分：300

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

# 上一轮训练反馈 (iter_5, score=254.155255)
# Training Feedback

## Final-policy outcome
score=254.155255, len=1201.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-2.402510, 299.153658]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 565.666914 | 96.5% | 96.5% | 100.0% |
| torso_stability | -19.767908 | -3.4% | 3.4% | 100.0% |
| vertical_velocity_penalty | -0.470975 | -0.1% | 0.1% | 100.0% |

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
| -98.5 | -98.8 | -53.4 | 60.2 | 241.0 | 83.9 | -26.9 | 266.8 | 173.0 | 247.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| forward_progress | 8.578 | 9.121 | 0.721 | 192.241 | 529.212 | 302.349 | 139.672 | 584.936 | 468.834 | 547.598 |
| torso_stability | -1.624 | -2.076 | -0.808 | -8.916 | -30.603 | -19.972 | -5.264 | -17.062 | -9.410 | -17.221 |
| vertical_velocity_penalty | -0.110 | -0.103 | -0.047 | -0.267 | -0.779 | -0.425 | -0.199 | -0.529 | -0.345 | -0.419 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| forward_progress | 100.0% | 100.0% | 96.3% | 98.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| torso_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vertical_velocity_penalty | 100.0% | 99.9% | 7.6% | 62.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_4's proven concave tanh saturation, raise the
    # moderate-speed slope 0.6 -> 0.62 while keeping the 2.0 ceiling unchanged.
    # This nudges forward reward up ~2-3% in the mid-speed range (the operating
    # point) to push the mean score from 298.56 over 300, without re-opening the
    # high-speed farming channel that crashed iter_2.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.62 * v) = (e^(1.24*v) - 1) / (e^(1.24*v) + 1)
        e2v = 2.718281828 ** (1.24 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep iter_1's proven asymmetric backward penalty: moving backward is
        # punished harder than moving forward is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1/iter_4's proven coefficients:
    # they were not the exploit source, and changing them risks trading away
    # forward progress (as seen in iter_3) or destabilizing the gait.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
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

现在开始调查，做出一个语义方向上的修改。

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