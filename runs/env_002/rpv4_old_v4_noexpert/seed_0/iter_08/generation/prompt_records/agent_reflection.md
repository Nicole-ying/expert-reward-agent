# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的数据和原则，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action
- `read_training_feedback(N)` — 某轮训练的详细反馈：分数、每个组件的 ep_sum/active_rate/magnitude_share
- `read_reward_code(N)` — 某轮的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某组件在所有迭代中的 ep_sum_mean 和 active_rate 演化
- `read_environment_card()` — 环境信息：任务目标、观察空间（8维）、动作空间（4动作）、终止条件
- `read_checkpoint_trend(N)` — 某轮训练中各组件在不同训练阶段的趋势

# 修改层级

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态。常见模式参考：

| 证据模式 | 参考变换 | 验证标准 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | 稀疏→连续过程证据 | active_rate 改善，不产生 proxy 徘徊 |
| 极端值支配奖励 | 无界→归一化有界 | 极端轨迹支配下降，方差下降 |
| 占据好状态即可持续获奖 | 状态值→改善量/势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | 全局→阶段门控 | 早期探索与局部约束同时改善 |
| 多个小因子相乘导致塌缩 | 乘积→加性/几何平均 | 非零反馈增多，联合约束保留 |
| 持续事件被重复领取 | 持续→状态转移事件 | 重复积累下降，外部完成保持 |
| proxy 提高但外部任务不升 | 代理→任务完成对齐 | proxy 与外部分数重新同向 |
| 稠密 proxy 形成中分平台 | 全程 proxy→局部/转移任务信号 | 刷新 best，完成相关行为增加 |

- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。

# 关键规则

- **避免打转**：如果 `read_past_reflection` 显示之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先 L1**：如果当前得分或 best 得分接近目标（差距 < 10%），优先 L1 微调。因为之前的经验发现已多次出现分数接近目标时做 L2，结果越改越差。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

```

## User Prompt

```markdown
# Duplicate reward retry
The previous generation duplicated iter 6 (runs/env_002/rpv4_old_v4_noexpert/seed_0/iter_06/generation/reward_v6.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Revert iter_7's harmful L1 change (forward amplitude 2.05 -> 2.0).
    # iter_6 with amplitude 2.0, slope 0.60, and energy_penalty achieved the
    # best score 299.43. iter_7's +2.5% amplitude re-opened proxy farming under
    # saturation: forward ep_sum fell 593.6 -> 556.8, energy penalty worsened,
    # and the training trend collapsed at cp90. Restore the proven balance.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_6's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # Keep iter_6's torque-energy regularization unchanged.
    energy_penalty = -0.02 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = (
        forward_progress
        + torso_stability
        + vertical_velocity_penalty
        + energy_penalty
    )

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
        "energy_penalty": energy_penalty,
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

# 上一轮训练反馈 (iter_7, score=238.545058)
# Training Feedback

## Final-policy outcome
score=238.545058, len=1189.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-45.196538, 291.157353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 556.756231 | 90.5% | 90.5% | 100.0% |
| energy_penalty | -35.573389 | -5.8% | 5.8% | 100.0% |
| torso_stability | -22.434003 | -3.6% | 3.6% | 100.0% |
| vertical_velocity_penalty | -0.560140 | -0.1% | 0.1% | 100.0% |

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
| -95.6 | -96.4 | -89.1 | -88.1 | -45.6 | -11.0 | 80.8 | 86.7 | 26.1 | 198.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| energy_penalty | -0.149 | -0.305 | -0.932 | -1.153 | -7.348 | -12.122 | -17.705 | -18.410 | -15.954 | -31.448 |
| forward_progress | 13.490 | 13.255 | 28.052 | 30.903 | 126.210 | 193.312 | 335.144 | 342.160 | 263.315 | 487.262 |
| torso_stability | -1.145 | -1.314 | -0.768 | -2.709 | -6.167 | -7.627 | -12.843 | -13.159 | -11.207 | -19.764 |
| vertical_velocity_penalty | -0.094 | -0.108 | -0.076 | -0.065 | -0.284 | -0.435 | -0.465 | -0.532 | -0.488 | -0.568 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| energy_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| forward_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| torso_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vertical_velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_6's proven concave tanh saturation and the
    # energy_penalty that drove the 254 -> 299 improvement. Only scale the
    # forward amplitude 2.0 -> 2.05 (+2.5%) to nudge the mean score from
    # 299.43 over the 300 target, without re-opening the high-speed farming
    # channel (tanh still saturates at high speed).
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.05 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_4/iter_6's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # Keep iter_6's torque-energy regularization unchanged: it was the key
    # improvement this round (smooth gait, suppressed high-frequency control).
    energy_penalty = -0.02 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = (
        forward_progress
        + torso_stability
        + vertical_velocity_penalty
        + energy_penalty
    )

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
        "energy_penalty": energy_penalty,
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
```
